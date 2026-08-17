"""GPU JAXFMM demagnetizing-field backend for SpinFEMx.

This backend targets ``jaxfmm >= 0.3.3`` and is intended for one MPI rank
and one NVIDIA GPU. Repeated field evaluations stay on the device: PETSc CUDA
vectors are exposed to CuPy and JAX through DLPack without a host round trip.

For large tetrahedral meshes, the exterior triangular surface is extracted by
DOLFINx on the CPU and passed explicitly to JAXFMM. This avoids JAXFMM's
``get_tris`` path, which otherwise materializes four candidate faces per
tetrahedron on the GPU and can create a very large initialization-time memory
peak.
"""

from __future__ import annotations

import gc
import os
from typing import Any

# These variables must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from petsc4py import PETSc

try:
    from jaxfmm.apps.mag.strayfield import plan_strayfield
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The GPU FMM demag backend requires jaxFMM >= 0.3.3. "
        "Install the pinned jaxfmm package before using method='fmm'."
    ) from exc

jax.config.update("jax_enable_x64", True)


# The old SpinFEMx wrappers supplied these values under the semantics of the
# historical JAXFMM branch. In jaxfmm >= 0.3.1, mem_limit is a byte budget.
_LEGACY_MEM_LIMITS = {2_000_000, 4_000_000}
_DEFAULT_BATCH_BUDGET_BYTES = 512 * 1024**2


def cupy_to_jax(a_cp: cp.ndarray):
    """Convert a CuPy array to a JAX array through DLPack."""
    try:
        return jax.dlpack.from_dlpack(a_cp)
    except TypeError:  # Compatibility with older CuPy/JAX DLPack APIs.
        return jax.dlpack.from_dlpack(a_cp.toDlpack())


def jax_to_cupy(a_jax) -> cp.ndarray:
    """Convert a JAX array to a CuPy array through DLPack."""
    try:
        return cp.from_dlpack(a_jax)
    except TypeError:  # Compatibility with older CuPy/JAX DLPack APIs.
        return cp.from_dlpack(jax.dlpack.to_dlpack(a_jax))


def _release_unused_cupy_memory() -> None:
    """Return unused CuPy allocations to CUDA without touching live arrays."""
    cp.cuda.get_current_stream().synchronize()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _fix_tet_orientations_numpy(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    batch_size: int = 250_000,
) -> np.ndarray:
    """Orient tetrahedra on the CPU using the JAXFMM convention.

    JAXFMM's ``fix_tet_orientations`` swaps the first two vertices of every
    tetrahedron with negative signed volume. Performing the same operation in
    NumPy avoids creating an initialization-sized JAX/GPU temporary for large
    meshes.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    oriented = np.ascontiguousarray(tetrahedra, dtype=np.int64).copy()

    if oriented.ndim != 2 or oriented.shape[1] != 4:
        raise ValueError(
            "The JAXFMM backend requires tetrahedral connectivity with shape "
            f"(N, 4); received {oriented.shape}."
        )
    if batch_size <= 0:
        raise ValueError("orientation_batch_size must be positive.")

    n_tets = int(oriented.shape[0])
    for first in range(0, n_tets, int(batch_size)):
        last = min(first + int(batch_size), n_tets)
        block = oriented[first:last]

        p0 = vertices[block[:, 0]]
        p1 = vertices[block[:, 1]]
        p2 = vertices[block[:, 2]]
        p3 = vertices[block[:, 3]]

        signed_volume6 = np.einsum(
            "ij,ij->i",
            np.cross(p1 - p0, p2 - p0),
            p3 - p0,
            optimize=True,
        )

        if not np.all(np.isfinite(signed_volume6)):
            raise ValueError("The tetrahedral mesh contains non-finite geometry.")
        if np.any(signed_volume6 == 0.0):
            count = int(np.count_nonzero(signed_volume6 == 0.0))
            raise ValueError(
                f"The tetrahedral mesh contains {count} exactly degenerate "
                "tetrahedron/tetrahedra."
            )

        negative = signed_volume6 < 0.0
        if np.any(negative):
            rows = np.flatnonzero(negative)
            tmp = block[rows, 0].copy()
            block[rows, 0] = block[rows, 1]
            block[rows, 1] = tmp

    return oriented


def _extract_oriented_surface_triangles(
    domain_mesh,
    tetrahedra: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Extract outward-oriented exterior triangles with DOLFINx.

    The returned triangle indices reference ``domain_mesh.geometry.x`` and use
    JAXFMM's normal convention

        n = (v0 - v2) x (v1 - v2).

    SpinFEMx currently supplies a scalar saturation magnetization, so only the
    external material boundary is required; internal Ms discontinuities are not
    represented by this helper.
    """
    topology = domain_mesh.topology
    tdim = int(topology.dim)
    if tdim != 3:
        raise ValueError(
            "DemagFieldFMMJAXGPU requires a three-dimensional tetrahedral mesh."
        )

    fdim = tdim - 1
    topology.create_entities(fdim)
    topology.create_connectivity(fdim, tdim)
    topology.create_connectivity(fdim, 0)

    boundary_facets = np.asarray(
        dmesh.exterior_facet_indices(topology),
        dtype=np.int32,
    )
    if boundary_facets.size == 0:
        raise RuntimeError("No exterior facets were found in the mesh.")

    # For first-order tetrahedra this returns three geometry-node indices per
    # triangular facet, already referencing mesh.geometry.x.
    triangles = np.asarray(
        dmesh.entities_to_geometry(
            domain_mesh,
            fdim,
            boundary_facets,
            permute=False,
        ),
        dtype=np.int32,
    )
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "The current JAXFMM interface supports first-order triangular "
            "boundary facets only; entities_to_geometry returned shape "
            f"{triangles.shape}."
        )
    triangles = np.ascontiguousarray(triangles, dtype=np.int64)

    facet_to_cell = topology.connectivity(fdim, tdim)
    if facet_to_cell is None:
        raise RuntimeError("Missing facet-to-cell connectivity.")

    adjacent_cells = np.empty(boundary_facets.size, dtype=np.int32)
    for i, facet in enumerate(boundary_facets):
        links = np.asarray(facet_to_cell.links(int(facet)), dtype=np.int32)
        if links.size != 1:
            raise RuntimeError(
                "An exterior facet must have exactly one adjacent cell; "
                f"facet {int(facet)} has {links.size}."
            )
        adjacent_cells[i] = links[0]

    cell_vertices = tetrahedra[adjacent_cells]

    # Identify the tetrahedral vertex opposite each exterior triangle.
    opposite_mask = np.ones(cell_vertices.shape, dtype=bool)
    for local_vertex in range(3):
        opposite_mask &= cell_vertices != triangles[:, local_vertex, None]

    opposite_count = opposite_mask.sum(axis=1)
    if np.any(opposite_count != 1):
        bad = int(np.flatnonzero(opposite_count != 1)[0])
        raise RuntimeError(
            "Could not identify the unique tetrahedral vertex opposite an "
            f"exterior facet (row {bad})."
        )

    opposite = cell_vertices[
        np.arange(cell_vertices.shape[0]),
        np.argmax(opposite_mask, axis=1),
    ]

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v0 - v2, v1 - v2)
    normal_norm = np.linalg.norm(normals, axis=1)
    if np.any(normal_norm == 0.0):
        count = int(np.count_nonzero(normal_norm == 0.0))
        raise ValueError(
            f"The exterior surface contains {count} degenerate triangle(s)."
        )

    centroid = (v0 + v1 + v2) / 3.0
    toward_interior = vertices[opposite] - centroid

    # An outward normal has a negative dot product with the vector pointing
    # from the boundary face toward the tetrahedron's interior.
    inward = np.einsum(
        "ij,ij->i",
        normals,
        toward_interior,
        optimize=True,
    ) > 0.0

    if np.any(inward):
        rows = np.flatnonzero(inward)
        tmp = triangles[rows, 0].copy()
        triangles[rows, 0] = triangles[rows, 1]
        triangles[rows, 1] = tmp

    return np.ascontiguousarray(triangles, dtype=np.int64)


def _normalize_surface_ms(
    Ms_surf: Any,
    *,
    Ms: float,
    n_triangles: int,
) -> np.ndarray:
    """Return scalar or per-triangle surface saturation magnetization."""
    if Ms_surf is None:
        return np.asarray([Ms], dtype=np.float64)

    values = np.asarray(Ms_surf, dtype=np.float64).reshape(-1)
    if values.size not in (1, n_triangles):
        raise ValueError(
            "Ms_surf must be scalar or contain one value per surface triangle; "
            f"received {values.size} values for {n_triangles} triangles."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Ms_surf contains non-finite values.")
    return np.ascontiguousarray(values, dtype=np.float64)


class DemagFieldFMMJAXGPU:
    """Demagnetizing field evaluated with JAXFMM on one GPU/MPI rank.

    Parameters specific to this wrapper
    -----------------------------------
    tris:
        Optional outward-oriented boundary triangles referencing
        ``mesh.geometry.x``. If omitted, DOLFINx extracts them on the CPU.
    Ms_surf:
        Scalar or per-triangle surface saturation magnetization. Defaults to
        the volume ``Ms`` used by SpinFEMx.
    mem_limit:
        JAXFMM batch-memory budget in bytes. Historical SpinFEMx defaults of
        2,000,000 and 4,000,000 are replaced by a conservative 512 MiB budget.
    orientation_batch_size:
        Number of tetrahedra processed per CPU orientation batch.
    release_cupy_pool_before_plan:
        Return unused CuPy blocks before JAXFMM creates its geometry plan.
    """

    def __init__(
        self,
        domain_mesh,
        V,
        V1,
        Ms,
        VolN,
        mem_limit=None,
        err_tol=1.0e-2,
        field_mode="grad",
        jaxfmm_verbose=False,
        tris=None,
        Ms_surf=None,
        orientation_batch_size=250_000,
        release_cupy_pool_before_plan=True,
        **jaxfmm_kwargs,
    ):
        self.mesh = domain_mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * np.pi * 1.0e-7
        self.comm = self.mesh.comm

        if self.comm.size != 1:
            raise RuntimeError(
                "DemagFieldFMMJAXGPU currently supports one MPI rank only."
            )
        if float(err_tol) <= 0.0:
            raise ValueError("err_tol must be positive.")

        requested_mem_limit = mem_limit
        if mem_limit is not None:
            mem_limit = int(mem_limit)
            if mem_limit <= 0:
                raise ValueError("mem_limit must be a positive byte count or None.")
            if mem_limit in _LEGACY_MEM_LIMITS:
                # Drop-in compatibility with llg_*_GPU.py versions that still
                # inject 4_000_000 when the user specifies no memory budget.
                mem_limit = _DEFAULT_BATCH_BUDGET_BYTES

        if mem_limit is not None:
            jaxfmm_kwargs.setdefault("mem_limit", int(mem_limit))

        self.H_d = fem.Function(self.V, name="H_demag_fmm")

        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = int(self.end - self.start)
        self.local_size = 3 * self.local_dofs

        vol = np.asarray(
            VolN[: self.local_size],
            dtype=np.float64,
        ).reshape((-1, 3))
        self.vol_nodes_cp = cp.asarray(vol[:, 0])

        self.X = np.ascontiguousarray(
            self.mesh.geometry.x,
            dtype=np.float64,
        )

        tdim = int(self.mesh.topology.dim)
        if tdim != 3:
            raise ValueError(
                "DemagFieldFMMJAXGPU requires a three-dimensional mesh."
            )

        cell_map = self.mesh.topology.index_map(tdim)
        if cell_map is None:
            raise RuntimeError("The mesh has no cell index map.")
        n_cells = int(cell_map.size_local)
        if n_cells == 0:
            raise RuntimeError("The mesh contains no local tetrahedra.")

        geometry_dofmap = np.asarray(
            self.mesh.geometry.dofmap[:n_cells],
            dtype=np.int64,
        )
        if geometry_dofmap.ndim != 2 or geometry_dofmap.shape[1] != 4:
            raise ValueError(
                "The JAXFMM backend requires first-order tetrahedral geometry "
                f"with four geometry nodes per cell; received "
                f"{geometry_dofmap.shape}."
            )

        self.cells = _fix_tet_orientations_numpy(
            self.X,
            geometry_dofmap,
            batch_size=int(orientation_batch_size),
        )

        if tris is None:
            self.tris = _extract_oriented_surface_triangles(
                self.mesh,
                self.cells,
                self.X,
            )
            surface_source = "dolfinx"
        else:
            self.tris = np.ascontiguousarray(tris, dtype=np.int64)
            if self.tris.ndim != 2 or self.tris.shape[1] != 3:
                raise ValueError(
                    "tris must have shape (N_surface_triangles, 3); received "
                    f"{self.tris.shape}."
                )
            if self.tris.size and (
                self.tris.min() < 0 or self.tris.max() >= self.X.shape[0]
            ):
                raise ValueError("tris contains vertex indices outside mesh.geometry.x.")
            surface_source = "provided"

        self.Ms_surf = _normalize_surface_ms(
            Ms_surf,
            Ms=self.Ms,
            n_triangles=int(self.tris.shape[0]),
        )

        if release_cupy_pool_before_plan:
            _release_unused_cupy_memory()

        if jaxfmm_verbose:
            budget_text = (
                "jaxfmm-default"
                if mem_limit is None
                else f"{mem_limit / 1024**2:.1f} MiB"
            )
            print(
                "[Demag JAXFMM GPU] "
                f"vertices={self.X.shape[0]:,} | "
                f"tets={self.cells.shape[0]:,} | "
                f"surface tris={self.tris.shape[0]:,} ({surface_source}) | "
                f"mem_limit={budget_text}",
                flush=True,
            )

        verts_jax = jnp.asarray(self.X, dtype=jnp.float64)

        # JAXFMM 0.3.3 uses ``jnp.iinfo(connectivity.dtype).max`` directly as
        # the padding value in its batched correction/gradient routines. With
        # x64 enabled, that Python integer is materialized as int64. Therefore
        # the connectivity must also be int64; int32 connectivity raises
        # ``pad operand and padding_value must be same dtype`` inside lax.pad.
        tets_jax = jnp.asarray(self.cells, dtype=jnp.int64)
        tris_jax = jnp.asarray(self.tris, dtype=jnp.int64)
        Ms_jax = jnp.asarray([self.Ms], dtype=jnp.float64)
        Ms_surf_jax = jnp.asarray(self.Ms_surf, dtype=jnp.float64)

        try:
            self._eval_strayfield = plan_strayfield(
                verts_jax,
                tets=tets_jax,
                Ms=Ms_jax,
                tris=tris_jax,
                Ms_surf=Ms_surf_jax,
                err_tol=float(err_tol),
                field_mode=str(field_mode),
                verbose=bool(jaxfmm_verbose),
                **jaxfmm_kwargs,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "resource_exhausted" in message:
                raise RuntimeError(
                    "JAXFMM ran out of GPU memory while creating the frozen "
                    "stray-field plan. The exterior surface was already passed "
                    "explicitly, so try a smaller byte budget, for example "
                    "mem_limit=256*1024**2, close other GPU processes, or use "
                    "a GPU with more free memory."
                ) from exc
            raise

        self.info = dict(getattr(self._eval_strayfield, "info", {}))
        self.info.update(
            {
                "backend": "jaxfmm-gpu",
                "vertices": int(self.X.shape[0]),
                "tetrahedra": int(self.cells.shape[0]),
                "surface_triangles": int(self.tris.shape[0]),
                "surface_source": surface_source,
                "requested_mem_limit": requested_mem_limit,
                "effective_mem_limit": mem_limit,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "connectivity_dtype": "int64",
            }
        )
        self.H_gpu = None

    def compute_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """Evaluate ``H_demag`` into a PETSc CUDA output vector."""
        m_cp_all = cp.from_dlpack(m_vec.toDLPack("r"))
        out_cp_all = cp.from_dlpack(out_vec.toDLPack("rw"))

        m_cp = m_cp_all[: self.local_size].reshape((-1, 3))
        out_cp = out_cp_all[: self.local_size].reshape((-1, 3))

        H_jax = self._eval_strayfield(cupy_to_jax(m_cp))
        H_cp = jax_to_cupy(H_jax)

        if H_cp.shape != out_cp.shape:
            raise RuntimeError(
                "JAXFMM returned an unexpected field shape: "
                f"expected {out_cp.shape}, received {H_cp.shape}."
            )

        out_cp[:, :] = H_cp

        if out_cp_all.size > self.local_size:
            out_cp_all[self.local_size :] = 0.0

        self.H_gpu = out_vec
        return out_vec

    def copy_to_function(self, H_vec: PETSc.Vec):
        """Copy a PETSc CUDA field vector to the host-side DOLFINx Function."""
        H_vec.copy(self.H_d.x.petsc_vec)
        self.H_d.x.scatter_forward()
        return self.H_d

    def Energy_lumped_gpu(self, m_vec: PETSc.Vec, H_vec: PETSc.Vec):
        """Return the lumped demagnetizing energy in joules."""
        m_cp_all = cp.from_dlpack(m_vec.toDLPack("r"))
        H_cp_all = cp.from_dlpack(H_vec.toDLPack("r"))

        m_cp = m_cp_all[: self.local_size].reshape((-1, 3))
        H_cp = H_cp_all[: self.local_size].reshape((-1, 3))

        mdH = cp.sum(m_cp * H_cp, axis=1)
        value = cp.sum(self.vol_nodes_cp * mdH)

        # Mesh coordinates/volumes are in nm/nm^3.
        return float((-0.5 * self.mu0 * self.Ms * value * 1.0e-27).item())

    def Energy(self, m_fun):
        """Host-side FEM energy using the most recently copied field."""
        self.H_d.x.scatter_forward()
        integrand = ufl.inner(m_fun, self.H_d) * ufl.dx(domain=self.mesh)
        local_integral = fem.assemble_scalar(fem.form(integrand))
        return -0.5 * self.mu0 * self.Ms * float(local_integral) * 1.0e-27
