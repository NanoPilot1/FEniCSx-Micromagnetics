import gc
import math
import os
from pathlib import Path

import numpy as np
import cupy as cp
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem.petsc import assemble_matrix

try:
    from .Demag_Lindholm import (
        DoubleLayerDenseOwnedRowsMPI,
        build_nullspace_per_component,
    )
except ImportError:
    from Demag_Lindholm import (
        DoubleLayerDenseOwnedRowsMPI,
        build_nullspace_per_component,
    )

try:
    from .lindholm_hmatrix_gpu.backend_fused import HMatrixGPUPackedFused
    from .lindholm_hmatrix_gpu.builder import (
        HMatrixBuildConfig,
        build_or_load_hmatrix,
        print_hmatrix_summary,
    )
    from .lindholm_hmatrix_gpu.entry_provider import LindholmEntryProviderCPU
    from .lindholm_hmatrix_gpu.entry_provider_gpu import LindholmEntryProviderGPU
except ImportError as exc:
    raise ImportError(
    "Could not import the compressed Lindholm GPU backend located in "
    "'fields/lindholm_hmatrix_gpu'. Verify that the folder exists and "
    "contains its '__init__.py' file."
    ) from exc


def _set_vec_cuda(vec: PETSc.Vec, block_size=None):
    try:
        vec.setType(PETSc.Vec.Type.CUDA)
    except Exception:
        pass

    if block_size is not None:
        try:
            vec.setBlockSize(block_size)
        except Exception:
            pass

    try:
        vec.bindToCPU(False)
    except Exception:
        pass

    return vec


def _set_mat_cuda(mat: PETSc.Mat):
    try:
        mat = mat.convert(PETSc.Mat.Type.AIJCUSPARSE)
    except Exception:
        mat = mat.convert("aijcusparse")

    try:
        mat.bindToCPU(False)
    except Exception:
        pass

    return mat


def _vec_to_cupy(vec: PETSc.Vec, mode="rw"):
    return cp.from_dlpack(vec.toDLPack(mode))


class DemagFieldLindholmGPU:
    """
    GPU-oriented FEM/BEM demagnetizing field using the Lindholm double-layer
    operator.

    Notes
    -----
    - Matrices are assembled with DOLFINx on the host.
    - FEM matrices are converted to PETSc AIJCUSPARSE.
    - FEM matrices are converted to PETSc AIJCUSPARSE and remain on GPU.
    - The Lindholm boundary operator can use:
        * dense: the original dense GPU matrix;
        * hmatrix: compressed near-field CSR + packed low-rank CUDA kernel;
        * auto: dense below a configurable memory threshold, hmatrix otherwise.
    - The per-step B @ u1_boundary multiplication remains entirely on GPU.
    - The implementation is intended first for single-GPU / one MPI rank.
      MPI support is possible but needs careful treatment of boundary reductions.
    """

    def __init__(
        self,
        mesh,
        V,
        V1,
        Ms,
        VolN,
        serial_mesh_path=None,
        cache_dir=None,
        cache_prefix="Bcache_lindholm_ownedrows",
        cache_tag=None,
        map_tol=1e-10,
        ksp_rtol_u1=1e-6,
        ksp_rtol_u2=1e-6,
        incident_policy="skip",
        volume_scale=1e27,
        B_dtype=np.float64,
        boundary_backend="dense",
        dense_gpu_max_bytes=512 * 1024**2,
        hmatrix_cache_path=None,
        hmatrix_cache_dir="./hmatrix_cache",
        hmatrix_epsilon=1e-5,
        hmatrix_eta=2.0,
        hmatrix_leaf_size=64,
        hmatrix_compressor="fullaca",
        hmatrix_max_rank=None,
        hmatrix_max_temporary_block_bytes=256 * 1024**2,
        hmatrix_force_rebuild=False,
        hmatrix_cache_compressed=False,
        # H-matrix construction backend.  The boundary MatVec remains on GPU
        # independently of this option.
        hmatrix_build_backend="gpu",
        hmatrix_build_device_id=0,
        hmatrix_build_threads_per_block=128,
        hmatrix_gpu_entry_kernel="auto",
        hmatrix_gpu_fast_math=False,
        hmatrix_gpu_index_cache_max_bytes=128 * 1024**2,
        hmatrix_gpu_memory_fraction=0.75,
        hmatrix_gpu_fallback_to_cpu=True,
        hmatrix_gpu_aca_residual_check_interval=1,
        hmatrix_gpu_rsvd_initial_rank=16,
        hmatrix_gpu_rsvd_oversampling=8,
        hmatrix_gpu_rsvd_power_iterations=1,
        hmatrix_gpu_progress_interval_seconds=5.0,
        # Packed far-field MatVec settings.
        hmatrix_threads_per_block=128,
        hmatrix_use_fp32=True,
        # Stable default: discard transient per-block GPU factors and create
        # the final packed arrays from the already available host cache copy.
        # This preserves GPU construction while avoiding retained build-pool
        # memory before temporal integration.
        hmatrix_direct_gpu_pack=False,
        hmatrix_release_build_memory=True,
        hmatrix_verbose=False,
    ):
        self.mesh = mesh
        self.V = V
        self.V1 = V1
        self.Ms = float(Ms)
        self.mu0 = 4.0 * math.pi * 1e-7
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.volume_scale = float(volume_scale)
        self.last_its_u1 = 1
        self.last_its_u2 = 1
        if self.comm.size != 1:
            raise RuntimeError(
                "DemagFieldLindholmGPU is currently implemented for single-rank GPU execution only."
            )

        self.local_dofs = V.dofmap.index_map.size_local
        self.local_size = 3 * self.local_dofs
        self.B_dtype = B_dtype



        # Scalar nodal volume from vector VolN.
        VolN = np.asarray(VolN, dtype=np.float64)
        self.vol_scalar_gpu = cp.asarray(
            VolN[: self.local_size].reshape((-1, 3))[:, 0]
        )

        self.inv_vol_scalar_gpu = 1.0 / self.vol_scalar_gpu

        # ------------------------------------------------------------
        # u1 problem: A_u1 u1 = Div m
        # ------------------------------------------------------------
        v1 = ufl.TestFunction(V1)
        u1 = ufl.TrialFunction(V1)

        a_u1 = fem.form(ufl.inner(ufl.grad(v1), ufl.grad(u1)) * ufl.dx)
        A_u1_cpu = assemble_matrix(a_u1)
        A_u1_cpu.assemble()

        self.ns_u1 = build_nullspace_per_component(mesh, V1, A_u1_cpu)

        self.A_u1 = _set_mat_cuda(A_u1_cpu)

        v2 = ufl.TestFunction(V1)
        m_trial = ufl.TrialFunction(V)

        Div_form = fem.form(
            self.Ms * ufl.inner(ufl.grad(v2), m_trial) * ufl.dx
        )

        Div_cpu = assemble_matrix(Div_form)
        Div_cpu.assemble()
        self.Div = _set_mat_cuda(Div_cpu)

        self.b_u1 = self.Div.createVecLeft()
        self.u1 = self.A_u1.createVecRight()

        _set_vec_cuda(self.b_u1)
        _set_vec_cuda(self.u1)

        self.ksp_u1 = PETSc.KSP().create(self.comm)
        self.ksp_u1.setType("cg")
        self.ksp_u1.setOperators(self.A_u1)
        self.ksp_u1.setTolerances(rtol=ksp_rtol_u1, atol=1e-12, max_it=2000)
        #self.ksp_u1.setInitialGuessNonzero(True)

        pc1 = self.ksp_u1.getPC()
        pc1.setType("gamg")
        pc1.setReusePreconditioner(True)    

        self.ksp_u1.setFromOptions()

        # ------------------------------------------------------------
        # u2 problem with Dirichlet data on boundary
        # ------------------------------------------------------------
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(fdim, 0)

        boundary_facets = exterior_facet_indices(mesh.topology)
        self.boundary_dofs = fem.locate_dofs_topological(V1, fdim, boundary_facets)
        self.boundary_dofs = np.unique(self.boundary_dofs).astype(np.int32)

        zero = fem.Function(V1)
        zero.x.array[:] = 0.0
        zero.x.scatter_forward()

        bc_zero = fem.dirichletbc(zero, self.boundary_dofs)

        uu = ufl.TrialFunction(V1)
        vv = ufl.TestFunction(V1)

        a_u2 = fem.form(ufl.inner(ufl.grad(uu), ufl.grad(vv)) * ufl.dx)

        # Raw matrix: used to form b = -A_raw * g.
        A_u2_raw_cpu = assemble_matrix(a_u2)
        A_u2_raw_cpu.assemble()

        # BC matrix: identity rows on boundary.
        A_u2_bc_cpu = assemble_matrix(a_u2, bcs=[bc_zero])
        A_u2_bc_cpu.assemble()

        self.A_u2_raw = _set_mat_cuda(A_u2_raw_cpu)
        self.A_u2 = _set_mat_cuda(A_u2_bc_cpu)

        self.g_u2 = self.A_u2.createVecRight()
        self.b_u2 = self.A_u2.createVecLeft()
        self.u2 = self.A_u2.createVecRight()

        _set_vec_cuda(self.g_u2)
        _set_vec_cuda(self.b_u2)
        _set_vec_cuda(self.u2)

        self.ksp_u2 = PETSc.KSP().create(self.comm)
        self.ksp_u2.setType("cg")
        self.ksp_u2.setOperators(self.A_u2)
        self.ksp_u2.setTolerances(rtol=ksp_rtol_u2, atol=1e-12, max_it=12000)
        

        pc2 = self.ksp_u2.getPC()
        pc2.setType("gamg")
        pc2.setReusePreconditioner(True)     

        self.ksp_u2.setFromOptions()

        # ------------------------------------------------------------
        # Lindholm B operator
        # ------------------------------------------------------------
        #
        # DoubleLayerDenseOwnedRowsMPI is reused only to construct boundary
        # geometry and maps. For the compressed backend, do NOT call assemble():
        # that method materializes the dense matrix and defeats the purpose of
        # the H-matrix representation.
        self.opB = DoubleLayerDenseOwnedRowsMPI(
            mesh_par=mesh,
            V1_par=V1,
            serial_mesh_path=serial_mesh_path,
            cache_dir=cache_dir,
            cache_prefix=cache_prefix,
            cache_tag=cache_tag,
            map_tol=map_tol,
            incident_policy=incident_policy,
        )

        if self.opB.nrows != self.opB.Nb:
            raise RuntimeError(
                "The current GPU Lindholm implementation expects one owned row "
                "per boundary node in single-rank execution."
            )

        expected_rows = np.arange(self.opB.Nb, dtype=np.int32)
        if not np.array_equal(self.opB.rows, expected_rows):
            raise RuntimeError(
                "Boundary rows are not the contiguous global ordering expected "
                "by the single-rank GPU backend."
            )

        self.opB.diag_jump = (
            self.opB.omega_sum / (4.0 * math.pi) - 1.0
        ).astype(np.float64)

        requested_backend = str(boundary_backend).strip().lower()
        if requested_backend not in {"dense", "hmatrix", "auto"}:
            raise ValueError(
                "boundary_backend must be 'dense', 'hmatrix', or 'auto'."
            )

        dense_theoretical_bytes = (
            int(self.opB.Nb) * int(self.opB.Nb) * np.dtype(self.B_dtype).itemsize
        )

        if requested_backend == "auto":
            self.boundary_backend = (
                "dense"
                if dense_theoretical_bytes <= int(dense_gpu_max_bytes)
                else "hmatrix"
            )
        else:
            self.boundary_backend = requested_backend

        self.B_dense_gpu = None
        self.B_hmatrix_cpu = None
        self.B_hmatrix_gpu = None
        self.hmatrix_cache_path = None

        if self.boundary_backend == "dense":
            self.opB.assemble(force=False)
            self.B_dense_gpu = cp.asarray(self.opB.Bloc, dtype=self.B_dtype)

        else:
            if np.dtype(self.B_dtype) != np.dtype(np.float64):
                raise TypeError(
                    "The packed fused H-matrix backend currently requires float64."
                )

            if hmatrix_cache_path is None:
                cache_root = Path(
                    os.environ.get("DEMAG_HMATRIX_CACHE_DIR", hmatrix_cache_dir)
                )
                cache_root.mkdir(parents=True, exist_ok=True)

                mesh_tag = (
                    str(cache_tag)
                    if cache_tag is not None
                    else Path(self.opB.serial_mesh_path).stem
                )

                eps_tag = f"{float(hmatrix_epsilon):.0e}".replace("+", "")
                eta_tag = f"{float(hmatrix_eta):g}".replace(".", "p")
                compressor_tag = (
                    str(hmatrix_compressor).strip().lower().replace("-", "_")
                )
                build_tag = str(hmatrix_build_backend).strip().lower()

                hmatrix_cache_path = cache_root / (
                    f"{mesh_tag}_eps{eps_tag}_eta{eta_tag}_"
                    f"leaf{int(hmatrix_leaf_size)}_{compressor_tag}_{build_tag}.npz"
                )

            self.hmatrix_cache_path = str(Path(hmatrix_cache_path))

            requested_build_backend = str(hmatrix_build_backend).strip().lower()
            if requested_build_backend == "cpu":
                provider = LindholmEntryProviderCPU.from_opB(self.opB)
            else:
                provider = LindholmEntryProviderGPU.from_opB(
                    self.opB,
                    device_id=int(hmatrix_build_device_id),
                    threads_per_block=int(hmatrix_build_threads_per_block),
                    kernel_mode=str(hmatrix_gpu_entry_kernel),
                    fast_math=bool(hmatrix_gpu_fast_math),
                    index_cache_max_bytes=int(hmatrix_gpu_index_cache_max_bytes),
                )

            config = HMatrixBuildConfig(
                epsilon=float(hmatrix_epsilon),
                eta=float(hmatrix_eta),
                leaf_size=int(hmatrix_leaf_size),
                compressor=str(hmatrix_compressor),
                max_rank=hmatrix_max_rank,
                max_temporary_block_bytes=int(
                    hmatrix_max_temporary_block_bytes
                ),
                build_backend=requested_build_backend,
                gpu_device_id=int(hmatrix_build_device_id),
                gpu_memory_fraction=float(hmatrix_gpu_memory_fraction),
                gpu_fallback_to_cpu=bool(hmatrix_gpu_fallback_to_cpu),
                gpu_aca_residual_check_interval=int(
                    hmatrix_gpu_aca_residual_check_interval
                ),
                gpu_rsvd_initial_rank=int(hmatrix_gpu_rsvd_initial_rank),
                gpu_rsvd_oversampling=int(hmatrix_gpu_rsvd_oversampling),
                gpu_rsvd_power_iterations=int(
                    hmatrix_gpu_rsvd_power_iterations
                ),
                gpu_progress_interval_seconds=float(
                    hmatrix_gpu_progress_interval_seconds
                ),
            )

            self.B_hmatrix_cpu = build_or_load_hmatrix(
                cache_path=self.hmatrix_cache_path,
                Xb=self.opB.Xb,
                diag_jump=self.opB.diag_jump,
                provider=provider,
                config=config,
                force_rebuild=bool(hmatrix_force_rebuild),
                verbose=bool(hmatrix_verbose),
                cache_compressed=bool(hmatrix_cache_compressed),
            )

            factor_precision = "FP32" if hmatrix_use_fp32 else "FP64"

            if hmatrix_verbose:
                print(
                    "[Demag Lindholm HMatrixGPU] "
                    f"low-rank factor precision={factor_precision}",
                    flush=True,
                )


            # The GPU builder stores one transient CuPy U/V pair per far
            # block in ``_gpu_far_blocks``.  Direct packing avoids one host to
            # device copy, but for large problems it can leave several GiB of
            # now-unused allocations cached in the CuPy pool immediately before
            # PETSc TS starts.  The host factors already exist because they are
            # required by the persistent cache, so the stable default reproduces
            # the historical packed layout without giving up GPU construction.
            if not bool(hmatrix_direct_gpu_pack):
                transient_blocks = getattr(
                    self.B_hmatrix_cpu,
                    "_gpu_far_blocks",
                    None,
                )
                if transient_blocks is not None:
                    cp.cuda.Device().synchronize()
                    transient_blocks.clear()
                    try:
                        delattr(self.B_hmatrix_cpu, "_gpu_far_blocks")
                    except AttributeError:
                        pass
                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()
                    try:
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass

            self.B_hmatrix_gpu = HMatrixGPUPackedFused(
                self.B_hmatrix_cpu,
                threads_per_block=int(hmatrix_threads_per_block),
                use_fp32=bool(hmatrix_use_fp32),
            )

            # Complete all packing work before any timing of the temporal
            # integrator and return free construction blocks to the CUDA driver.
            cp.cuda.Device().synchronize()
            if bool(hmatrix_release_build_memory):
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                try:
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            if hmatrix_verbose:
                print_hmatrix_summary(
                    self.B_hmatrix_cpu,
                    prefix="[Demag Lindholm HMatrixGPU]",
                )
                self.B_hmatrix_gpu.print_memory_report(
                    prefix="[Demag Lindholm HMatrixGPU]"
                )
            elif self.rank == 0:
                meta = self.B_hmatrix_cpu.metadata
                memory = self.B_hmatrix_gpu.memory_report()
                if meta.get("cache_loaded"):
                    status = "cache loaded"
                else:
                    backend = str(
                        meta.get("resolved_build_backend", "cpu")
                    ).upper()
                    elapsed = float(meta.get("build_time_seconds", 0.0))
                    status = f"built on {backend} in {elapsed:.2f} s"

                print(
                    "[Demag Lindholm HMatrixGPU] "
                    f"{status} | far={len(self.B_hmatrix_cpu.far_blocks):,} | "
                    f"rank avg/max="
                    f"{float(meta.get('average_far_rank', 0.0)):.1f}/"
                    f"{int(meta.get('max_observed_rank', 0))} | "
                    f"compression="
                    f"{self.B_hmatrix_cpu.compression_ratio_entries:.2f}x | "
                    f"{factor_precision} | "
                    f"GPU={memory['accounted_gpu_human']}",
                    flush=True,
                )

        self.xb_gpu = cp.zeros(self.opB.Nb, dtype=self.B_dtype)

        self.bdofs_owned_gpu = cp.asarray(
            self.opB.bdofs_owned,
            dtype=cp.int32,
        )

        self.bidx_owned = self.opB.bidx_owned.astype(np.int32)
        self.bidx_owned_gpu = cp.asarray(
            self.bidx_owned,
            dtype=cp.int32,
        )

        self.rowpos_for_bdof_gpu = cp.asarray(
            self.opB.rowpos_for_bdof,
            dtype=cp.int32,
        )
        self.boundary_dofs_gpu = cp.asarray(
            self.boundary_dofs,
            dtype=cp.int32,
        )

        # In single-rank mode, nrows == Nb. The H-matrix backend returns its
        # result in the original global boundary ordering.
        self.y_rows_gpu = cp.empty(self.opB.Nb, dtype=self.B_dtype)

        if self.rank == 0 and (
            self.boundary_backend == "dense" or hmatrix_verbose
        ):
            dense_mib = dense_theoretical_bytes / 1024**2
            print(
                "[Demag Lindholm GPU] "
                f"boundary_backend={self.boundary_backend}, "
                f"Nb={self.opB.Nb}, dense_theoretical={dense_mib:.3f} MiB",
                flush=True,
            )
        # ------------------------------------------------------------
        # H = -grad(u1 + u2)
        # ------------------------------------------------------------
        vvec = ufl.TestFunction(V)
        phi = ufl.TrialFunction(V1)

        G_form = fem.form(ufl.inner(vvec, -ufl.grad(phi)) * ufl.dx)
        G_cpu = assemble_matrix(G_form)
        G_cpu.assemble()

        self.G = _set_mat_cuda(G_cpu)

        self.phi = self.G.createVecRight()
        self.H_tmp = self.G.createVecLeft()

        _set_vec_cuda(self.phi)
        _set_vec_cuda(self.H_tmp, block_size=3)

    def _solve_u1_gpu(self, m_vec):
        self.Div.mult(m_vec, self.b_u1)

        try:
            self.ns_u1.remove(self.b_u1)
        except Exception:
            pass

        self.ksp_u1.solve(self.b_u1, self.u1)

        try:
            self.ns_u1.remove(self.u1)
        except Exception:
            pass

 
    def _apply_B_gpu(self):
        """
        Compute boundary Dirichlet data:

            g = B u1_boundary

        Single-rank GPU version.
        """

        u1_cp = _vec_to_cupy(self.u1, "r")

        self.xb_gpu.fill(0.0)

        self.xb_gpu[self.bidx_owned_gpu] = u1_cp[self.bdofs_owned_gpu]

        if self.boundary_backend == "dense":
            cp.matmul(
                self.B_dense_gpu,
                self.xb_gpu,
                out=self.y_rows_gpu,
            )
        else:
            self.B_hmatrix_gpu.matvec(
                self.xb_gpu,
                out=self.y_rows_gpu,
            )

        g_cp = _vec_to_cupy(self.g_u2, "rw")

        g_cp.fill(0.0)

        g_cp[self.bdofs_owned_gpu] = self.y_rows_gpu[self.rowpos_for_bdof_gpu]

    def _solve_u2_gpu(self):
        """
        Solve Laplace equation for u2 with Dirichlet data g_u2.

        Equivalent to:
            A_raw u = 0 in the interior,
            u = g on the boundary.

        Algebraically:
            b = -A_raw g
            b[boundary] = g[boundary]
            A_bc u = b
        """

        self.A_u2_raw.mult(self.g_u2, self.b_u2)
        self.b_u2.scale(-1.0)

        b_cp = _vec_to_cupy(self.b_u2, "rw")
        g_cp = _vec_to_cupy(self.g_u2, "r")

        b_cp[self.boundary_dofs_gpu] = g_cp[self.boundary_dofs_gpu]

        self.ksp_u2.solve(self.b_u2, self.u2)


    def compute_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Compute demagnetizing field on GPU.

        Parameters
        ----------
        m_vec:
            PETSc CUDA vector with magnetization.
        out_vec:
            PETSc CUDA vector where H_demag is written.
        """

        self._solve_u1_gpu(m_vec)
        self._apply_B_gpu()
        self._solve_u2_gpu()

        self.u1.copy(self.phi)
        self.phi.axpy(1.0, self.u2)

        self.G.mult(self.phi, out_vec)

        h_cp = _vec_to_cupy(out_vec, "rw")
        h_owned = h_cp[: self.local_size].reshape((-1, 3))

        h_owned *= self.inv_vol_scalar_gpu[:, None]

        if h_cp.size > self.local_size:
            h_cp[self.local_size :] = 0.0

    def boundary_memory_report(self):
        """
        Return a compact memory report for the active boundary backend.
        """
        if self.boundary_backend == "dense":
            nbytes = int(self.B_dense_gpu.nbytes)
            return {
                "backend": "dense",
                "Nb": int(self.opB.Nb),
                "accounted_gpu_bytes": nbytes,
                "accounted_gpu_mib": nbytes / 1024**2,
            }

        report = self.B_hmatrix_gpu.memory_report()
        report["Nb"] = int(self.opB.Nb)
        report["cache_path"] = self.hmatrix_cache_path
        return report

    def print_boundary_memory_report(self):
        report = self.boundary_memory_report()

        if report["backend"] == "dense":
            print(
                "[Demag Lindholm GPU] "
                f"backend=dense, Nb={report['Nb']}, "
                f"GPU memory={report['accounted_gpu_mib']:.3f} MiB",
                flush=True,
            )
            return

        self.B_hmatrix_gpu.print_memory_report(
            prefix="[Demag Lindholm HMatrixGPU]"
        )

    def Energy_lumped_gpu(self, m_vec: PETSc.Vec, h_vec: PETSc.Vec):
        """
        Lumped demagnetizing energy:

            E = -0.5 mu0 Ms int m · H_demag dV
        """

        m_cp = _vec_to_cupy(m_vec, "r")
        h_cp = _vec_to_cupy(h_vec, "r")

        m = m_cp[: self.local_size].reshape((-1, 3))
        h = h_cp[: self.local_size].reshape((-1, 3))

        density = cp.sum(m * h, axis=1)
        integral = cp.sum(density * self.vol_scalar_gpu)

        local_energy = float(integral.get())

        return -0.5 * self.mu0 * self.Ms * local_energy / self.volume_scale
    





def synchronize_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def set_vec_cuda(vec: PETSc.Vec, block_size=None):
    try:
        vec.setType(PETSc.Vec.Type.CUDA)
    except Exception:
        pass

    if block_size is not None:
        try:
            vec.setBlockSize(block_size)
        except Exception:
            pass

    try:
        vec.bindToCPU(False)
    except Exception:
        pass

    return vec
