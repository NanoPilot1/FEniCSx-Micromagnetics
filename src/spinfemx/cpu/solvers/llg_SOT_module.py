from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem import Function, Constant, form
from dolfinx.fem.petsc import assemble_vector

from ..fields.Exchange import ExchangeField
from ..fields.Anisotropy import AnisotropyField
from ..fields.DMI_Bulk import DMIBULK
from ..fields.DMI_Interfacial import DMIInterfacial




def _resolve_ksp_pc_configuration(
    pc_side: str,
    pc_norm_type: str,
):
    """Resolve the Krylov preconditioning side and convergence norm."""
    side_key = str(pc_side).strip().lower()
    side_aliases = {
        "left": "left",
        "l": "left",
        "right": "right",
        "r": "right",
    }
    if side_key not in side_aliases:
        raise ValueError(
            "pc_side must be 'left' or 'right'. "
            f"Received {pc_side!r}."
        )
    side_key = side_aliases[side_key]

    norm_key = str(pc_norm_type).strip().lower().replace("-", "_")
    norm_aliases = {
        "preconditioned": "preconditioned",
        "pc": "preconditioned",
        "unpreconditioned": "unpreconditioned",
        "true": "unpreconditioned",
        "true_residual": "unpreconditioned",
    }
    if norm_key not in norm_aliases:
        raise ValueError(
            "pc_norm_type must be 'preconditioned' or 'unpreconditioned'. "
            f"Received {pc_norm_type!r}."
        )
    norm_key = norm_aliases[norm_key]

    if side_key == "right" and norm_key != "unpreconditioned":
        raise ValueError(
            "Right-preconditioned GMRES must use the unpreconditioned "
            "(true) residual norm in this solver."
        )

    side_value = {
        "left": PETSc.PC.Side.LEFT,
        "right": PETSc.PC.Side.RIGHT,
    }[side_key]
    norm_value = {
        "preconditioned": PETSc.KSP.NormType.PRECONDITIONED,
        "unpreconditioned": PETSc.KSP.NormType.UNPRECONDITIONED,
    }[norm_key]
    return side_key, norm_key, side_value, norm_value


# Local SOT / LLG kernels


@njit(fastmath=True, inline="always")
def _llg_sot_jv_one(
    mx, my, mz,
    hx, hy, hz,
    vx, vy, vz,
    hvx, hvy, hvz,
    px, py, pz,
    gamma, alpha, do_precess, Stab,
    c_fl, c_dl,
):
    coef = -gamma / (1.0 + alpha * alpha)
    cvhx = vy * hz - vz * hy
    cvhy = vz * hx - vx * hz
    cvhz = vx * hy - vy * hx

    cmhx = my * hvz - mz * hvy
    cmhy = mz * hvx - mx * hvz
    cmhz = mx * hvy - my * hvx

    prec_x = do_precess * (cvhx + cmhx)
    prec_y = do_precess * (cvhy + cmhy)
    prec_z = do_precess * (cvhz + cmhz)

    md_h = mx * hx + my * hy + mz * hz
    md_hv = mx * hvx + my * hvy + mz * hvz
    vd_h = vx * hx + vy * hy + vz * hz
    md_v = mx * vx + my * vy + mz * vz
    md_m = mx * mx + my * my + mz * mz
    s = vd_h + md_hv

    damp_x = vx * md_h - 2.0 * hx * md_v + mx * s - hvx * md_m
    damp_y = vy * md_h - 2.0 * hy * md_v + my * s - hvy * md_m
    damp_z = vz * md_h - 2.0 * hz * md_v + mz * s - hvz * md_m

    cx = my * pz - mz * py
    cy = mz * px - mx * pz
    cz = mx * py - my * px

    dcx = vy * pz - vz * py
    dcy = vz * px - vx * pz
    dcz = vx * py - vy * px

    vxc_x = vy * cz - vz * cy
    vxc_y = vz * cx - vx * cz
    vxc_z = vx * cy - vy * cx

    mxdc_x = my * dcz - mz * dcy
    mxdc_y = mz * dcx - mx * dcz
    mxdc_z = mx * dcy - my * dcx

    dsot_x = c_fl * dcx + c_dl * (vxc_x + mxdc_x)
    dsot_y = c_fl * dcy + c_dl * (vxc_y + mxdc_y)
    dsot_z = c_fl * dcz + c_dl * (vxc_z + mxdc_z)

    one_minus = 1.0 - md_m
    stab_x = Stab * (vx * one_minus - 2.0 * mx * md_v)
    stab_y = Stab * (vy * one_minus - 2.0 * my * md_v)
    stab_z = Stab * (vz * one_minus - 2.0 * mz * md_v)

    out_x = coef * (prec_x + alpha * damp_x) + dsot_x + stab_x
    out_y = coef * (prec_y + alpha * damp_y) + dsot_y + stab_y
    out_z = coef * (prec_z + alpha * damp_z) + dsot_z + stab_z
    return out_x, out_y, out_z


@njit(fastmath=True)
def jac_vec_sot_local_kernel(
    M, Hm, V, Hv, Pvec, out,
    gamma, alpha, do_precess, Stab, c_fl, c_dl,
):
    """Evaluate J(m) v for LLG + local SOT over owned nodes."""
    for i in range(M.shape[0]):
        out[3 * i + 0], out[3 * i + 1], out[3 * i + 2] = _llg_sot_jv_one(
            M[i, 0], M[i, 1], M[i, 2],
            Hm[i, 0], Hm[i, 1], Hm[i, 2],
            V[i, 0], V[i, 1], V[i, 2],
            Hv[i, 0], Hv[i, 1], Hv[i, 2],
            Pvec[i, 0], Pvec[i, 1], Pvec[i, 2],
            gamma, alpha, do_precess, Stab, c_fl, c_dl,
        )


@njit(fastmath=True)
def llg_rhs_sot_local_kernel(
    M, H, Pvec, out,
    gamma, alpha, do_precess, Stab, c_fl, c_dl,
):
    """CPU/Numba RHS for explicit LLG + local DL/FL spin-orbit torque."""
    coef = -gamma / (1.0 + alpha * alpha)

    for i in range(M.shape[0]):
        mx, my, mz = M[i, 0], M[i, 1], M[i, 2]
        hx, hy, hz = H[i, 0], H[i, 1], H[i, 2]
        px, py, pz = Pvec[i, 0], Pvec[i, 1], Pvec[i, 2]

        # m x H
        cx = my * hz - mz * hy
        cy = mz * hx - mx * hz
        cz = mx * hy - my * hx

        # m x (m x H)
        ccx = my * cz - mz * cy
        ccy = mz * cx - mx * cz
        ccz = mx * cy - my * cx

        # m x p
        spx = my * pz - mz * py
        spy = mz * px - mx * pz
        spz = mx * py - my * px

        # m x (m x p)
        sdpx = my * spz - mz * spy
        sdpy = mz * spx - mx * spz
        sdpz = mx * spy - my * spx

        norm2 = mx * mx + my * my + mz * mz
        stab = Stab * (1.0 - norm2)

        out[3 * i + 0] = coef * (do_precess * cx + alpha * ccx) + c_fl * spx + c_dl * sdpx + stab * mx
        out[3 * i + 1] = coef * (do_precess * cy + alpha * ccy) + c_fl * spy + c_dl * sdpy + stab * my
        out[3 * i + 2] = coef * (do_precess * cz + alpha * ccz) + c_fl * spz + c_dl * sdpz + stab * mz


@njit(fastmath=True)
def pc_build_inv3_sot_kernel(
    M, H, Pvec, diagK,
    i00, i01, i02,
    i10, i11, i12,
    i20, i21, i22,
    shift, gamma, alpha, do_precess, Stab, c_fl, c_dl,
    eps_reg, det_eps,
):
    """
    Build local inverse blocks for shift*I - J_local.
    """
    s = shift + eps_reg

    for idx in range(M.shape[0]):
        mx, my, mz = M[idx, 0], M[idx, 1], M[idx, 2]
        hx, hy, hz = H[idx, 0], H[idx, 1], H[idx, 2]
        px, py, pz = Pvec[idx, 0], Pvec[idx, 1], Pvec[idx, 2]
        kx, ky, kz = diagK[idx, 0], diagK[idx, 1], diagK[idx, 2]

        # Columns of J_local obtained by applying the exact local derivative to
        # Cartesian basis vectors, with Hv ~= diag(K_total) v.
        j00, j10, j20 = _llg_sot_jv_one(
            mx, my, mz, hx, hy, hz,
            1.0, 0.0, 0.0, kx, 0.0, 0.0,
            px, py, pz,
            gamma, alpha, do_precess, Stab, c_fl, c_dl,
        )
        j01, j11, j21 = _llg_sot_jv_one(
            mx, my, mz, hx, hy, hz,
            0.0, 1.0, 0.0, 0.0, ky, 0.0,
            px, py, pz,
            gamma, alpha, do_precess, Stab, c_fl, c_dl,
        )
        j02, j12, j22 = _llg_sot_jv_one(
            mx, my, mz, hx, hy, hz,
            0.0, 0.0, 1.0, 0.0, 0.0, kz,
            px, py, pz,
            gamma, alpha, do_precess, Stab, c_fl, c_dl,
        )

        A00 = s - j00
        A01 = -j01
        A02 = -j02
        A10 = -j10
        A11 = s - j11
        A12 = -j12
        A20 = -j20
        A21 = -j21
        A22 = s - j22

        det = (
            A00 * (A11 * A22 - A12 * A21)
            - A01 * (A10 * A22 - A12 * A20)
            + A02 * (A10 * A21 - A11 * A20)
        )

        if abs(det) < det_eps:
            if det >= 0.0:
                det += det_eps
            else:
                det -= det_eps

        invdet = 1.0 / det
        i00[idx] =  (A11 * A22 - A12 * A21) * invdet
        i01[idx] = -(A01 * A22 - A02 * A21) * invdet
        i02[idx] =  (A01 * A12 - A02 * A11) * invdet
        i10[idx] = -(A10 * A22 - A12 * A20) * invdet
        i11[idx] =  (A00 * A22 - A02 * A20) * invdet
        i12[idx] = -(A00 * A12 - A02 * A10) * invdet
        i20[idx] =  (A10 * A21 - A11 * A20) * invdet
        i21[idx] = -(A00 * A21 - A01 * A20) * invdet
        i22[idx] =  (A00 * A11 - A01 * A10) * invdet


@njit(fastmath=True)
def pc_apply_inv3_kernel(
    x, y,
    i00, i01, i02,
    i10, i11, i12,
    i20, i21, i22,
):
    """Apply local 3x3 inverse blocks."""
    for i in range(x.size // 3):
        j = 3 * i
        x0, x1, x2 = x[j], x[j + 1], x[j + 2]
        y[j] = i00[i] * x0 + i01[i] * x1 + i02[i] * x2
        y[j + 1] = i10[i] * x0 + i11[i] * x1 + i12[i] * x2
        y[j + 2] = i20[i] * x0 + i21[i] * x1 + i22[i] * x2



class JvContextSOTCPU:
    """Matrix-free Jacobian and local block-Jacobi PC for CPU SOT BDF."""

    def __init__(self, hef):
        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0
        self.enable_pc = True
        self._pc_ready = False

        diag = self.hef.K_total.getDiagonal()
        diagK = diag.getArray(readonly=True).copy()

        ani = getattr(self.hef, "anisotropy_field", None)
        if ani is not None and not hasattr(ani, "K") and hasattr(ani, "diagonal_array"):
            ani_diag = ani.diagonal_array(owned_only=True)
            diagK[: ani_diag.size] += ani_diag

        self.diagK_signed = diagK.reshape((-1, 3)).astype(np.float64)
        self.diagK_abs = np.abs(self.diagK_signed)

        n = self.hef.M_cached.shape[0]
        self.i00 = np.empty(n, dtype=np.float64)
        self.i01 = np.empty(n, dtype=np.float64)
        self.i02 = np.empty(n, dtype=np.float64)
        self.i10 = np.empty(n, dtype=np.float64)
        self.i11 = np.empty(n, dtype=np.float64)
        self.i12 = np.empty(n, dtype=np.float64)
        self.i20 = np.empty(n, dtype=np.float64)
        self.i21 = np.empty(n, dtype=np.float64)
        self.i22 = np.empty(n, dtype=np.float64)

    def update_pc_full_fast(self, shift, use_abs_diag=False, eps_reg=1e-14, det_eps=1e-30):
        self.shift = float(shift)
        diagK = self.diagK_abs if use_abs_diag else self.diagK_signed

        pc_build_inv3_sot_kernel(
            self.hef.M_cached,
            self.hef.Hm_cached,
            self.hef.p_owned,
            diagK,
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
            float(self.shift),
            float(self.hef.gamma),
            float(self.hef.alpha),
            float(self.hef.do_precess),
            float(self.hef.Stab),
            float(self.hef.c_fl),
            float(self.hef.c_dl),
            float(eps_reg),
            float(det_eps),
        )
        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1
        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        pc_apply_inv3_kernel(
            x.getArray(readonly=True),
            y.getArray(),
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
        )

    def mult(self, A, x, y):
        self.calls += 1
        xv = x.getArray(readonly=True)
        yv = y.getArray()
        self.hef.jac_vec_times_SOT(xv, out=self.hef.Jv_buffer)
        yv[:] = self.shift * xv - self.hef.Jv_buffer



# Effective field + SOT


class EffectiveFieldSOT:
    """Effective field and local SOT for one ferromagnetic magnetization."""

    def __init__(
        self,
        mesh,
        Ms,
        Aex,
        Ku,
        n_ani_vec,
        D_bulk,
        D_int,
        n0_int_vec,
        H0_vec,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        H_dl=0.0,
        H_fl=0.0,
        polarization_vec=None,
        use_demag=True,
        demag_method="lindholm",
        demag_kwargs=None,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        self.V1 = fem.functionspace(mesh, ("Lagrange", 1))

        # Unique P1-node pairs connected by mesh edges.  Built once and reused
        # only when writing monitoring information (see max_neighbor_angle_deg).
        self.neighbor_pairs = self._build_neighbor_pairs()

        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.A = float(Aex)
        self.Ku = float(Ku)
        self.D_bulk = float(D_bulk)
        self.D_int = float(D_int)
        self.do_precess = float(do_precess)
        self.use_demag = bool(use_demag)
        self.mu0 = 4.0 * np.pi * 1e-7

        self.H_dl = float(H_dl)
        self.H_fl = float(H_fl)
        den = 1.0 + self.alpha * self.alpha
        self.c_fl = -self.gamma * (self.H_fl - self.alpha * self.H_dl) / den
        self.c_dl = -self.gamma * (self.H_dl + self.alpha * self.H_fl) / den

        self.m = Function(self.V, name="m")
        self.dmdt = Function(self.V, name="dmdt")
        self.H_eff = Function(self.V, name="H_eff")
        self.H0_ext = Function(self.V, name="H0_ext")
        self.p_field = Function(self.V, name="sot_polarization")

        self.n_nodes_local = int(self.V.dofmap.index_map.size_local)
        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = int(self.end - self.start)
        self.local_size = 3 * self.local_dofs

        self._assign_vector_field(self.H0_ext, H0_vec, normalize=False, name="H0_vec")
        self._assign_vector_field(self.p_field, polarization_vec, normalize=True, name="polarization_vec")
        self.p_owned = self.p_field.x.array[: self.local_size].reshape((-1, 3)).copy()

        self.Stab = self.Ms * self.gamma / (1.0 + self.alpha**2) * 0.5
        self.He = np.zeros(self.local_size, dtype=np.float64)
        self.Hfield = np.zeros(self.local_size, dtype=np.float64)

        # Lumped nodal volumes, used for the external Zeeman energy.
        v = ufl.TestFunction(self.V)
        tmp_0 = ufl.dot(v, Constant(mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx
        volN_f = Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp_0))
        volN_f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        volN_f.x.scatter_forward()
        volN = volN_f.x.array
        self.vol_nodes = np.asarray(volN[: self.local_size], dtype=np.float64).reshape((-1, 3))[:, 0]
        self.volume_scale_energy = 1e-27  # mesh coordinates are expected in nm

        self.demag_field = None
        if self.use_demag:
            from ..fields.Demag import make_demag_field
            kwargs = {} if demag_kwargs is None else dict(demag_kwargs)
            if self.comm.rank == 0:
                print(f"[Demag-SOT] Precomputing demag method='{demag_method}' ...", flush=True)
            t0 = perf_counter()
            self.demag_field = make_demag_field(demag_method, mesh, self.V, self.V1, self.Ms, **kwargs)
            if self.comm.rank == 0:
                print(f"[Demag-SOT] Precomputation finished in {perf_counter() - t0:.2f} s", flush=True)

        self.exchange_field = ExchangeField(mesh, self.V, self.A, self.Ms, volN)
        self.anisotropy_field = AnisotropyField(mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN)
        self.DMIBULK = DMIBULK(mesh, self.V, self.V1, self.D_bulk, self.Ms, volN)

        self.DMI_int = None
        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(mesh, self.V, self.V1, self.D_int, n0_int_vec, self.Ms, volN)

        self.K_total = self.exchange_field.K + self.DMIBULK.K
        if self.anisotropy_field is not None and hasattr(self.anisotropy_field, "K"):
            self.K_total = self.K_total + self.anisotropy_field.K
        if self.DMI_int is not None:
            self.K_total = self.K_total + self.DMI_int.K

        self.v_jac = Function(self.V, name="v_jac")
        self.H_m = Function(self.V, name="H_m")
        self.H_v = Function(self.V, name="H_v")
        self.Jv_buffer = np.zeros(self.local_size, dtype=np.float64)
        self.M_cached = np.zeros((self.local_dofs, 3), dtype=np.float64)
        self.Hm_cached = np.zeros((self.local_dofs, 3), dtype=np.float64)
        self.Hani_m_cache = np.zeros((self.local_dofs, 3), dtype=np.float64)
        self.Hani_v_cache = np.zeros((self.local_dofs, 3), dtype=np.float64)
        self.Hv_total = np.zeros((self.local_dofs, 3), dtype=np.float64)

        self.JacStep = 0
        self.LLGStep = 0

        # Trigger Numba compilation early.
        jac_vec_sot_local_kernel(
            self.M_cached,
            self.Hm_cached,
            self.M_cached,
            self.Hm_cached,
            self.p_owned,
            self.Jv_buffer,
            self.gamma,
            self.alpha,
            self.do_precess,
            self.Stab,
            self.c_fl,
            self.c_dl,
        )

    def _build_neighbor_pairs(self):
        """
        Build unique pairs of neighboring P1 nodes connected by mesh edges.

        The P1 scalar space V1 and the blocked vector space V use the same
        nodal ordering.  For each locally owned cell, every pair of cell
        vertices is an edge candidate.  Duplicate pairs are removed locally.
        A later MPI reduction is sufficient because only the global maximum
        angle is required.
        """
        tdim = self.mesh.topology.dim
        n_cells_local = self.mesh.topology.index_map(tdim).size_local

        pairs = set()

        for cell in range(n_cells_local):
            dofs = self.V1.dofmap.cell_dofs(cell)

            for a in range(len(dofs)):
                for b in range(a + 1, len(dofs)):
                    i = int(dofs[a])
                    j = int(dofs[b])

                    if i != j:
                        pairs.add((min(i, j), max(i, j)))

        if not pairs:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(sorted(pairs), dtype=np.int32)

    def max_neighbor_angle_deg(self, m=None):
        """
        Return the global maximum angle, in degrees, between neighboring
        nodal magnetic moments.

        Ghost values are synchronized before evaluating local mesh edges.
        Zero-length moments are ignored defensively.
        """
        if m is None:
            m = self.m

        m.x.scatter_forward()
        pairs = self.neighbor_pairs

        if pairs.size == 0:
            local_max = 0.0
        else:
            moments = m.x.array.reshape((-1, 3))

            mi = moments[pairs[:, 0]]
            mj = moments[pairs[:, 1]]

            norm_i = np.linalg.norm(mi, axis=1)
            norm_j = np.linalg.norm(mj, axis=1)

            valid = (norm_i > 1e-14) & (norm_j > 1e-14)

            if np.any(valid):
                cosine = np.einsum(
                    "ij,ij->i",
                    mi[valid],
                    mj[valid],
                ) / (norm_i[valid] * norm_j[valid])

                cosine = np.clip(cosine, -1.0, 1.0)
                local_max = float(np.degrees(np.arccos(cosine)).max())
            else:
                local_max = 0.0

        return float(self.comm.allreduce(local_max, op=MPI.MAX))

    def _assign_vector_field(self, function, values, *, normalize, name):
        if values is None:
            function.x.array[:] = 0.0
            function.x.scatter_forward()
            return

        arr = np.asarray(values, dtype=np.float64)
        out = function.x.array.reshape((-1, 3))

        if arr.ndim == 1 and arr.size == 3:
            out[:, :] = arr[None, :]
        elif arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] == out.shape[0]:
            out[:, :] = arr
        elif arr.size == function.x.array.size:
            function.x.array[:] = arr.reshape(-1)
        elif arr.size == self.local_size:
            function.x.array[: self.local_size] = arr.reshape(-1)
        else:
            raise ValueError(
                f"{name} must have shape (3,), ({out.shape[0]}, 3), "
                f"({function.x.array.size},), or ({self.local_size},). Got {arr.shape}."
            )

        if normalize:
            norms = np.linalg.norm(out, axis=1)
            mask = norms > 0.0
            out[mask, :] /= norms[mask, None]

        function.x.scatter_forward()

    def set_uniform_field(self, Hx, Hy, Hz):
        self._assign_vector_field(self.H0_ext, np.array([Hx, Hy, Hz]), normalize=False, name="H0_vec")

    def set_sot(self, H_dl=None, H_fl=None, polarization_vec=None):
        if H_dl is not None:
            self.H_dl = float(H_dl)
        if H_fl is not None:
            self.H_fl = float(H_fl)
        if polarization_vec is not None:
            self._assign_vector_field(self.p_field, polarization_vec, normalize=True, name="polarization_vec")
            self.p_owned[:, :] = self.p_field.x.array[: self.local_size].reshape((-1, 3))

        den = 1.0 + self.alpha * self.alpha
        self.c_fl = -self.gamma * (self.H_fl - self.alpha * self.H_dl) / den
        self.c_dl = -self.gamma * (self.H_dl + self.alpha * self.H_fl) / den

    def compute_H_eff(self, m):
        self.He[:] = self.exchange_field.compute(m).x.petsc_vec.array[: self.local_size]
        if self.demag_field is not None:
            self.He += self.demag_field.compute(m).x.petsc_vec.array[: self.local_size]
        if abs(self.Ku) > 0.0:
            self.He += self.anisotropy_field.compute(m).x.petsc_vec.array[: self.local_size]
        if abs(self.D_bulk) > 0.0:
            self.He += self.DMIBULK.compute(m).x.petsc_vec.array[: self.local_size]
        if self.DMI_int is not None and abs(self.D_int) > 0.0:
            self.He += self.DMI_int.compute(m).x.petsc_vec.array[: self.local_size]
        self.He += self.H0_ext.x.petsc_vec.array[: self.local_size]
        return self.He

    def _field_energy_global(self, local_value):
        return float(self.comm.allreduce(float(local_value), op=MPI.SUM))

    def _external_energy_global(self, m):
        M = m.x.array[: self.local_size].reshape((-1, 3))
        H = self.H0_ext.x.array[: self.local_size].reshape((-1, 3))
        local = np.sum(self.vol_nodes * np.einsum("ij,ij->i", M, H))
        return -self.mu0 * self.Ms * self.volume_scale_energy * float(self.comm.allreduce(float(local), op=MPI.SUM))

    def compute_Energy_terms(self, m):
        E_exch = self._field_energy_global(self.exchange_field.Energy(m))
        E_demag = 0.0
        if self.demag_field is not None:
            # Refresh H_d for the accepted state before evaluating its energy.
            # This matters because the field buffer may have been overwritten by
            # nonlinear/Jacobian evaluations at another state.
            self.demag_field.compute(m)
            E_demag = self._field_energy_global(self.demag_field.Energy(m))
        E_ani = 0.0 if abs(self.Ku) == 0.0 else self._field_energy_global(self.anisotropy_field.Energy(m))
        E_db = 0.0 if abs(self.D_bulk) == 0.0 else self._field_energy_global(self.DMIBULK.Energy(m))
        E_di = 0.0 if self.DMI_int is None or abs(self.D_int) == 0.0 else self._field_energy_global(self.DMI_int.Energy(m))
        E_ext = self._external_energy_global(m)
        return {
            "E_demag": E_demag,
            "E_exch": E_exch,
            "E_ani": E_ani,
            "E_dmi_bulk": E_db,
            "E_dmi_int": E_di,
            "E_ext": E_ext,
            "E_total": E_demag + E_exch + E_ani + E_db + E_di + E_ext,
        }

    def compute_Energy(self, m):
        return self.compute_Energy_terms(m)["E_total"]

    def update_jac_state(self):
        self.K_total.mult(self.m.x.petsc_vec, self.H_m.x.petsc_vec)
        M_loc = self.m.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
        Hm_loc = self.H_m.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
        self.M_cached[:, :] = M_loc

        if abs(self.Ku) > 0.0 and not hasattr(self.anisotropy_field, "K") and hasattr(self.anisotropy_field, "apply_array"):
            self.anisotropy_field.apply_array(
                self.m.x.petsc_vec.getArray(readonly=True),
                out_flat=self.Hani_m_cache.reshape(-1),
                owned_only=True,
            )
            self.Hm_cached[:, :] = Hm_loc + self.Hani_m_cache
        else:
            self.Hm_cached[:, :] = Hm_loc

        self.Hm_cached[:, :] += self.H0_ext.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))

    def jac_vec_times_SOT(self, v, out):
        self.JacStep += 1
        self.v_jac.x.array[: self.local_size] = v
        self.v_jac.x.scatter_forward()
        self.K_total.mult(self.v_jac.x.petsc_vec, self.H_v.x.petsc_vec)

        Hv_lin = self.H_v.x.petsc_vec.getArray(readonly=True).reshape((-1, 3))
        if abs(self.Ku) > 0.0 and not hasattr(self.anisotropy_field, "K") and hasattr(self.anisotropy_field, "apply_array"):
            self.anisotropy_field.apply_array(v, out_flat=self.Hani_v_cache.reshape(-1), owned_only=True)
            np.add(Hv_lin, self.Hani_v_cache, out=self.Hv_total)
            Hv = self.Hv_total
        else:
            Hv = Hv_lin

        jac_vec_sot_local_kernel(
            self.M_cached,
            self.Hm_cached,
            np.asarray(v).reshape((-1, 3)),
            Hv,
            self.p_owned,
            out,
            self.gamma,
            self.alpha,
            self.do_precess,
            self.Stab,
            self.c_fl,
            self.c_dl,
        )

    def llg_rhs_SOT(self, m):
        self.LLGStep += 1
        M = m.x.petsc_vec.getArray(readonly=True)[: self.local_size].reshape((-1, 3))
        H = self.compute_H_eff(m)[: self.local_size].reshape((-1, 3))
        out = self.dmdt.x.petsc_vec.array[: self.local_size]
        llg_rhs_sot_local_kernel(
            M, H, self.p_owned, out,
            self.gamma, self.alpha, self.do_precess, self.Stab, self.c_fl, self.c_dl,
        )
        return self.dmdt

    def ifunction_SOT(self, ts, t, y, ydot, f):
        y.copy(self.m.x.petsc_vec)
        self.m.x.scatter_forward()
        dmdt = self.llg_rhs_SOT(self.m)
        f.waxpy(-1.0, dmdt.x.petsc_vec, ydot)
        return 0



# Public solver


class LLG_SOT:
    """CPU BDF solver for FM LLG with local damping-like / field-like SOT."""

    def __init__(self, mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=1):
        self.mesh = mesh
        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.do_precess = float(do_precess)

        self._Aex = 0.0
        self._Ku = 0.0
        self._n_ani = None
        self._D_bulk = 0.0
        self._D_int = 0.0
        self._n0_int = None
        self._H0_vec = None
        self._H_dl = 0.0
        self._H_fl = 0.0
        self._polarization_vec = None

        self._has_exchange = False
        self._has_demag = False
        self._demag_method = "lindholm"
        self._demag_kwargs = {}
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_H0 = False
        self._has_sot = False

        self.hef: EffectiveFieldSOT | None = None

    def add_exchange(self, Aex):
        self._Aex = float(Aex)
        self._has_exchange = True

    def add_demag(self, method="lindholm", **kwargs):
        self._has_demag = True
        self._demag_method = method
        self._demag_kwargs = dict(kwargs)

    def add_anisotropy(self, Ku, n_vec):
        self._Ku = float(Ku)
        self._n_ani = n_vec
        self._has_anisotropy = True

    def add_dmi_bulk(self, D_bulk):
        self._D_bulk = float(D_bulk)
        self._has_dmi_bulk = True

    def add_dmi_interfacial(self, D_int, n0_vec):
        self._D_int = float(D_int)
        self._n0_int = n0_vec
        self._has_dmi_int = True

    def add_external_field(self, H0_vec):
        self._H0_vec = H0_vec
        self._has_H0 = True

    def add_sot(self, H_dl, H_fl=0.0, polarization_vec=(0.0, 1.0, 0.0)):
        """Enable local SOT.

        Parameters
        ----------
        H_dl, H_fl : float
            Damping-like and field-like effective fields in A/m.
        polarization_vec : array-like
            Uniform direction (3,) or nodal vector field.  Nonzero vectors are
            normalized node by node.
        """
        self._H_dl = float(H_dl)
        self._H_fl = float(H_fl)
        self._polarization_vec = polarization_vec
        self._has_sot = True

    def add_sot_from_spin_hall(self, J, theta_sh, thickness, polarization_vec=(0.0, 1.0, 0.0), H_fl=0.0):
        """Convenience wrapper for H_dl = hbar*theta_sh*J/(2*e*mu0*Ms*t)."""
        hbar = 1.054571817e-34
        e = 1.602176634e-19
        mu0 = 4.0 * np.pi * 1e-7
        H_dl = hbar * float(theta_sh) * float(J) / (2.0 * e * mu0 * self.Ms * float(thickness))
        self.add_sot(H_dl=H_dl, H_fl=H_fl, polarization_vec=polarization_vec)
        return H_dl

    def set_uniform_field(self, Hx, Hy, Hz):
        """Update a uniform static field after the effective field was built."""
        self._H0_vec = np.array([Hx, Hy, Hz], dtype=np.float64)
        self._has_H0 = True
        if self.hef is not None:
            self.hef.set_uniform_field(Hx, Hy, Hz)

    def set_sot(self, H_dl=None, H_fl=None, polarization_vec=None):
        """Update SOT parameters after construction without rebuilding FEM operators."""
        if H_dl is not None:
            self._H_dl = float(H_dl)
        if H_fl is not None:
            self._H_fl = float(H_fl)
        if polarization_vec is not None:
            self._polarization_vec = polarization_vec
        self._has_sot = True
        if self.hef is not None:
            self.hef.set_sot(H_dl=H_dl, H_fl=H_fl, polarization_vec=polarization_vec)

    def _default_vector_field(self):
        return np.zeros(3 * len(self.mesh.geometry.x), dtype=np.float64)

    def _build_effective_field(self):
        Aex = self._Aex if self._has_exchange else 0.0
        Ku = self._Ku if self._has_anisotropy else 0.0
        n_ani_vec = self._n_ani if self._has_anisotropy and self._n_ani is not None else self._default_vector_field()
        D_bulk = self._D_bulk if self._has_dmi_bulk else 0.0
        D_int = self._D_int if self._has_dmi_int else 0.0
        n0_int_vec = self._n0_int if self._has_dmi_int and self._n0_int is not None else self._default_vector_field()
        H0_vec = self._H0_vec if self._has_H0 and self._H0_vec is not None else self._default_vector_field()
        H_dl = self._H_dl if self._has_sot else 0.0
        H_fl = self._H_fl if self._has_sot else 0.0
        pvec = self._polarization_vec if self._has_sot and self._polarization_vec is not None else (0.0, 1.0, 0.0)

        self.hef = EffectiveFieldSOT(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            H0_vec,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            H_dl=H_dl,
            H_fl=H_fl,
            polarization_vec=pvec,
            use_demag=self._has_demag,
            demag_method=self._demag_method,
            demag_kwargs=self._demag_kwargs,
        )

    def solve(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        dt_save=None,
        dt_snap=None,
        output_dir="output",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        monitor_fn=None,
        pc_python=True,
        pc_use_abs_diag=False,
        pc_side="right",
        pc_norm_type="unpreconditioned",
    ):
        if self.hef is None:
            self._build_effective_field()
        hef = self.hef

        (
            pc_side_key,
            pc_norm_key,
            pc_side_value,
            pc_norm_value,
        ) = _resolve_ksp_pc_configuration(
            pc_side=pc_side,
            pc_norm_type=pc_norm_type,
        )

        hef.m.x.array[:] = np.asarray(m0_array, dtype=np.float64).reshape(-1)
        hef.m.x.scatter_forward()

        ts = PETSc.TS().create(self.mesh.comm)
        opts = PETSc.Options()
        opts["ts_type"] = "bdf"
        opts["ts_adapt_type"] = "basic"
        opts["ts_adapt_clip"] = "0.1, 3.0"
        opts["ts_adapt_safety"] = 0.9
        opts["ts_adapt_reject_safety"] = 0.1
        opts["ts_adapt_scale_solve_failed"] = 0.25
        opts["ts_adapt_dt_min"] = 1e-17
        opts["ts_adapt_dt_max"] = 1e-10
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "bt"
        opts["snes_linesearch_order"] = 2
        opts["ts_rtol"] = ts_rtol
        opts["ts_atol"] = ts_atol
        opts["ts_max_steps"] = 5_000_000
        opts["snes_rtol"] = snes_rtol
        opts["snes_atol"] = snes_atol
        opts["snes_max_it"] = 8
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = ksp_rtol
        opts["ts_max_snes_failures"] = -1

        ts.setTime(t0)
        ts.setTimeStep(dt_init)
        ts.setMaxTime(t_final)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        dm = PETSc.DMShell().create(comm=self.mesh.comm)
        dm.setGlobalVector(hef.m.x.petsc_vec.copy())
        ts.setDM(dm)
        snes = ts.getSNES()

        n_loc = hef.m.x.petsc_vec.getLocalSize()
        n_glob = hef.m.x.petsc_vec.getSize()
        J = PETSc.Mat().create(comm=self.mesh.comm)
        ctx = JvContextSOTCPU(hef)
        J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
        J.setType("python")
        J.setPythonContext(ctx)
        J.setUp()

        ksp = snes.getKSP()
        pc = ksp.getPC()
        if pc_python:
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(ctx)
        else:
            pc.setType(PETSc.PC.Type.NONE)
            ctx.enable_pc = False

        def IJac(ts_, t, y, ydot, shift, A, B):
            y.copy(hef.m.x.petsc_vec)
            hef.m.x.scatter_forward()
            hef.update_jac_state()
            if pc_python:
                ctx.update_pc_full_fast(
                    shift,
                    use_abs_diag=bool(pc_use_abs_diag),
                )
            else:
                ctx.shift = float(shift)
            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

        ts.setIFunction(hef.ifunction_SOT)
        ts.setIJacobian(IJac, J)
        ts.setFromOptions()

        # Re-apply the explicit API configuration after setFromOptions(), so
        # ambient PETSc options cannot silently change the comparison.
        ksp = ts.getSNES().getKSP()
        pc = ksp.getPC()
        if pc_python:
            ctx.enable_pc = True
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(ctx)
            ksp.setPCSide(pc_side_value)
            ksp.setNormType(pc_norm_value)
        else:
            ctx.enable_pc = False
            pc.setType(PETSc.PC.Type.NONE)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

        y = hef.m.x.petsc_vec.copy()
        y.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        hef.compute_H_eff(hef.m)

        if dt_save is not None:
            if dt_save <= 0.0:
                raise ValueError("dt_save must be positive.")
            if dt_snap is None:
                dt_snap = dt_save
            if dt_snap <= 0.0:
                raise ValueError("dt_snap must be positive.")

            if self.mesh.comm.rank == 0:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(output_dir) / "log.txt"
            last_save_n = {"n": -1}
            last_snap_n = {"n": -1}
            snap_counter = {"k": 0}
            first_print = {"done": False}

            def default_monitor(ts_, step, t, u, hef_, mesh_):
                u.copy(hef_.m.x.petsc_vec)
                hef_.m.x.scatter_forward()
                energies = hef_.compute_Energy_terms(hef_.m)
                mag = mesh_.comm.gather(hef_.m.x.petsc_vec.getArray(readonly=True), root=0)

                # Spatial-resolution diagnostic: maximum angle between magnetic
                # moments at neighboring mesh nodes.  Collective (MPI allreduce),
                # so it must run on every rank, outside the rank-0 block.
                max_neighbor_angle_deg = hef_.max_neighbor_angle_deg(hef_.m)

                n_snap = int(np.trunc(t / dt_snap))
                if n_snap != last_snap_n["n"]:
                    last_snap_n["n"] = n_snap
                    filename = Path(output_dir) / f"m{snap_counter['k']:03d}.xdmf"
                    snap_counter["k"] += 1
                    with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                        xdmf.write_mesh(mesh_)
                        xdmf.write_function(hef_.m)

                if mesh_.comm.rank == 0:
                    mag = np.reshape(np.concatenate(mag), (-1, 3))
                    if not first_print["done"]:
                        header = (
                            f"{'time':>10} {'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                            f"{'max_nn_angle(deg)':>18} "
                            f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                            f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} {'E_ext':>15} {'E_total':>15}"
                        )
                        print(header)
                        with open(log_path, "w") as f:
                            f.write(header + "\n")
                        first_print["done"] = True

                    line = (
                        f"{t * 1e9:10.4f} "
                        f"{mag[:, 0].mean():15.6f} {mag[:, 1].mean():15.6f} {mag[:, 2].mean():15.6f} "
                        f"{max_neighbor_angle_deg:18.6f} "
                        f"{energies['E_demag']:15.4e} {energies['E_exch']:15.4e} {energies['E_ani']:15.4e} "
                        f"{energies['E_dmi_bulk']:15.4e} {energies['E_dmi_int']:15.4e} "
                        f"{energies['E_ext']:15.4e} {energies['E_total']:15.4e}"
                    )
                    print(line)
                    with open(log_path, "a") as f:
                        f.write(line + "\n")
                    sys.stdout.flush()

            def monitor(ts_, step, t, u):
                n = int(np.trunc(t / dt_save))
                if n != last_save_n["n"]:
                    last_save_n["n"] = n
                    if monitor_fn is not None:
                        monitor_fn(ts_, step, t, u, hef, self.mesh)
                    else:
                        default_monitor(ts_, step, t, u, hef, self.mesh)

            ts.setMonitor(monitor)

        ts.setSolution(y)
        tstart = perf_counter()
        ts.solve(y)
        elapsed = perf_counter() - tstart

        # Synchronize the accepted state before returning it.
        y.copy(hef.m.x.petsc_vec)
        hef.m.x.scatter_forward()

        if self.mesh.comm.rank == 0:
            print(f"\n  ts.solve : {elapsed:.3f} s")
            print(f"  accepted steps         : {ts.getStepNumber()}")
            print(f"  rejected steps         : {ts.getStepRejections()}")
            print(f"  SNES failures          : {ts.getSNESFailures()}")
            print(f"  SNES iterations        : {ts.getSNESIterations()}")
            print(f"  KSP iterations         : {ts.getKSPIterations()}")
            print(f"  Jacobian matvec calls  : {ctx.calls}")
            print(f"  PC applications        : {ctx.callsPre}")
            print(f"  PC side                : {pc_side_key if pc_python else 'none'}")
            print(f"  KSP norm               : {pc_norm_key if pc_python else 'unpreconditioned'}")
            print(f"  PC abs(diag)           : {bool(pc_use_abs_diag) if pc_python else None}")
            print(f"  LLG RHS calls          : {hef.LLGStep}")

        return y, ctx, elapsed
