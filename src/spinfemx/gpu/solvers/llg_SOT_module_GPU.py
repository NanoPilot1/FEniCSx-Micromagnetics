

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

try:
    import adios4dolfinx as ad
    _HAVE_ADIOS = True
except ImportError:
    ad = None
    _HAVE_ADIOS = False

import cupy as cp
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.fem import Constant, form
from dolfinx.fem.petsc import assemble_vector

from ..fields.Exchange_GPU import ExchangeField
from ..fields.Anisotropy_GPU import AnisotropyField
from ..fields.DMI_Bulk_GPU import DMIBULK
from ..fields.DMI_Interfacial_GPU import DMIInterfacial



# PETSc / CuPy helpers

def _set_vec_cuda(vec: PETSc.Vec, block_size: int | None = None) -> PETSc.Vec:
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


def _set_mat_cuda(mat: PETSc.Mat) -> PETSc.Mat:
    try:
        mtype = str(mat.getType()).lower()
    except Exception:
        mtype = ""
    if "cuda" not in mtype:
        try:
            mat = mat.convert(PETSc.Mat.Type.AIJCUSPARSE)
        except Exception:
            mat = mat.convert("aijcusparse")
    try:
        mat.bindToCPU(False)
    except Exception:
        pass
    return mat


def _dup_cuda_vec(template: PETSc.Vec, block_size: int | None = None) -> PETSc.Vec:
    out = template.duplicate()
    _set_vec_cuda(out, block_size=block_size)
    out.zeroEntries()
    return out


def _vec_to_cupy(vec: PETSc.Vec, mode: str = "rw") -> cp.ndarray:
    return cp.from_dlpack(vec.toDLPack(mode))


def _sync_vec_to_function(vec: PETSc.Vec, fun: fem.Function) -> fem.Function:
    vec.copy(fun.x.petsc_vec)
    fun.x.scatter_forward()
    return fun


def _require_single_rank_gpu(mesh, backend_name: str) -> None:
    """Reject unsupported multi-rank execution for the single-GPU backend.

    The current CUDA implementation is intentionally restricted to one Python
    process and one GPU.  Checking MPI.COMM_WORLD as well as mesh.comm catches
    accidental ``mpiexec -n > 1`` launches even when a mesh was created with
    MPI.COMM_SELF.
    """
    world_size = int(MPI.COMM_WORLD.size)
    mesh_size = int(mesh.comm.size)

    if world_size != 1 or mesh_size != 1:
        raise RuntimeError(
            f"{backend_name} supports only single-rank GPU execution. "
            f"Detected MPI.COMM_WORLD.size={world_size} and "
            f"mesh.comm.size={mesh_size}. Run with:\n\n"
            "    python your_script.py\n\n"
            "Do not use mpiexec or mpirun with -n > 1."
        )




def _resolve_ksp_pc_configuration(
    pc_side: str,
    pc_norm_type: str,
):
    """Resolve and validate the Krylov preconditioning convention.

    A Python block preconditioner changes the scaling of the residual.  For
    nonlinear time stepping the safest default is right preconditioning with
    the unpreconditioned residual norm, so the KSP stopping test remains tied
    to the true linear residual rather than to a spatially varying block
    scaling.
    """
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


# Fused CuPy kernels: RHS, Jv and local PC


_LLG_RHS_SOT_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 px, float64 py, float64 pz, "
    "float64 prefactor, float64 alpha, float64 do_precess, "
    "float64 Stab, float64 c_fl, float64 c_dl",
    "float64 rx, float64 ry, float64 rz",
    r"""
    double mcx = my*hz - mz*hy;
    double mcy = mz*hx - mx*hz;
    double mcz = mx*hy - my*hx;

    double mcmx = my*mcz - mz*mcy;
    double mcmy = mz*mcx - mx*mcz;
    double mcmz = mx*mcy - my*mcx;

    double spx = my*pz - mz*py;
    double spy = mz*px - mx*pz;
    double spz = mx*py - my*px;

    double sdpx = my*spz - mz*spy;
    double sdpy = mz*spx - mx*spz;
    double sdpz = mx*spy - my*spx;

    double norm2 = mx*mx + my*my + mz*mz;
    double stab = Stab * (1.0 - norm2);

    rx = prefactor * (do_precess*mcx + alpha*mcmx)
         + c_fl*spx + c_dl*sdpx + stab*mx;
    ry = prefactor * (do_precess*mcy + alpha*mcmy)
         + c_fl*spy + c_dl*sdpy + stab*my;
    rz = prefactor * (do_precess*mcz + alpha*mcmz)
         + c_fl*spz + c_dl*sdpz + stab*mz;
    """,
    name="llg_rhs_sot_gpu_kernel",
)


_JAC_VEC_SOT_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 px, float64 py, float64 pz, "
    "float64 vx, float64 vy, float64 vz, "
    "float64 hvx, float64 hvy, float64 hvz, "
    "float64 prefactor, float64 alpha, float64 do_precess, "
    "float64 Stab, float64 c_fl, float64 c_dl",
    "float64 ox, float64 oy, float64 oz",
    r"""
    // d[v x H + m x Hv]
    double cvx = vy*hz - vz*hy;
    double cvy = vz*hx - vx*hz;
    double cvz = vx*hy - vy*hx;

    double cmx = my*hvz - mz*hvy;
    double cmy = mz*hvx - mx*hvz;
    double cmz = mx*hvy - my*hvx;

    double prec_x = do_precess * (cvx + cmx);
    double prec_y = do_precess * (cvy + cmy);
    double prec_z = do_precess * (cvz + cmz);

    // d[m x (m x H)] using m(m.H) - H(m.m)
    double mdH = mx*hx + my*hy + mz*hz;
    double mdHv = mx*hvx + my*hvy + mz*hvz;
    double vdH = vx*hx + vy*hy + vz*hz;
    double mdv = mx*vx + my*vy + mz*vz;
    double mdm = mx*mx + my*my + mz*mz;
    double common = vdH + mdHv;

    double damp_x = vx*mdH - 2.0*hx*mdv + mx*common - hvx*mdm;
    double damp_y = vy*mdH - 2.0*hy*mdv + my*common - hvy*mdm;
    double damp_z = vz*mdH - 2.0*hz*mdv + mz*common - hvz*mdm;

    // C = m x p and dC = v x p
    double cx = my*pz - mz*py;
    double cy = mz*px - mx*pz;
    double cz = mx*py - my*px;

    double dcx = vy*pz - vz*py;
    double dcy = vz*px - vx*pz;
    double dcz = vx*py - vy*px;

    // d[m x C] = v x C + m x dC
    double vxc_x = vy*cz - vz*cy;
    double vxc_y = vz*cx - vx*cz;
    double vxc_z = vx*cy - vy*cx;

    double mxdc_x = my*dcz - mz*dcy;
    double mxdc_y = mz*dcx - mx*dcz;
    double mxdc_z = mx*dcy - my*dcx;

    double dsot_x = c_fl*dcx + c_dl*(vxc_x + mxdc_x);
    double dsot_y = c_fl*dcy + c_dl*(vxc_y + mxdc_y);
    double dsot_z = c_fl*dcz + c_dl*(vxc_z + mxdc_z);

    double stab_common = Stab * (1.0 - mdm);

    ox = prefactor * (prec_x + alpha*damp_x) + dsot_x
         + stab_common*vx - 2.0*Stab*mx*mdv;
    oy = prefactor * (prec_y + alpha*damp_y) + dsot_y
         + stab_common*vy - 2.0*Stab*my*mdv;
    oz = prefactor * (prec_z + alpha*damp_z) + dsot_z
         + stab_common*vz - 2.0*Stab*mz*mdv;
    """,
    name="jac_vec_sot_gpu_kernel",
)


_PC_BUILD_INV3_SOT_KERNEL = cp.ElementwiseKernel(
    "float64 mx, float64 my, float64 mz, "
    "float64 hx, float64 hy, float64 hz, "
    "float64 px, float64 py, float64 pz, "
    "float64 d0, float64 d1, float64 d2, "
    "float64 shift, float64 eps_reg, float64 det_eps, "
    "float64 prefactor, float64 alpha, float64 do_precess, "
    "float64 Stab, float64 c_fl, float64 c_dl, int32 use_abs_diag",
    "float64 i00, float64 i01, float64 i02, "
    "float64 i10, float64 i11, float64 i12, "
    "float64 i20, float64 i21, float64 i22",
    r"""
    double kx = (use_abs_diag != 0) ? fabs(d0) : d0;
    double ky = (use_abs_diag != 0) ? fabs(d1) : d1;
    double kz = (use_abs_diag != 0) ? fabs(d2) : d2;

    // Column 0: v=e_x, Hv=(kx,0,0)
    double vx = 1.0, vy = 0.0, vz = 0.0;
    double hvx = kx, hvy = 0.0, hvz = 0.0;
    double cvx = vy*hz - vz*hy;
    double cvy = vz*hx - vx*hz;
    double cvz = vx*hy - vy*hx;
    double cmx = my*hvz - mz*hvy;
    double cmy = mz*hvx - mx*hvz;
    double cmz = mx*hvy - my*hvx;
    double mdH = mx*hx + my*hy + mz*hz;
    double mdHv = mx*hvx + my*hvy + mz*hvz;
    double vdH = vx*hx + vy*hy + vz*hz;
    double mdv = mx*vx + my*vy + mz*vz;
    double mdm = mx*mx + my*my + mz*mz;
    double common = vdH + mdHv;
    double dampx = vx*mdH - 2.0*hx*mdv + mx*common - hvx*mdm;
    double dampy = vy*mdH - 2.0*hy*mdv + my*common - hvy*mdm;
    double dampz = vz*mdH - 2.0*hz*mdv + mz*common - hvz*mdm;
    double cx = my*pz - mz*py;
    double cy = mz*px - mx*pz;
    double cz = mx*py - my*px;
    double dcx = vy*pz - vz*py;
    double dcy = vz*px - vx*pz;
    double dcz = vx*py - vy*px;
    double vxcx = vy*cz - vz*cy;
    double vxcy = vz*cx - vx*cz;
    double vxcz = vx*cy - vy*cx;
    double mxdcx = my*dcz - mz*dcy;
    double mxdcy = mz*dcx - mx*dcz;
    double mxdcz = mx*dcy - my*dcx;
    double stab_common = Stab * (1.0 - mdm);
    double j00 = prefactor*(do_precess*(cvx+cmx) + alpha*dampx)
               + c_fl*dcx + c_dl*(vxcx+mxdcx)
               + stab_common*vx - 2.0*Stab*mx*mdv;
    double j10 = prefactor*(do_precess*(cvy+cmy) + alpha*dampy)
               + c_fl*dcy + c_dl*(vxcy+mxdcy)
               + stab_common*vy - 2.0*Stab*my*mdv;
    double j20 = prefactor*(do_precess*(cvz+cmz) + alpha*dampz)
               + c_fl*dcz + c_dl*(vxcz+mxdcz)
               + stab_common*vz - 2.0*Stab*mz*mdv;

    // Column 1: v=e_y, Hv=(0,ky,0)
    vx = 0.0; vy = 1.0; vz = 0.0;
    hvx = 0.0; hvy = ky; hvz = 0.0;
    cvx = vy*hz - vz*hy;
    cvy = vz*hx - vx*hz;
    cvz = vx*hy - vy*hx;
    cmx = my*hvz - mz*hvy;
    cmy = mz*hvx - mx*hvz;
    cmz = mx*hvy - my*hvx;
    mdHv = mx*hvx + my*hvy + mz*hvz;
    vdH = vx*hx + vy*hy + vz*hz;
    mdv = mx*vx + my*vy + mz*vz;
    common = vdH + mdHv;
    dampx = vx*mdH - 2.0*hx*mdv + mx*common - hvx*mdm;
    dampy = vy*mdH - 2.0*hy*mdv + my*common - hvy*mdm;
    dampz = vz*mdH - 2.0*hz*mdv + mz*common - hvz*mdm;
    dcx = vy*pz - vz*py;
    dcy = vz*px - vx*pz;
    dcz = vx*py - vy*px;
    vxcx = vy*cz - vz*cy;
    vxcy = vz*cx - vx*cz;
    vxcz = vx*cy - vy*cx;
    mxdcx = my*dcz - mz*dcy;
    mxdcy = mz*dcx - mx*dcz;
    mxdcz = mx*dcy - my*dcx;
    double j01 = prefactor*(do_precess*(cvx+cmx) + alpha*dampx)
               + c_fl*dcx + c_dl*(vxcx+mxdcx)
               + stab_common*vx - 2.0*Stab*mx*mdv;
    double j11 = prefactor*(do_precess*(cvy+cmy) + alpha*dampy)
               + c_fl*dcy + c_dl*(vxcy+mxdcy)
               + stab_common*vy - 2.0*Stab*my*mdv;
    double j21 = prefactor*(do_precess*(cvz+cmz) + alpha*dampz)
               + c_fl*dcz + c_dl*(vxcz+mxdcz)
               + stab_common*vz - 2.0*Stab*mz*mdv;

    // Column 2: v=e_z, Hv=(0,0,kz)
    vx = 0.0; vy = 0.0; vz = 1.0;
    hvx = 0.0; hvy = 0.0; hvz = kz;
    cvx = vy*hz - vz*hy;
    cvy = vz*hx - vx*hz;
    cvz = vx*hy - vy*hx;
    cmx = my*hvz - mz*hvy;
    cmy = mz*hvx - mx*hvz;
    cmz = mx*hvy - my*hvx;
    mdHv = mx*hvx + my*hvy + mz*hvz;
    vdH = vx*hx + vy*hy + vz*hz;
    mdv = mx*vx + my*vy + mz*vz;
    common = vdH + mdHv;
    dampx = vx*mdH - 2.0*hx*mdv + mx*common - hvx*mdm;
    dampy = vy*mdH - 2.0*hy*mdv + my*common - hvy*mdm;
    dampz = vz*mdH - 2.0*hz*mdv + mz*common - hvz*mdm;
    dcx = vy*pz - vz*py;
    dcy = vz*px - vx*pz;
    dcz = vx*py - vy*px;
    vxcx = vy*cz - vz*cy;
    vxcy = vz*cx - vx*cz;
    vxcz = vx*cy - vy*cx;
    mxdcx = my*dcz - mz*dcy;
    mxdcy = mz*dcx - mx*dcz;
    mxdcz = mx*dcy - my*dcx;
    double j02 = prefactor*(do_precess*(cvx+cmx) + alpha*dampx)
               + c_fl*dcx + c_dl*(vxcx+mxdcx)
               + stab_common*vx - 2.0*Stab*mx*mdv;
    double j12 = prefactor*(do_precess*(cvy+cmy) + alpha*dampy)
               + c_fl*dcy + c_dl*(vxcy+mxdcy)
               + stab_common*vy - 2.0*Stab*my*mdv;
    double j22 = prefactor*(do_precess*(cvz+cmz) + alpha*dampz)
               + c_fl*dcz + c_dl*(vxcz+mxdcz)
               + stab_common*vz - 2.0*Stab*mz*mdv;

    double s = shift + eps_reg;
    double A00 = s - j00;
    double A01 = -j01;
    double A02 = -j02;
    double A10 = -j10;
    double A11 = s - j11;
    double A12 = -j12;
    double A20 = -j20;
    double A21 = -j21;
    double A22 = s - j22;

    double det = A00*(A11*A22 - A12*A21)
               - A01*(A10*A22 - A12*A20)
               + A02*(A10*A21 - A11*A20);

    if (fabs(det) < det_eps) {
        double sign = (det >= 0.0) ? 1.0 : -1.0;
        det += sign*det_eps;
    }

    double invdet = 1.0 / det;
    i00 =  (A11*A22 - A12*A21) * invdet;
    i01 = -(A01*A22 - A02*A21) * invdet;
    i02 =  (A01*A12 - A02*A11) * invdet;
    i10 = -(A10*A22 - A12*A20) * invdet;
    i11 =  (A00*A22 - A02*A20) * invdet;
    i12 = -(A00*A12 - A02*A10) * invdet;
    i20 =  (A10*A21 - A11*A20) * invdet;
    i21 = -(A00*A21 - A01*A20) * invdet;
    i22 =  (A00*A11 - A01*A10) * invdet;
    """,
    name="pc_build_inv3_sot_gpu_kernel",
)


_PC_APPLY_KERNEL = cp.ElementwiseKernel(
    "float64 i00, float64 i01, float64 i02, "
    "float64 i10, float64 i11, float64 i12, "
    "float64 i20, float64 i21, float64 i22, "
    "float64 x0, float64 x1, float64 x2",
    "float64 y0, float64 y1, float64 y2",
    r"""
    y0 = i00*x0 + i01*x1 + i02*x2;
    y1 = i10*x0 + i11*x1 + i12*x2;
    y2 = i20*x0 + i21*x1 + i22*x2;
    """,
    name="pc_apply_sot_gpu_kernel",
)



# Stopping criterion


class StopByMaxDmdtFDGPU:
    def __init__(
        self,
        comm,
        stopping_dm_dt_deg_ns,
        vec_template,
        n_owned_scalar,
        check_every=10,
        print_every_hit=True,
    ):
        self.comm = comm
        self.thresh_deg_ns = float(stopping_dm_dt_deg_ns)
        self.check_every = int(check_every)
        self.print_every_hit = bool(print_every_hit)
        self.n_owned_scalar = int(n_owned_scalar)
        self.n_owned_scalar3 = (self.n_owned_scalar // 3) * 3

        self.u_prev = _dup_cuda_vec(vec_template, block_size=3)
        self.du = _dup_cuda_vec(vec_template, block_size=3)
        vec_template.copy(self.u_prev)

        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")

    def __call__(self, ts):
        if self.thresh_deg_ns <= 0.0:
            return 0

        step = ts.getStepNumber()
        if self.check_every > 1 and (step % self.check_every) != 0:
            return 0

        u = ts.getSolution()
        t = ts.getTime()

        if self.t_prev is None:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        dt = t - self.t_prev
        if dt <= 0.0:
            u.copy(self.u_prev)
            self.t_prev = t
            return 0

        self.u_prev.copy(self.du)
        self.du.scale(-1.0)
        self.du.axpy(1.0, u)

        du_all = _vec_to_cupy(self.du, "r")
        du = du_all[: self.n_owned_scalar3].reshape((-1, 3))
        if du.size:
            max_local = float(cp.sqrt((du * du).sum(axis=1)).max().item()) / dt
        else:
            max_local = 0.0

        max_global = self.comm.allreduce(max_local, op=MPI.MAX)
        max_deg_ns = max_global * (180.0 / np.pi) * 1e-9
        self.last_max_dmdt_deg_ns = max_deg_ns

        if self.comm.rank == 0 and self.print_every_hit:
            print(
                f"[dmdt] step={step} t={t*1e9:.6f} ns  "
                f"max|dm/dt|={max_deg_ns:.6e} deg/ns",
                flush=True,
            )

        if max_deg_ns < self.thresh_deg_ns:
            ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)

        u.copy(self.u_prev)
        self.t_prev = t
        return 0

    def reset(self, vec_current):
        vec_current.copy(self.u_prev)
        self.t_prev = None
        self.last_max_dmdt_deg_ns = float("nan")



# Effective field + SOT on GPU


class EffectiveFieldSOTGPU:
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
        H0_static=None,
        gamma=2.211e5,
        alpha=0.5,
        do_precess=1,
        H_dl=0.0,
        H_fl=0.0,
        polarization_vec=None,
        use_demag=False,
        demag_method="fmm",
        demag_kwargs=None,
        H_time_func=None,
    ):
        _require_single_rank_gpu(mesh, "EffectiveFieldSOTGPU")

        self.mesh = mesh
        self.comm = mesh.comm
        self.V = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        self.V1 = fem.functionspace(mesh, ("Lagrange", 1))

        self.Ms = float(Ms)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.A = float(Aex)
        self.Ku = float(Ku)
        self.D_bulk = float(D_bulk)
        self.D_int = float(D_int)
        self.do_precess = float(do_precess)
        self.use_demag = bool(use_demag)
        self.H_time_func = H_time_func
        self.current_time = 0.0
        self.mu0 = 4.0 * np.pi * 1e-7
        self.volume_scale_energy = 1e-27  # mesh coordinates are expected in nm

        self.H_dl = float(H_dl)
        self.H_fl = float(H_fl)
        self._update_sot_coefficients()

        self.prefactor = -self.gamma / (1.0 + self.alpha**2)
        self.Stab = self.Ms * self.gamma / (1.0 + self.alpha**2) * 0.5

        try:
            self.mesh.name = "Grid"
        except Exception:
            pass

        # Host-side functions used only for I/O and selected diagnostics.
        self.m = fem.Function(self.V, name="f")
        self.dmdt = fem.Function(self.V, name="dmdt")
        self.H_eff = fem.Function(self.V, name="H_eff")

        self.start, self.end = self.V.dofmap.index_map.local_range
        self.local_dofs = int(self.end - self.start)
        self.local_size = 3 * self.local_dofs

        # Spatial-resolution diagnostic. Build the mesh-edge list only once.
        # In the common single-rank GPU path, the edge endpoints are also kept
        # on the device so max_neighbor_angle_deg_gpu() avoids CPU transfers.
        self.neighbor_pairs_host = self._build_neighbor_pairs()
        if self.neighbor_pairs_host.size:
            self.neighbor_i_gpu = cp.asarray(
                self.neighbor_pairs_host[:, 0],
                dtype=cp.int32,
            )
            self.neighbor_j_gpu = cp.asarray(
                self.neighbor_pairs_host[:, 1],
                dtype=cp.int32,
            )
        else:
            self.neighbor_i_gpu = None
            self.neighbor_j_gpu = None

        # Lumped nodal volumes.
        v = ufl.TestFunction(self.V)
        tmp = ufl.dot(v, Constant(self.mesh, PETSc.ScalarType((1.0, 1.0, 1.0)))) * ufl.dx
        volN_f = fem.Function(self.V)
        volN_f.x.petsc_vec.set(0.0)
        assemble_vector(volN_f.x.petsc_vec, form(tmp))
        volN_f.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        volN_f.x.scatter_forward()
        volN = volN_f.x.array.copy()
        self.volN = volN
        self.vol_nodes = np.asarray(volN[: self.local_size], dtype=np.float64).reshape((-1, 3))[:, 0]
        self.vol_nodes_gpu = cp.asarray(self.vol_nodes)

        # Linear and local magnetic operators.
        self.exchange_field = None
        self.anisotropy_field = None
        self.DMIBULK = None
        self.DMI_int = None
        self.demag_field = None
        self.H_demag_gpu = None
        self.linear_terms: list[tuple[str, PETSc.Mat, PETSc.Vec]] = []
        self.local_terms = []
        # Host matrices used only once to assemble the combined sparse CUDA operator.
        cpu_lin_mats = []

        template_in = None
        template_out = None

        if abs(self.A) > 0.0:
            self.exchange_field = ExchangeField(self.mesh, self.V, self.A, self.Ms, volN)
            cpu_lin_mats.append(self.exchange_field.K)
            self.exchange_field.K = _set_mat_cuda(self.exchange_field.K)
            buf = self.exchange_field.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("exchange", self.exchange_field.K, buf))
            template_in = self.exchange_field.K.createVecRight()
            template_out = self.exchange_field.K.createVecLeft()

        if abs(self.Ku) > 0.0:
            self.anisotropy_field = AnisotropyField(self.mesh, self.V, self.Ku, self.Ms, n_ani_vec, volN)
            if hasattr(self.anisotropy_field, "K"):
                cpu_lin_mats.append(self.anisotropy_field.K)
                self.anisotropy_field.K = _set_mat_cuda(self.anisotropy_field.K)
                buf = self.anisotropy_field.K.createVecLeft()
                _set_vec_cuda(buf, block_size=3)
                self.linear_terms.append(("anisotropy", self.anisotropy_field.K, buf))
                if template_in is None:
                    template_in = self.anisotropy_field.K.createVecRight()
                    template_out = self.anisotropy_field.K.createVecLeft()
            elif hasattr(self.anisotropy_field, "compute_vec"):
                buf = getattr(self.anisotropy_field, "h_gpu", None)
                if buf is None:
                    buf = self.m.x.petsc_vec.duplicate()
                _set_vec_cuda(buf, block_size=3)
                self.local_terms.append(("anisotropy", self.anisotropy_field, buf))
            else:
                raise TypeError(
                    "AnisotropyField must expose either .K or "
                    "compute_vec(m_vec, out_vec)."
                )

        if abs(self.D_bulk) > 0.0:
            self.DMIBULK = DMIBULK(self.mesh, self.V, self.V1, self.D_bulk, self.Ms, volN)
            cpu_lin_mats.append(self.DMIBULK.K)
            self.DMIBULK.K = _set_mat_cuda(self.DMIBULK.K)
            buf = self.DMIBULK.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("dmi_bulk", self.DMIBULK.K, buf))
            if template_in is None:
                template_in = self.DMIBULK.K.createVecRight()
                template_out = self.DMIBULK.K.createVecLeft()

        if abs(self.D_int) > 0.0:
            self.DMI_int = DMIInterfacial(self.mesh, self.V, self.V1, self.D_int, n0_int_vec, self.Ms, volN)
            cpu_lin_mats.append(self.DMI_int.K)
            self.DMI_int.K = _set_mat_cuda(self.DMI_int.K)
            buf = self.DMI_int.K.createVecLeft()
            _set_vec_cuda(buf, block_size=3)
            self.linear_terms.append(("dmi_interfacial", self.DMI_int.K, buf))
            if template_in is None:
                template_in = self.DMI_int.K.createVecRight()
                template_out = self.DMI_int.K.createVecLeft()

        # SOT is local

        # Combine sparse linear magnetic operators once.  Per-field matrices
        # are retained for diagnostics and energy decomposition.  The hot RHS
        # and Jv paths use one cuSPARSE matvec whenever several sparse terms
        # are active.  With exactly one sparse term, reuse its CUDA matrix
        # directly to avoid an unnecessary GPU allocation.
        if len(cpu_lin_mats) == 0:
            self.K_lin = None
        elif len(cpu_lin_mats) == 1:
            self.K_lin = self.linear_terms[0][1]
        else:
            K_lin_cpu = cpu_lin_mats[0].copy()
            for Kc in cpu_lin_mats[1:]:
                K_lin_cpu.axpy(
                    1.0,
                    Kc,
                    structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
                )
            K_lin_cpu.assemble()
            self.K_lin = _set_mat_cuda(K_lin_cpu)

        
        if template_in is None:
            template_in = self.m.x.petsc_vec.duplicate()
            template_out = self.m.x.petsc_vec.duplicate()

        _set_vec_cuda(template_in, block_size=3)
        _set_vec_cuda(template_out, block_size=3)

        # Persistent device vectors.
        self.m_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.dmdt_gpu = _dup_cuda_vec(template_in, block_size=3)
        self.H_eff_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hm_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hv_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.H0_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Ht_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Hext_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.p_gpu = _dup_cuda_vec(template_out, block_size=3)
        self.Jv_buffer = _dup_cuda_vec(template_in, block_size=3)
        self.M_state_gpu = _dup_cuda_vec(template_in, block_size=3)

        # Diagonal of the local/sparse field approximation used by the PC.
        self.diagK = _dup_cuda_vec(template_out, block_size=3)
        self.diagK.zeroEntries()
        for _, K, _ in self.linear_terms:
            d = K.getDiagonal()
            _set_vec_cuda(d, block_size=3)
            self.diagK.axpy(1.0, d)
        for _, term, buf in self.local_terms:
            if hasattr(term, "diagonal_vec"):
                term.diagonal_vec(buf)
                self.diagK.axpy(1.0, buf)

        self.diagK_abs = _dup_cuda_vec(template_out, block_size=3)
        self.diagK.copy(self.diagK_abs)
        self.diagK_abs.abs()

        self._set_gpu_vector_field(self.H0_gpu, H0_static, normalize=False, name="H0_static")
        if polarization_vec is None:
            polarization_vec = (0.0, 1.0, 0.0)
        self._set_gpu_vector_field(self.p_gpu, polarization_vec, normalize=True, name="polarization_vec")

        if self.use_demag:
            method = str(demag_method).lower()
            kwargs = {} if demag_kwargs is None else dict(demag_kwargs)
            if method in ["fmm", "jaxfmm"]:
                from ..fields.Demag_FMM_GPU import DemagFieldFMMJAXGPU
                mem_limit = kwargs.pop("mem_limit", 4_000_000)
                self.demag_field = DemagFieldFMMJAXGPU(
                    self.mesh,
                    self.V,
                    self.V1,
                    self.Ms,
                    volN,
                    mem_limit=mem_limit,
                    **kwargs,
                )
            elif method in ["lindholm", "lindholm_gpu"]:
                from ..fields.Demag_Lindholm_GPU import DemagFieldLindholmGPU
                self.demag_field = DemagFieldLindholmGPU(
                    mesh=self.mesh,
                    V=self.V,
                    V1=self.V1,
                    Ms=self.Ms,
                    VolN=volN,
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    "Unsupported demag_method. Use 'jaxfmm', 'fmm', "
                    "'lindholm', or 'lindholm_gpu'."
                )
            self.H_demag_gpu = _dup_cuda_vec(template_out, block_size=3)

        self.LLGSteps = 0
        self.JacSteps = 0


    # SOT configuration

    def _update_sot_coefficients(self):
        den = 1.0 + self.alpha * self.alpha
        self.c_fl = -self.gamma * (self.H_fl - self.alpha * self.H_dl) / den
        self.c_dl = -self.gamma * (self.H_dl + self.alpha * self.H_fl) / den

    def _set_gpu_vector_field(self, vec, values, *, normalize, name):
        vec.zeroEntries()
        if values is None:
            return

        target = _vec_to_cupy(vec, "rw")
        owned = target[: self.local_size].reshape((-1, 3))

        if isinstance(values, PETSc.Vec):
            values.copy(vec)
            target = _vec_to_cupy(vec, "rw")
            owned = target[: self.local_size].reshape((-1, 3))
        else:
            arr = values if isinstance(values, cp.ndarray) else cp.asarray(np.asarray(values, dtype=np.float64))
            if arr.ndim == 1 and arr.size == 3:
                owned[:, 0] = arr[0]
                owned[:, 1] = arr[1]
                owned[:, 2] = arr[2]
            elif arr.ndim == 1 and arr.size >= self.local_size:
                target[: self.local_size] = arr[: self.local_size]
            elif arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= self.local_dofs:
                owned[:, :] = arr[: self.local_dofs, :]
            else:
                raise ValueError(
                    f"{name} must have shape (3,), ({self.local_size},), "
                    f"or ({self.local_dofs}, 3). Got {arr.shape}."
                )

        if normalize:
            norms = cp.sqrt((owned * owned).sum(axis=1))
            mask = norms > 0.0
            owned[mask, :] /= norms[mask, None]

        if target.size > self.local_size:
            target[self.local_size:] = 0.0

    def set_sot(self, H_dl=None, H_fl=None, polarization_vec=None):
        if H_dl is not None:
            self.H_dl = float(H_dl)
        if H_fl is not None:
            self.H_fl = float(H_fl)
        if polarization_vec is not None:
            self._set_gpu_vector_field(self.p_gpu, polarization_vec, normalize=True, name="polarization_vec")
        self._update_sot_coefficients()


    # Host/device synchronization

    def set_m_from_cpu(self, m0_array):
        arr = np.asarray(m0_array, dtype=np.float64).reshape(-1)
        if arr.size == self.m.x.array.size:
            self.m.x.array[:] = arr
        elif arr.size == self.local_size:
            self.m.x.array[: self.local_size] = arr
            if self.m.x.array.size > self.local_size:
                self.m.x.array[self.local_size:] = 0.0
        else:
            raise ValueError(
                f"m0_array has {arr.size} entries; expected {self.local_size} "
                f"owned entries or {self.m.x.array.size} local+ghost entries."
            )
        self.m.x.scatter_forward()
        self.m.x.petsc_vec.copy(self.m_gpu)

    def _build_neighbor_pairs(self):
        """Build unique mesh-edge pairs for first-order simplex meshes.

        The magnetic space is a blocked three-component P1 space.  Its cell
        dofs index the nodal vectors stored by ``m.reshape((-1, 3))`` directly,
        so the diagnostic does not depend on a separate scalar-space ordering.
        """
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        if cell_map is None:
            raise RuntimeError("Missing cell index map.")

        if self.V.dofmap.index_map_bs != 3:
            raise RuntimeError(
                "Expected a blocked magnetic function space with three components."
            )

        n_cells = int(cell_map.size_local + cell_map.num_ghosts)
        chunks = []

        for cell in range(n_cells):
            dofs = np.asarray(self.V.dofmap.cell_dofs(cell), dtype=np.int32)

            # Every pair of P1 simplex vertices is an edge.  The same rule is
            # not valid for hexahedra or prisms because it would include cell
            # diagonals.
            if dofs.size != tdim + 1:
                raise NotImplementedError(
                    "max_neighbor_angle_deg_gpu currently supports only "
                    "first-order simplex meshes: triangles and tetrahedra."
                )

            ia, ib = np.triu_indices(dofs.size, k=1)
            pairs = np.column_stack((dofs[ia], dofs[ib])).astype(
                np.int32,
                copy=False,
            )
            pairs.sort(axis=1)
            chunks.append(pairs)

        if not chunks:
            return np.empty((0, 2), dtype=np.int32)

        return np.unique(np.vstack(chunks), axis=0).astype(
            np.int32,
            copy=False,
        )

    def max_neighbor_angle_deg_gpu(self, m_vec: PETSc.Vec | None = None):
        """Return the maximum angle between neighboring nodal moments.

        This diagnostic is evaluated only in the supported single-rank CUDA
        path.  All vector algebra stays on the GPU and only the final scalar is
        copied to the host.
        """
        if self.comm.size != 1:
            raise RuntimeError(
                "max_neighbor_angle_deg_gpu() supports single-rank GPU "
                "execution only."
            )

        if self.neighbor_pairs_host.size == 0:
            return 0.0

        if m_vec is None:
            m_vec = self.m_gpu

        m_all = _vec_to_cupy(m_vec, "r")
        moments = m_all[: self.local_size].reshape((-1, 3))

        mi = moments[self.neighbor_i_gpu]
        mj = moments[self.neighbor_j_gpu]

        dot = cp.sum(mi * mj, axis=1)
        norm_product = cp.sqrt(
            cp.sum(mi * mi, axis=1) * cp.sum(mj * mj, axis=1)
        )

        # Avoid an intermediate division by zero before applying the mask.
        safe_norm_product = cp.maximum(norm_product, 1.0e-30)
        cosine = dot / safe_norm_product
        cosine = cp.where(norm_product > 1.0e-30, cosine, 1.0)
        cosine = cp.clip(cosine, -1.0, 1.0)

        return float(
            (cp.arccos(cosine) * (180.0 / np.pi)).max().item()
        )

    def sync_m_to_function(self):
        return _sync_vec_to_function(self.m_gpu, self.m)

    def sync_H_to_function(self):
        return _sync_vec_to_function(self.H_eff_gpu, self.H_eff)

    def set_uniform_field(self, Hx, Hy, Hz):
        self._set_gpu_vector_field(
            self.H0_gpu,
            np.array([Hx, Hy, Hz], dtype=np.float64),
            normalize=False,
            name="H0_static",
        )

    def set_time_dependent_field(self, H_time_func):
        self.H_time_func = H_time_func

    def clear_time_dependent_field(self):
        self.H_time_func = None
        self.Ht_gpu.zeroEntries()


    # External and magnetic fields

    def _eval_H_time_vec(self, t):
        self.Ht_gpu.zeroEntries()
        if self.H_time_func is None:
            return None

        try:
            out = self.H_time_func(float(t), self.mesh.geometry.x)
        except TypeError:
            out = self.H_time_func(float(t))

        H = _vec_to_cupy(self.Ht_gpu, "rw")
        owned = H[: self.local_size].reshape((-1, 3))

        if isinstance(out, PETSc.Vec):
            out.copy(self.Ht_gpu)
            return self.Ht_gpu

        arr = out if isinstance(out, cp.ndarray) else cp.asarray(np.asarray(out, dtype=np.float64))
        if arr.ndim == 1 and arr.size == 3:
            owned[:, 0] = arr[0]
            owned[:, 1] = arr[1]
            owned[:, 2] = arr[2]
        elif arr.ndim == 1 and arr.size >= self.local_size:
            H[: self.local_size] = arr[: self.local_size]
        elif arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= self.local_dofs:
            owned[:, :] = arr[: self.local_dofs, :]
        else:
            raise ValueError(f"Unsupported H_time_func output shape: {arr.shape}")

        if H.size > self.local_size:
            H[self.local_size:] = 0.0
        return self.Ht_gpu

    def _external_field_vec(self):
        self.H0_gpu.copy(self.Hext_gpu)
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            self.Hext_gpu.axpy(1.0, self.Ht_gpu)
        return self.Hext_gpu

    def apply_linear_field_vec(self, x_vec: PETSc.Vec, out_vec: PETSc.Vec):
        if self.K_lin is not None:
            self.K_lin.mult(x_vec, out_vec)
        else:
            out_vec.zeroEntries()

        for _, term, buf in self.local_terms:
            term.compute_vec(x_vec, buf)
            out_vec.axpy(1.0, buf)

    def compute_H_eff_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.apply_linear_field_vec(m_vec, out_vec)
        if self.demag_field is not None:
            self.demag_field.compute_vec(m_vec, self.H_demag_gpu)
            out_vec.axpy(1.0, self.H_demag_gpu)
        out_vec.axpy(1.0, self.H0_gpu)
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)

    def compute_H_jac_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """Field used by Jv/PC: magnetic local/linear terms + external field.

        Demag is excluded deliberately to keep Jv and the PC inexpensive.
        """
        self.apply_linear_field_vec(m_vec, out_vec)
        out_vec.axpy(1.0, self.H0_gpu)
        if self.H_time_func is not None:
            self._eval_H_time_vec(self.current_time)
            out_vec.axpy(1.0, self.Ht_gpu)


    # RHS and matrix-free Jacobian action

    def llg_rhs_SOT_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.LLGSteps += 1
        self.compute_H_eff_vec(m_vec, self.H_eff_gpu)

        m_all = _vec_to_cupy(m_vec, "r")
        h_all = _vec_to_cupy(self.H_eff_gpu, "r")
        p_all = _vec_to_cupy(self.p_gpu, "r")
        rhs_all = _vec_to_cupy(out_vec, "rw")

        M = m_all[: self.local_size].reshape((-1, 3))
        H = h_all[: self.local_size].reshape((-1, 3))
        P = p_all[: self.local_size].reshape((-1, 3))
        RHS = rhs_all[: self.local_size].reshape((-1, 3))

        _LLG_RHS_SOT_KERNEL(
            M[:, 0], M[:, 1], M[:, 2],
            H[:, 0], H[:, 1], H[:, 2],
            P[:, 0], P[:, 1], P[:, 2],
            float(self.prefactor),
            float(self.alpha),
            float(self.do_precess),
            float(self.Stab),
            float(self.c_fl),
            float(self.c_dl),
            RHS[:, 0], RHS[:, 1], RHS[:, 2],
        )
        if rhs_all.size > self.local_size:
            rhs_all[self.local_size:] = 0.0

    def rhs_function(self, ts, t, y, f):
        self.current_time = float(t)
        self.llg_rhs_SOT_vec(y, f)
        return 0

    def ifunction_SOT(self, ts, t, y, ydot, f):
        self.current_time = float(t)
        self.llg_rhs_SOT_vec(y, self.dmdt_gpu)
        ydot.copy(f)
        f.axpy(-1.0, self.dmdt_gpu)
        return 0

    def update_jac_state_SOT(self, m_vec: PETSc.Vec):
        m_vec.copy(self.M_state_gpu)
        self.compute_H_jac_vec(m_vec, self.Hm_gpu)

    def jac_vec_times_SOT_vec(self, v_vec: PETSc.Vec, out_vec: PETSc.Vec):
        self.JacSteps += 1
        self.apply_linear_field_vec(v_vec, self.Hv_gpu)

        m_all = _vec_to_cupy(self.M_state_gpu, "r")
        hm_all = _vec_to_cupy(self.Hm_gpu, "r")
        p_all = _vec_to_cupy(self.p_gpu, "r")
        v_all = _vec_to_cupy(v_vec, "r")
        hv_all = _vec_to_cupy(self.Hv_gpu, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        M = m_all[: self.local_size].reshape((-1, 3))
        Hm = hm_all[: self.local_size].reshape((-1, 3))
        P = p_all[: self.local_size].reshape((-1, 3))
        V = v_all[: self.local_size].reshape((-1, 3))
        Hv = hv_all[: self.local_size].reshape((-1, 3))
        OUT = out_all[: self.local_size].reshape((-1, 3))

        _JAC_VEC_SOT_KERNEL(
            M[:, 0], M[:, 1], M[:, 2],
            Hm[:, 0], Hm[:, 1], Hm[:, 2],
            P[:, 0], P[:, 1], P[:, 2],
            V[:, 0], V[:, 1], V[:, 2],
            Hv[:, 0], Hv[:, 1], Hv[:, 2],
            float(self.prefactor),
            float(self.alpha),
            float(self.do_precess),
            float(self.Stab),
            float(self.c_fl),
            float(self.c_dl),
            OUT[:, 0], OUT[:, 1], OUT[:, 2],
        )
        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0


    # Energy diagnostics

    def _external_energy_global(self):
        ext = self._external_field_vec()
        m_all = _vec_to_cupy(self.m_gpu, "r")
        h_all = _vec_to_cupy(ext, "r")
        M = m_all[: self.local_size].reshape((-1, 3))
        H = h_all[: self.local_size].reshape((-1, 3))
        local = float(cp.sum(self.vol_nodes_gpu * cp.sum(M * H, axis=1)).item())
        global_sum = float(self.comm.allreduce(local, op=MPI.SUM))
        return -self.mu0 * self.Ms * self.volume_scale_energy * global_sum

    def compute_Energy_terms(self):
        m_fun = self.sync_m_to_function()

        E_exch = 0.0
        E_ani = 0.0
        E_dmi_bulk = 0.0
        E_dmi_int = 0.0
        E_demag = 0.0

        if self.exchange_field is not None:
            E_exch = float(self.exchange_field.Energy(m_fun))
        if self.anisotropy_field is not None:
            E_ani = float(self.anisotropy_field.Energy(m_fun))
        if self.demag_field is not None:
            self.demag_field.compute_vec(self.m_gpu, self.H_demag_gpu)
            E_demag = float(self.demag_field.Energy_lumped_gpu(self.m_gpu, self.H_demag_gpu))
        if self.DMIBULK is not None:
            self.DMIBULK.compute(self.m_gpu)
            E_dmi_bulk = float(self.DMIBULK.Energy(m_fun))
        if self.DMI_int is not None:
            self.DMI_int.compute(self.m_gpu)
            E_dmi_int = float(self.DMI_int.Energy(m_fun))

        E_ext = self._external_energy_global()
        E_total = E_exch + E_demag + E_ani + E_dmi_bulk + E_dmi_int + E_ext
        return {
            "E_demag": E_demag,
            "E_exch": E_exch,
            "E_ani": E_ani,
            "E_dmi_bulk": E_dmi_bulk,
            "E_dmi_int": E_dmi_int,
            "E_ext": E_ext,
            "E_total": E_total,
        }



# Matrix-free context and local block-Jacobi PC


class JvContextSOT:
    def __init__(self, hef: EffectiveFieldSOTGPU, eps_reg=1e-14, det_eps=1e-30):
        self.hef = hef
        self.shift = 0.0
        self.calls = 0
        self.callsPre = 0
        self.eps_reg = float(eps_reg)
        self.det_eps = float(det_eps)
        self.enable_pc = True
        self._pc_ready = False

        n = self.hef.local_dofs
        self.i00 = cp.empty(n); self.i01 = cp.empty(n); self.i02 = cp.empty(n)
        self.i10 = cp.empty(n); self.i11 = cp.empty(n); self.i12 = cp.empty(n)
        self.i20 = cp.empty(n); self.i21 = cp.empty(n); self.i22 = cp.empty(n)

    def update_pc_full_fast_gpu(self, shift, use_abs_diag=False):
        self.shift = float(shift)
        local_size = self.hef.local_size

        M_all = _vec_to_cupy(self.hef.M_state_gpu, "r")
        H_all = _vec_to_cupy(self.hef.Hm_gpu, "r")
        P_all = _vec_to_cupy(self.hef.p_gpu, "r")
        D_all = _vec_to_cupy(self.hef.diagK, "r")

        M = M_all[:local_size].reshape((-1, 3))
        H = H_all[:local_size].reshape((-1, 3))
        P = P_all[:local_size].reshape((-1, 3))
        D = D_all[:local_size].reshape((-1, 3))

        _PC_BUILD_INV3_SOT_KERNEL(
            M[:, 0], M[:, 1], M[:, 2],
            H[:, 0], H[:, 1], H[:, 2],
            P[:, 0], P[:, 1], P[:, 2],
            D[:, 0], D[:, 1], D[:, 2],
            float(self.shift),
            float(self.eps_reg),
            float(self.det_eps),
            float(self.hef.prefactor),
            float(self.hef.alpha),
            float(self.hef.do_precess),
            float(self.hef.Stab),
            float(self.hef.c_fl),
            float(self.hef.c_dl),
            np.int32(1 if use_abs_diag else 0),
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
        )
        self._pc_ready = True

    def apply(self, pc, x, y):
        self.callsPre += 1
        if (not self.enable_pc) or (not self._pc_ready):
            x.copy(y)
            return

        local_size = self.hef.local_size
        x_all = _vec_to_cupy(x, "r")
        y_all = _vec_to_cupy(y, "rw")
        X = x_all[:local_size].reshape((-1, 3))
        Y = y_all[:local_size].reshape((-1, 3))
        _PC_APPLY_KERNEL(
            self.i00, self.i01, self.i02,
            self.i10, self.i11, self.i12,
            self.i20, self.i21, self.i22,
            X[:, 0], X[:, 1], X[:, 2],
            Y[:, 0], Y[:, 1], Y[:, 2],
        )
        if y_all.size > local_size:
            y_all[local_size:] = 0.0

    def mult(self, A, x, y):
        self.calls += 1
        self.hef.jac_vec_times_SOT_vec(x, self.hef.Jv_buffer)
        x.copy(y)
        y.scale(self.shift)
        y.axpy(-1.0, self.hef.Jv_buffer)



# Public driver


class LLG_SOT_GPU:
    def __init__(self, mesh, Ms, gamma=2.211e5, alpha=0.5, do_precess=1):
        _require_single_rank_gpu(mesh, "LLG_SOT_GPU")

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
        self._H_time_func = None
        self._H_dl = 0.0
        self._H_fl = 0.0
        self._polarization_vec = None

        self._has_exchange = False
        self._has_demag = False
        self._has_anisotropy = False
        self._has_dmi_bulk = False
        self._has_dmi_int = False
        self._has_H0 = False
        self._has_sot = False

        self._demag_method = "fmm"
        self._demag_kwargs = {}

        self.hef: EffectiveFieldSOTGPU | None = None
        self.ts = None
        self.y = None
        self.stopper = None
        self.ctx = None
        self.J = None
        self._solver_ready = False

    def add_exchange(self, Aex):
        self._Aex = float(Aex)
        self._has_exchange = True

    def add_demag(self, method="fmm", **kwargs):
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

    def add_external_field(self, H0_vec=None, H_time_func=None):
        self._H0_vec = H0_vec
        self._H_time_func = H_time_func
        self._has_H0 = H0_vec is not None
        if self.hef is not None:
            if H0_vec is not None:
                self.hef._set_gpu_vector_field(self.hef.H0_gpu, H0_vec, normalize=False, name="H0_vec")
            self.hef.set_time_dependent_field(H_time_func)

    def add_sot(self, H_dl, H_fl=0.0, polarization_vec=(0.0, 1.0, 0.0)):
        """Enable local SOT with scalar amplitudes in A/m."""
        self._H_dl = float(H_dl)
        self._H_fl = float(H_fl)
        self._polarization_vec = polarization_vec
        self._has_sot = True
        if self.hef is not None:
            self.hef.set_sot(H_dl=H_dl, H_fl=H_fl, polarization_vec=polarization_vec)

    def add_sot_from_spin_hall(
        self,
        J,
        theta_sh,
        thickness,
        polarization_vec=(0.0, 1.0, 0.0),
        H_fl=0.0,
    ):
        """Convenience wrapper: H_dl = hbar*theta_sh*J/(2*e*mu0*Ms*t)."""
        hbar = 1.054571817e-34
        e = 1.602176634e-19
        mu0 = 4.0 * np.pi * 1e-7
        H_dl = hbar * float(theta_sh) * float(J) / (2.0 * e * mu0 * self.Ms * float(thickness))
        self.add_sot(H_dl=H_dl, H_fl=H_fl, polarization_vec=polarization_vec)
        return H_dl

    def set_uniform_field(self, Hx, Hy, Hz):
        self._H0_vec = np.array([Hx, Hy, Hz], dtype=np.float64)
        self._has_H0 = True
        if self.hef is not None:
            self.hef.set_uniform_field(Hx, Hy, Hz)

    def set_time_dependent_field(self, H_time_func):
        self._H_time_func = H_time_func
        if self.hef is not None:
            self.hef.set_time_dependent_field(H_time_func)

    def clear_time_dependent_field(self):
        self._H_time_func = None
        if self.hef is not None:
            self.hef.clear_time_dependent_field()

    def set_sot(self, H_dl=None, H_fl=None, polarization_vec=None):
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

        self.hef = EffectiveFieldSOTGPU(
            self.mesh,
            self.Ms,
            Aex,
            Ku,
            n_ani_vec,
            D_bulk,
            D_int,
            n0_int_vec,
            H0_static=H0_vec,
            gamma=self.gamma,
            alpha=self.alpha,
            do_precess=self.do_precess,
            H_dl=H_dl,
            H_fl=H_fl,
            polarization_vec=pvec,
            use_demag=self._has_demag,
            demag_method=self._demag_method,
            demag_kwargs=self._demag_kwargs,
            H_time_func=self._H_time_func,
        )

    def _cancel_ts_monitors(self):
        if self.ts is None:
            return
        try:
            self.ts.monitorCancel()
        except Exception:
            try:
                self.ts.setMonitor(None)
            except Exception:
                pass

    def _prepare_function_for_io(self):
        if self.hef is None:
            return
        self.hef.sync_m_to_function()
        try:
            self.mesh.name = "Grid"
        except Exception:
            pass
        try:
            self.hef.m.name = "m"
        except Exception:
            pass

    def _ensure_solver_bdf(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-4,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
        pc_python=True,
        pc_use_abs_diag=False,
        pc_side="right",
        pc_norm_type="unpreconditioned",
    ):
        if self.hef is None:
            self._build_effective_field()
        hef = self.hef
        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)

        (
            pc_side_key,
            pc_norm_key,
            pc_side_value,
            pc_norm_value,
        ) = _resolve_ksp_pc_configuration(
            pc_side=pc_side,
            pc_norm_type=pc_norm_type,
        )

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
        opts["snes_rtol"] = snes_rtol
        opts["snes_atol"] = snes_atol
        opts["snes_max_it"] = 8
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = ksp_rtol
        opts["ts_max_snes_failures"] = -1
        opts["ts_max_steps"] = 5_000_000

        ts.setTime(0.0)
        ts.setTimeStep(float(dt_init))
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        snes = ts.getSNES()
        n_loc = hef.m_gpu.getLocalSize()
        n_glob = hef.m_gpu.getSize()
        J = PETSc.Mat().create(comm=self.mesh.comm)
        ctx = JvContextSOT(hef)
        J.setSizes([[n_loc, n_glob], [n_loc, n_glob]])
        J.setType("python")
        J.setPythonContext(ctx)
        J.setUp()

        ksp = snes.getKSP()
        ksp.setType(PETSc.KSP.Type.GMRES)
        ksp.setTolerances(rtol=ksp_rtol, max_it=200)
        pc = ksp.getPC()
        if pc_python:
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(ctx)
        else:
            pc.setType(PETSc.PC.Type.NONE)
            ctx.enable_pc = False

        def IJac(ts_, t, y, ydot, shift, A, B):
            hef.current_time = float(t)
            hef.update_jac_state_SOT(y)
            if pc_python:
                # diagK is the signed diagonal of dH/dm.  Taking its absolute
                # value reverses the exchange-field sign and produces an
                # anti-preconditioner on the racetrack problem.
                ctx.update_pc_full_fast_gpu(
                    float(shift),
                    use_abs_diag=bool(pc_use_abs_diag),
                )
            else:
                ctx.shift = float(shift)
            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

        F_gpu = _dup_cuda_vec(hef.m_gpu, block_size=3)
        ts.setIFunction(hef.ifunction_SOT, F_gpu)
        ts.setIJacobian(IJac, J, J)
        ts.setFromOptions()

        # Apply the explicit API choice after PETSc options have been parsed.
        # This prevents ambient PETSc options from silently replacing the
        # preconditioner requested through solve(...).  Right preconditioning
        # preserves the true residual b-Ax; the signed local block remains fixed
        # during each individual KSP solve.
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

        self._pc_bdf_config = {
            "enabled": bool(pc_python),
            "use_abs_diag": bool(pc_use_abs_diag),
            "side": pc_side_key if pc_python else "none",
            "norm_type": pc_norm_key if pc_python else "unpreconditioned",
        }

        y = _dup_cuda_vec(hef.m_gpu, block_size=3)
        hef.m_gpu.copy(y)
        ts.setSolution(y)

        stopper = StopByMaxDmdtFDGPU(
            self.mesh.comm,
            stopping_dm_dt_deg_ns=stopping_dmdt,
            vec_template=y,
            n_owned_scalar=hef.local_size,
            check_every=check_every_stop,
            print_every_hit=stop_print,
        )
        ts.setPostStep(stopper)

        self.ts = ts
        self.y = y
        self.stopper = stopper
        self.ctx = ctx
        self.J = J
        self._solver_ready = True

    def _install_monitor(self, *, dt_save, dt_snap, output_dir, monitor_fn, prefix):
        if dt_save is None:
            return
        if dt_save <= 0.0:
            raise ValueError("dt_save must be positive.")
        if dt_snap is None:
            dt_snap = dt_save
        if dt_snap <= 0.0:
            raise ValueError("dt_snap must be positive.")

        hef = self.hef
        log_path = Path(output_dir) / f"log_{prefix}.txt"
        last_save_n = {"n": -1}
        last_snap_n = {"n": -1}
        snap_counter = {"k": 0}
        first_print = {"done": False}

        def default_monitor(ts_, step, t, u, hef_, mesh_):
            dt_ts = ts_.getTimeStep()
            _sync_vec_to_function(u, hef_.m)
            u.copy(hef_.m_gpu)

            mag_local = hef_.m.x.array[: hef_.local_size].reshape((-1, 3))
            mag = mesh_.comm.gather(mag_local, root=0)
            energy = hef_.compute_Energy_terms()

            # GPU spatial-resolution diagnostic, evaluated only at log time.
            max_neighbor_angle_deg = hef_.max_neighbor_angle_deg_gpu(u)

            n_snap = int(np.trunc(t / dt_snap))
            if n_snap != last_snap_n["n"]:
                last_snap_n["n"] = n_snap
                filename = Path(output_dir) / f"m_sot_{prefix}_{snap_counter['k']:03d}.xdmf"
                snap_counter["k"] += 1

                try:
                    mesh_.name = "Grid"
                except Exception:
                    pass

                try:
                    hef_.m.name = "f"
                except Exception:
                    pass

                with io.XDMFFile(mesh_.comm, str(filename), "w") as xdmf:
                    xdmf.write_mesh(mesh_)
                    xdmf.write_function(hef_.m)

            if mesh_.comm.rank == 0:
                mag = np.reshape(np.concatenate(mag), (-1, 3))
                maxdmdt = (
                    float(self.stopper.last_max_dmdt_deg_ns)
                    if self.stopper is not None
                    else 0.0
                )
                if not np.isfinite(maxdmdt):
                    maxdmdt = 0.0

                if not first_print["done"]:
                    header = (
                        f"{'time':>10} {'dt':>10} "
                        f"{'<mx>':>15} {'<my>':>15} {'<mz>':>15} "
                        f"{'maxdmdt(deg/ns)':>18} "
                        f"{'max_nn_angle(deg)':>18} "
                        f"{'E_demag':>15} {'E_exch':>15} {'E_ani':>15} "
                        f"{'E_dmi_bulk':>15} {'E_dmi_int':>15} "
                        f"{'E_ext':>15} {'E_total':>15}"
                    )
                    print(header)
                    with open(log_path, "w") as f:
                        f.write(header + "\n")
                    first_print["done"] = True

                line = (
                    f"{t*1e9:10.4f} {dt_ts*1e9:10.4f} "
                    f"{mag[:,0].mean():15.6f} {mag[:,1].mean():15.6f} {mag[:,2].mean():15.6f} "
                    f"{maxdmdt:18.6e} "
                    f"{max_neighbor_angle_deg:18.6f} "
                    f"{energy['E_demag']:15.4e} {energy['E_exch']:15.4e} "
                    f"{energy['E_ani']:15.4e} {energy['E_dmi_bulk']:15.4e} "
                    f"{energy['E_dmi_int']:15.4e} {energy['E_ext']:15.4e} "
                    f"{energy['E_total']:15.4e}"
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

        self.ts.setMonitor(monitor)

    def solve_bdf(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        dt_save=None,
        dt_snap=None,
        output_dir="output_sot_gpu_bdf",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        snes_rtol=1e-2,
        snes_atol=1e-5,
        ksp_rtol=1e-4,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
        pc_python=True,
        pc_use_abs_diag=False,
        pc_side="right",
        pc_norm_type="unpreconditioned",
    ):
        self._ensure_solver_bdf(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol,
            ts_atol=ts_atol,
            snes_rtol=snes_rtol,
            snes_atol=snes_atol,
            ksp_rtol=ksp_rtol,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
            pc_python=pc_python,
            pc_use_abs_diag=pc_use_abs_diag,
            pc_side=pc_side,
            pc_norm_type=pc_norm_type,
        )

        ts = self.ts
        hef = self.hef
        comm = self.mesh.comm
        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.restartStep()

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)
            hef.m_gpu.copy(self.y)
        if self.stopper is not None:
            self.stopper.reset(self.y)

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()
        self._cancel_ts_monitors()
        self._install_monitor(
            dt_save=dt_save,
            dt_snap=dt_snap,
            output_dir=output_dir,
            monitor_fn=monitor_fn,
            prefix="gpu_bdf",
        )

        tstart = perf_counter()
        ts.solve(self.y)
        elapsed = perf_counter() - tstart

        self.y.copy(hef.m_gpu)
        self._prepare_function_for_io()
        filename = Path(output_dir) / "Relax_SOT_GPU_BDF.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "step_rejections": int(ts.getStepRejections()),
            "snes_failures": int(ts.getSNESFailures()),
            "snes_iterations": int(ts.getSNESIterations()),
            "ksp_iterations": int(ts.getKSPIterations()),
            "maxdmdt_deg_ns": (
                float(self.stopper.last_max_dmdt_deg_ns)
                if self.stopper is not None else float("nan")
            ),
            "jv_calls": getattr(self.ctx, "calls", None),
            "pc_calls": getattr(self.ctx, "callsPre", None),
            "pc_enabled": bool(pc_python),
            "pc_use_abs_diag": (
                bool(pc_use_abs_diag) if pc_python else None
            ),
            "pc_side": (
                self._pc_bdf_config["side"]
                if pc_python else "none"
            ),
            "ksp_norm_type": self._pc_bdf_config["norm_type"],
        }

        if save_final_state and _HAVE_ADIOS:
            fname = Path(output_dir) / "Relax_SOT_GPU_BDF.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if comm.rank == 0:
            print("SOT BDF finished")
            print("nsteps:", stats["nsteps"])
            print("t_end:", stats["t_end"])
            print("dt_last:", stats["dt_last"])
            print("reason:", stats["reason"])
            print("step rejections:", stats["step_rejections"])
            print("SNES failures:", stats["snes_failures"])
            print("SNES iterations:", stats["snes_iterations"])
            print("KSP iterations:", stats["ksp_iterations"])
            print("Jv calls:", stats["jv_calls"])
            print("PC calls:", stats["pc_calls"])
            print("PC side:", stats["pc_side"])
            print("KSP norm:", stats["ksp_norm_type"])
            print("PC abs(diag):", stats["pc_use_abs_diag"])
            print("wall-clock:", elapsed)
            if save_final_state and not _HAVE_ADIOS:
                print("[SOT-GPU] adios4dolfinx unavailable: BP checkpoint skipped.")

        if return_stats:
            return self.y, self.ctx, elapsed, stats
        return self.y, self.ctx, elapsed

    def _ensure_solver_explicit(
        self,
        m0_array,
        dt_init,
        ts_rtol=1e-6,
        ts_atol=1e-6,
        rk_type="5dp",
        stopping_dmdt=0.0,
        check_every_stop=10,
        stop_print=False,
    ):
        if self.hef is None:
            self._build_effective_field()
        hef = self.hef
        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)

        ts = PETSc.TS().create(self.mesh.comm)
        ts.setType("rk")
        ts.setRKType(rk_type)
        ts.setTime(0.0)
        ts.setTimeStep(float(dt_init))
        ts.setTolerances(rtol=ts_rtol, atol=ts_atol)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)
        ts.setRHSFunction(hef.rhs_function, hef.dmdt_gpu)

        opts = PETSc.Options()
        opts["ts_adapt_type"] = "basic"
        opts["ts_adapt_clip"] = "0.1, 3.0"
        opts["ts_adapt_safety"] = 0.9
        opts["ts_adapt_reject_safety"] = 0.1
        opts["ts_adapt_dt_min"] = 1e-18
        opts["ts_adapt_dt_max"] = 1e-11
        opts["ts_max_steps"] = 50_000_000
        ts.setFromOptions()

        y = _dup_cuda_vec(hef.m_gpu, block_size=3)
        hef.m_gpu.copy(y)
        ts.setSolution(y)
        stopper = StopByMaxDmdtFDGPU(
            self.mesh.comm,
            stopping_dm_dt_deg_ns=stopping_dmdt,
            vec_template=y,
            n_owned_scalar=hef.local_size,
            check_every=check_every_stop,
            print_every_hit=stop_print,
        )
        ts.setPostStep(stopper)

        self.ts = ts
        self.y = y
        self.stopper = stopper
        self.ctx = None
        self.J = None
        self._solver_ready = True

    def solve_explicit(
        self,
        m0_array,
        t0,
        t_final,
        dt_init,
        rk_type="5dp",
        dt_save=None,
        dt_snap=None,
        output_dir="output_sot_gpu",
        ts_rtol=1e-6,
        ts_atol=1e-6,
        stopping_dmdt=0.0,
        monitor_fn=None,
        save_final_state=True,
        check_every_stop=5,
        stop_print=False,
        return_stats=False,
    ):
        self._ensure_solver_explicit(
            m0_array=m0_array,
            dt_init=dt_init,
            ts_rtol=ts_rtol,
            ts_atol=ts_atol,
            rk_type=rk_type,
            stopping_dmdt=stopping_dmdt,
            check_every_stop=check_every_stop,
            stop_print=stop_print,
        )

        ts = self.ts
        hef = self.hef
        comm = self.mesh.comm
        ts.setTime(float(t0))
        ts.setMaxTime(float(t_final))
        ts.setTimeStep(float(dt_init))
        ts.restartStep()

        if m0_array is not None:
            hef.set_m_from_cpu(m0_array)
            hef.m_gpu.copy(self.y)
        if self.stopper is not None:
            self.stopper.reset(self.y)

        if comm.rank == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        comm.barrier()
        self._cancel_ts_monitors()
        self._install_monitor(
            dt_save=dt_save,
            dt_snap=dt_snap,
            output_dir=output_dir,
            monitor_fn=monitor_fn,
            prefix="gpu",
        )

        tstart = perf_counter()
        ts.solve(self.y)
        elapsed = perf_counter() - tstart

        self.y.copy(hef.m_gpu)
        self._prepare_function_for_io()

        stats = {
            "t_end": float(ts.getTime()),
            "dt_last": float(ts.getTimeStep()),
            "nsteps": int(ts.getStepNumber()),
            "reason": int(ts.getConvergedReason()),
            "maxdmdt_deg_ns": (
                float(self.stopper.last_max_dmdt_deg_ns)
                if self.stopper is not None else float("nan")
            ),
        }

        filename = Path(output_dir) / "Relax_SOT_GPU.xdmf"
        with io.XDMFFile(self.mesh.comm, str(filename), "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(hef.m)

        if save_final_state and _HAVE_ADIOS:
            fname = Path(output_dir) / "Relax_SOT_GPU.bp"
            ad.write_mesh(fname, self.mesh)
            ad.write_function(fname, hef.m, time=0.0, name="m")

        if comm.rank == 0:
            print("SOT explicit RK finished")
            print("RK type:", rk_type)
            print("nsteps:", stats["nsteps"])
            print("t_end:", stats["t_end"])
            print("dt_last:", stats["dt_last"])
            print("reason:", stats["reason"])
            print("wall-clock:", elapsed)
            if save_final_state and not _HAVE_ADIOS:
                print("[SOT-GPU] adios4dolfinx unavailable: BP checkpoint skipped.")

        if return_stats:
            return self.y, elapsed, stats
        return self.y, elapsed

    def solve(self, *args, **kwargs):
        """Default path: implicit BDF, matching the GPU STT module."""
        return self.solve_bdf(*args, **kwargs)
