import numpy as np
import cupy as cp
import ufl

from dolfinx import fem
from petsc4py import PETSc


def _vec_to_cupy(vec: PETSc.Vec, mode="rw"):
    return cp.from_dlpack(vec.toDLPack(mode))


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


class CubicAnisotropyFieldGPU:
    """
    GPU cubic anisotropy field.

        H_cubic = + 2 K1 / (mu0 Ms) *
                  [ s1 u1 + s2 u2 + s3 u3 ]

    with:

        a1 = m cdot u1
        a2 = m cdotu2
        a3 = m cdot u3

        s1 = a1 * (a2^2 + a3^2)
        s2 = a2 * (a3^2 + a1^2)
        s3 = a3 * (a1^2 + a2^2)

    and the associated energy convention:

        E_cubic = -K1 int (a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2) dV

    Mesh coordinates are assumed to be in nm, so nodal volume is nm^3
    and energy is multiplied by 1e-27.
    """

    def __init__(self, mesh, V, K1, Ms, u1, u2, VolN):
        self.mesh = mesh
        self.V = V

        self.K1 = float(K1)
        self.M_s = float(Ms)
        self.mu_0 = 4.0 * np.pi * 1e-7
        self.vol_scale = 1e-27

        self.start, self.end = V.dofmap.index_map.local_range
        self.local_dofs = self.end - self.start
        self.local_size = 3 * self.local_dofs

        # ------------------------------------------------------------
        # Prepare axes on host in nodal/vector layout.
        # ------------------------------------------------------------
        u1_np = self._prepare_axis(u1, "u1")
        u2_np = self._prepare_axis(u2, "u2")
        u3_np = np.empty_like(u1_np)

        self._orthonormalize_axes(u1_np, u2_np, u3_np)

        # Host-side functions for diagnostic exact-style UFL energy.
        self.u1 = fem.Function(V, name="u1_cubic")
        self.u2 = fem.Function(V, name="u2_cubic")
        self.u3 = fem.Function(V, name="u3_cubic")

        self.u1.x.array[:] = u1_np.reshape(-1)
        self.u2.x.array[:] = u2_np.reshape(-1)
        self.u3.x.array[:] = u3_np.reshape(-1)

        self.u1.x.scatter_forward()
        self.u2.x.scatter_forward()
        self.u3.x.scatter_forward()

        # GPU copies.
        self.u1_gpu = cp.asarray(u1_np, dtype=cp.float64)
        self.u2_gpu = cp.asarray(u2_np, dtype=cp.float64)
        self.u3_gpu = cp.asarray(u3_np, dtype=cp.float64)

        VolN = np.asarray(VolN, dtype=np.float64)
        self.vol_gpu = cp.asarray(
            VolN[: self.local_size].reshape((-1, 3))[:, 0],
            dtype=cp.float64,
        )

        self.pref = (
            2.0 * self.K1 / (self.mu_0 * self.M_s)
            if self.M_s != 0.0
            else 0.0
        )

        self.H = fem.Function(V, name="H_cubic")

        self.h_gpu = self.H.x.petsc_vec.duplicate()
        _set_vec_cuda(self.h_gpu, block_size=3)

        # Reusable GPU buffers.
        self.a1 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.a2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.a3 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.s1 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.s2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.s3 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.tmp = cp.empty(self.local_dofs, dtype=cp.float64)

        # Reusable GPU buffers for Jacobian-vector products.
        self.da1 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.da2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.da3 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.a1_2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.a2_2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.a3_2 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.ds1 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.ds2 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.ds3 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.A11 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.A22 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.A33 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.A12 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.A13 = cp.empty(self.local_dofs, dtype=cp.float64)
        self.A23 = cp.empty(self.local_dofs, dtype=cp.float64)

        self.tmp2 = cp.empty(self.local_dofs, dtype=cp.float64)



        dp12 = np.einsum("ij,ij->i", u1_np, u2_np)
        print(f"[Cubic GPU] max|u1 . u2| = {np.max(np.abs(dp12)):.3e}", flush=True)


    def _dot3_to(self, A, B, out):
        """
        Compute out[:] = sum_j A[:, j] * B[:, j] without allocating
        an intermediate (A * B) array.
        """

        cp.multiply(A[:, 0], B[:, 0], out=out)

        cp.multiply(A[:, 1], B[:, 1], out=self.tmp)
        out += self.tmp

        cp.multiply(A[:, 2], B[:, 2], out=self.tmp)
        out += self.tmp

    def _prepare_axis(self, axis, name):
        arr = np.asarray(axis, dtype=np.float64)

        if arr.ndim == 1 and arr.size == 3:
            out = np.empty((self.local_dofs, 3), dtype=np.float64)
            out[:, :] = arr[None, :]
            return out

        if arr.ndim == 1 and arr.size == self.local_size:
            return arr.reshape((-1, 3)).copy()

        if arr.ndim == 2 and arr.shape == (self.local_dofs, 3):
            return arr.copy()

        raise ValueError(
            f"{name} must have shape (3,), ({self.local_size},), "
            f"or ({self.local_dofs}, 3). Got {arr.shape}."
        )

    @staticmethod
    def _normalize_rows(A, eps=1e-30):
        n = np.linalg.norm(A, axis=1)
        n = np.where(n > eps, n, 1.0)
        A /= n[:, None]

    def _orthonormalize_axes(self, u1, u2, u3):
        self._normalize_rows(u1)

        # Gram-Schmidt: u2 <- u2 - (u2 · u1) u1
        dp = np.einsum("ij,ij->i", u2, u1)
        u2[:] = u2 - dp[:, None] * u1
        self._normalize_rows(u2)

        # u3 = u1 x u2
        u3[:] = np.cross(u1, u2)
        self._normalize_rows(u3)

    def compute_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):
        """
        Compute H_cubic directly on GPU.

        Parameters
        ----------
        m_vec:
            PETSc CUDA vector with magnetization.
        out_vec:
            PETSc CUDA vector where the cubic field is written.
        """

        m_all = _vec_to_cupy(m_vec, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        H = out_all[: self.local_size].reshape((-1, 3))

        u1 = self.u1_gpu
        u2 = self.u2_gpu
        u3 = self.u3_gpu

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3
        tmp = self.tmp

        # a_i = m · u_i
        a1[:] = cp.sum(m * u1, axis=1)
        a2[:] = cp.sum(m * u2, axis=1)
        a3[:] = cp.sum(m * u3, axis=1)

        # s1 = a1 * (a2^2 + a3^2)
        cp.multiply(a2, a2, out=s1)
        cp.multiply(a3, a3, out=tmp)
        s1 += tmp
        s1 *= a1

        # s2 = a2 * (a3^2 + a1^2)
        cp.multiply(a3, a3, out=s2)
        cp.multiply(a1, a1, out=tmp)
        s2 += tmp
        s2 *= a2

        # s3 = a3 * (a1^2 + a2^2)
        cp.multiply(a1, a1, out=s3)
        cp.multiply(a2, a2, out=tmp)
        s3 += tmp
        s3 *= a3

        pref = self.pref

        H[:, 0] = pref * (
            s1 * u1[:, 0]
            + s2 * u2[:, 0]
            + s3 * u3[:, 0]
        )
        H[:, 1] = pref * (
            s1 * u1[:, 1]
            + s2 * u2[:, 1]
            + s3 * u3[:, 1]
        )
        H[:, 2] = pref * (
            s1 * u1[:, 2]
            + s2 * u2[:, 2]
            + s3 * u3[:, 2]
        )

        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0

        return out_vec

    def compute(self, m_gpu: PETSc.Vec):

        self.compute_vec(m_gpu, self.h_gpu)
        self.h_gpu.copy(self.H.x.petsc_vec)
        self.H.x.scatter_forward()
        return self.H

    def jac_times_vec(self, m_vec: PETSc.Vec, v_vec: PETSc.Vec, out_vec: PETSc.Vec):

        m_all = _vec_to_cupy(m_vec, "r")
        v_all = _vec_to_cupy(v_vec, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        v = v_all[: self.local_size].reshape((-1, 3))
        out = out_all[: self.local_size].reshape((-1, 3))

        u1 = self.u1_gpu
        u2 = self.u2_gpu
        u3 = self.u3_gpu

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3

        da1 = self.da1
        da2 = self.da2
        da3 = self.da3

        a1_2 = self.a1_2
        a2_2 = self.a2_2
        a3_2 = self.a3_2

        ds1 = self.ds1
        ds2 = self.ds2
        ds3 = self.ds3

        tmp = self.tmp
        tmp2 = self.tmp2

        self._dot3_to(m, u1, a1)
        self._dot3_to(m, u2, a2)
        self._dot3_to(m, u3, a3)

        self._dot3_to(v, u1, da1)
        self._dot3_to(v, u2, da2)
        self._dot3_to(v, u3, da3)

        cp.multiply(a1, a1, out=a1_2)
        cp.multiply(a2, a2, out=a2_2)
        cp.multiply(a3, a3, out=a3_2)

        cp.add(a2_2, a3_2, out=ds1)
        ds1 *= da1

        cp.multiply(a2, da2, out=tmp)
        tmp *= 2.0

        cp.multiply(a3, da3, out=tmp2)
        tmp2 *= 2.0

        tmp += tmp2
        tmp *= a1
        ds1 += tmp

        cp.add(a3_2, a1_2, out=ds2)
        ds2 *= da2

        cp.multiply(a3, da3, out=tmp)
        tmp *= 2.0

        cp.multiply(a1, da1, out=tmp2)
        tmp2 *= 2.0

        tmp += tmp2
        tmp *= a2
        ds2 += tmp

        cp.add(a1_2, a2_2, out=ds3)
        ds3 *= da3

        cp.multiply(a1, da1, out=tmp)
        tmp *= 2.0

        cp.multiply(a2, da2, out=tmp2)
        tmp2 *= 2.0

        tmp += tmp2
        tmp *= a3
        ds3 += tmp

        pref = self.pref

        cp.multiply(ds1, u1[:, 0], out=out[:, 0])
        cp.multiply(ds2, u2[:, 0], out=tmp)
        out[:, 0] += tmp
        cp.multiply(ds3, u3[:, 0], out=tmp)
        out[:, 0] += tmp
        out[:, 0] *= pref

        cp.multiply(ds1, u1[:, 1], out=out[:, 1])
        cp.multiply(ds2, u2[:, 1], out=tmp)
        out[:, 1] += tmp
        cp.multiply(ds3, u3[:, 1], out=tmp)
        out[:, 1] += tmp
        out[:, 1] *= pref

        cp.multiply(ds1, u1[:, 2], out=out[:, 2])
        cp.multiply(ds2, u2[:, 2], out=tmp)
        out[:, 2] += tmp
        cp.multiply(ds3, u3[:, 2], out=tmp)
        out[:, 2] += tmp
        out[:, 2] *= pref

        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0

        return out_vec

    def jac_diag_vec(self, m_vec: PETSc.Vec, out_vec: PETSc.Vec):

        m_all = _vec_to_cupy(m_vec, "r")
        out_all = _vec_to_cupy(out_vec, "rw")

        m = m_all[: self.local_size].reshape((-1, 3))
        out = out_all[: self.local_size].reshape((-1, 3))

        u1 = self.u1_gpu
        u2 = self.u2_gpu
        u3 = self.u3_gpu

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3

        a1_2 = self.a1_2
        a2_2 = self.a2_2
        a3_2 = self.a3_2

        A11 = self.A11
        A22 = self.A22
        A33 = self.A33
        A12 = self.A12
        A13 = self.A13
        A23 = self.A23

        tmp = self.tmp
        tmp2 = self.tmp2

        self._dot3_to(m, u1, a1)
        self._dot3_to(m, u2, a2)
        self._dot3_to(m, u3, a3)

        cp.multiply(a1, a1, out=a1_2)
        cp.multiply(a2, a2, out=a2_2)
        cp.multiply(a3, a3, out=a3_2)

        cp.add(a2_2, a3_2, out=A11)
        cp.add(a3_2, a1_2, out=A22)
        cp.add(a1_2, a2_2, out=A33)

        cp.multiply(a1, a2, out=A12)
        A12 *= 2.0

        cp.multiply(a1, a3, out=A13)
        A13 *= 2.0

        cp.multiply(a2, a3, out=A23)
        A23 *= 2.0

        pref = self.pref

        for c in range(3):
            u1c = u1[:, c]
            u2c = u2[:, c]
            u3c = u3[:, c]

            col = out[:, c]

            # A11 * u1c^2
            cp.multiply(u1c, u1c, out=tmp)
            cp.multiply(A11, tmp, out=col)

            # A22 * u2c^2
            cp.multiply(u2c, u2c, out=tmp)
            cp.multiply(A22, tmp, out=tmp2)
            col += tmp2

            # A33 * u3c^2
            cp.multiply(u3c, u3c, out=tmp)
            cp.multiply(A33, tmp, out=tmp2)
            col += tmp2

            # 2*A12*u1c*u2c
            cp.multiply(u1c, u2c, out=tmp)
            cp.multiply(A12, tmp, out=tmp2)
            tmp2 *= 2.0
            col += tmp2

            # 2*A13*u1c*u3c
            cp.multiply(u1c, u3c, out=tmp)
            cp.multiply(A13, tmp, out=tmp2)
            tmp2 *= 2.0
            col += tmp2

            # 2*A23*u2c*u3c
            cp.multiply(u2c, u3c, out=tmp)
            cp.multiply(A23, tmp, out=tmp2)
            tmp2 *= 2.0
            col += tmp2

            col *= pref

        if out_all.size > self.local_size:
            out_all[self.local_size:] = 0.0

        return out_vec



    def Energy_lumped_gpu(self, m_vec: PETSc.Vec):
        """
        Fast GPU energy for the minimizer.

            E = -K1 int S dV

        where:

            S = a1^2 a2^2 + a2^2 a3^2 + a3^2 a1^2
        """

        m_all = _vec_to_cupy(m_vec, "r")
        m = m_all[: self.local_size].reshape((-1, 3))

        u1 = self.u1_gpu
        u2 = self.u2_gpu
        u3 = self.u3_gpu

        a1 = cp.sum(m * u1, axis=1)
        a2 = cp.sum(m * u2, axis=1)
        a3 = cp.sum(m * u3, axis=1)

        a1_2 = a1 * a1
        a2_2 = a2 * a2
        a3_2 = a3 * a3

        S = a1_2 * a2_2 + a2_2 * a3_2 + a3_2 * a1_2

        val = cp.sum(self.vol_gpu * S)

        return float((-self.K1 * val * self.vol_scale).item())

    def Energy(self, m):

        mA = m.x.array.reshape((-1, 3))
        HA = self.H.x.array.reshape((-1, 3))

        u1A = self.u1.x.array.reshape((-1, 3))
        u2A = self.u2.x.array.reshape((-1, 3))
        u3A = self.u3.x.array.reshape((-1, 3))

        a1 = np.einsum("ij,ij->i", mA, u1A)
        a2 = np.einsum("ij,ij->i", mA, u2A)
        a3 = np.einsum("ij,ij->i", mA, u3A)

        s1 = a1 * (a2 * a2 + a3 * a3)
        s2 = a2 * (a3 * a3 + a1 * a1)
        s3 = a3 * (a1 * a1 + a2 * a2)

        pref = self.pref

        HA[:, :] = (
            (pref * s1)[:, None] * u1A
            + (pref * s2)[:, None] * u2A
            + (pref * s3)[:, None] * u3A
        )

        self.H.x.scatter_forward()

        dE = -ufl.dot(m, self.H) * ufl.dx(domain=self.mesh)
        Eloc = fem.assemble_scalar(fem.form(dE))

        return 0.25 * self.mu_0 * self.M_s * float(Eloc) * self.vol_scale
