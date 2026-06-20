import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix


class CubicAnisotropyField:
    def __init__(self, mesh, V, K1, Ms, u1, u2, length_scale_to_m=1.0):
        """
        u1, u2: arrays (3*N) in the order of depth of field (DOF) of V (vector P1)
        """

        self.mesh = mesh
        self.V = V
        self.M_s = float(Ms)
        self.K1 = float(K1)
        self.mu_0 = 4.0 * np.pi * 1e-7


        self.vol_scale = 1e-27

        self.u1 = fem.Function(V)
        self.u2 = fem.Function(V)
        self.u3 = fem.Function(V)



        self.u1.x.array[:] = np.asarray(u1, dtype=np.float64)
        self.u2.x.array[:] = np.asarray(u2, dtype=np.float64)


   

        # Views (with ghosts)
        self.u1A = self.u1.x.array.reshape(-1, 3)
        self.u2A = self.u2.x.array.reshape(-1, 3)
        self.u3A = self.u3.x.array.reshape(-1, 3)

        # --- orthonormalization per node ---
        def _normalize(A, eps=1e-30):
            n = np.linalg.norm(A, axis=1)
            n = np.where(n > eps, n, 1.0)
            A /= n[:, None]

        _normalize(self.u1A)
        # Gram-Schmidt: u2 <- u2 - (u2 . u1) u1

        dp = np.einsum("ij,ij->i", self.u2A, self.u1A)
        self.u2A[:] = self.u2A - dp[:, None] * self.u1A
        _normalize(self.u2A)

        # u3 = u1 x u2
        self.u3A[:] = np.cross(self.u1A, self.u2A)
        _normalize(self.u3A)

        # Orthogonality check (diagnostic)
        dp12 = np.einsum("ij,ij->i", self.u1A, self.u2A)
        if self.mesh.comm.rank == 0:
            print(f"[Cubic] max|u1 . u2| = {np.max(np.abs(dp12)):.3e}", flush=True)

        # Prefactor H (A/m)
        self.pref = -(2.0 * self.K1) / (self.mu_0 * self.M_s) if self.M_s != 0.0 else 0.0

        self.H = fem.Function(V)
        self.HA = self.H.x.array.reshape(-1, 3)

        # Buffers
        n = self.u1A.shape[0]
        self._a1 = np.empty(n, dtype=np.float64)
        self._a2 = np.empty(n, dtype=np.float64)
        self._a3 = np.empty(n, dtype=np.float64)
        self._s1 = np.empty(n, dtype=np.float64)
        self._s2 = np.empty(n, dtype=np.float64)
        self._s3 = np.empty(n, dtype=np.float64)
        self._tmp = np.empty(n, dtype=np.float64) 

    def jac_times_vec_owned(self, m_owned, v_owned, out_owned):
        """
        m_owned, v_owned, out_owned: (n_owned, 3) numpy arrays.
        out = (dHcubic/dm) v
        """
        n = m_owned.shape[0]
        u1 = self.u1A[:n]
        u2 = self.u2A[:n]
        u3 = self.u3A[:n]

        a1 = np.einsum("ij,ij->i", m_owned, u1)
        a2 = np.einsum("ij,ij->i", m_owned, u2)
        a3 = np.einsum("ij,ij->i", m_owned, u3)

        da1 = np.einsum("ij,ij->i", v_owned, u1)
        da2 = np.einsum("ij,ij->i", v_owned, u2)
        da3 = np.einsum("ij,ij->i", v_owned, u3)

        a1_2 = a1*a1; a2_2 = a2*a2; a3_2 = a3*a3

        ds1 = da1*(a2_2 + a3_2) + a1*(2.0*a2*da2 + 2.0*a3*da3)
        ds2 = da2*(a3_2 + a1_2) + a2*(2.0*a3*da3 + 2.0*a1*da1)
        ds3 = da3*(a1_2 + a2_2) + a3*(2.0*a1*da1 + 2.0*a2*da2)

        out_owned[:] = self.pref * (ds1[:,None]*u1 + ds2[:,None]*u2 + ds3[:,None]*u3)

    def compute(self, m):
        mA = m.x.array.reshape(-1, 3)

        a1 = self._a1; a2 = self._a2; a3 = self._a3
        s1 = self._s1; s2 = self._s2; s3 = self._s3
        tmp = self._tmp

        np.einsum("ij,ij->i", mA, self.u1A, out=a1)
        np.einsum("ij,ij->i", mA, self.u2A, out=a2)
        np.einsum("ij,ij->i", mA, self.u3A, out=a3)

        # s1 = a1*(a2^2 + a3^2)
        np.multiply(a2, a2, out=s1)      # s1 = a2^2
        np.multiply(a3, a3, out=tmp)     # tmp = a3^2
        s1 += tmp
        s1 *= a1

        # s2 = a2*(a3^2 + a1^2)
        np.multiply(a3, a3, out=s2)      # s2 = a3^2
        np.multiply(a1, a1, out=tmp)     # tmp = a1^2
        s2 += tmp
        s2 *= a2

        # s3 = a3*(a1^2 + a2^2)
        np.multiply(a1, a1, out=s3)      # s3 = a1^2
        np.multiply(a2, a2, out=tmp)     # tmp = a2^2   
        s3 += tmp
        s3 *= a3

        pref = self.pref
        HA = self.HA
        HA[:]  = (pref * s1)[:, None] * self.u1A
        HA[:] += (pref * s2)[:, None] * self.u2A
        HA[:] += (pref * s3)[:, None] * self.u3A

        self.H.x.scatter_forward()
        return self.H

    def Energy(self, m):

        self.compute(m)

        dE = -ufl.dot(m, self.H) * ufl.dx
        Eloc = fem.assemble_scalar(fem.form(dE))
        return 0.25 * self.mu_0 * self.M_s * float(Eloc) * self.vol_scale

