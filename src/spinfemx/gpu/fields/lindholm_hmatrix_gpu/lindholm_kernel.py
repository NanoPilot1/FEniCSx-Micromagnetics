from __future__ import annotations

import math
from numba import njit


@njit(fastmath=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath=True)
def _norm(a):
    return math.sqrt(_dot(a, a))


@njit(fastmath=True)
def _det3(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@njit(fastmath=True)
def solid_angle_with_atan2(x0, p0, p1, p2):
    tiny = 1e-300

    r0 = p0 - x0
    r1 = p1 - x0
    r2 = p2 - x0

    n0 = _norm(r0) + tiny
    n1 = _norm(r1) + tiny
    n2 = _norm(r2) + tiny

    det = _det3(r0, r1, r2)
    d01 = _dot(r0, r1)
    d12 = _dot(r1, r2)
    d20 = _dot(r2, r0)

    denom = n0 * n1 * n2 + d01 * n2 + d12 * n0 + d20 * n1
    return 2.0 * math.atan2(det, denom)


@njit(fastmath=True)
def _P_log1p(ri, rj, s):
    tiny = 1e-300
    A = ri + rj
    denom = A - s
    if denom < tiny:
        denom = tiny
    return math.log1p((2.0 * s) / denom)


@njit(fastmath=True)
def lindholm_weights_precomp(
    x0,
    p0,
    p1,
    p2,
    n_unit,
    area,
    s_edge,
    eta_vecs,
    gamma_mat,
):
    tiny = 1e-300

    rho0 = p0 - x0
    rho1 = p1 - x0
    rho2 = p2 - x0

    r0 = _norm(rho0) + tiny
    r1 = _norm(rho1) + tiny
    r2 = _norm(rho2) + tiny

    h = _dot(n_unit, rho0)

    eta0 = _dot(eta_vecs[0], rho0)
    eta1 = _dot(eta_vecs[1], rho1)
    eta2 = _dot(eta_vecs[2], rho2)

    s0 = s_edge[0]
    s1 = s_edge[1]
    s2 = s_edge[2]

    P0 = _P_log1p(r0, r1, s0)
    P1 = _P_log1p(r1, r2, s1)
    P2 = _P_log1p(r2, r0, s2)

    Omega = solid_angle_with_atan2(x0, p0, p1, p2)

    gP0 = gamma_mat[0, 0] * P0 + gamma_mat[0, 1] * P1 + gamma_mat[0, 2] * P2
    gP1 = gamma_mat[1, 0] * P0 + gamma_mat[1, 1] * P1 + gamma_mat[1, 2] * P2
    gP2 = gamma_mat[2, 0] * P0 + gamma_mat[2, 1] * P1 + gamma_mat[2, 2] * P2

    c = 1.0 / (8.0 * math.pi * (area + tiny))

    # Opposite-edge convention used by the current implementation.
    w0 = s1 * c * (eta1 * Omega - h * gP0)
    w1 = s2 * c * (eta2 * Omega - h * gP1)
    w2 = s0 * c * (eta0 * Omega - h * gP2)

    return w0, w1, w2
