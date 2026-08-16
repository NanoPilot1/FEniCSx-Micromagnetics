from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class CompressionResult:
    """Low-rank factorization A ~= U @ V on either host or device arrays."""

    U: object
    V: object
    rank: int
    relative_residual: float
    method: str

    @property
    def storage_entries(self) -> int:
        return int(self.U.size + self.V.size)


def _validate_matrix_shape(A):
    if A.ndim != 2:
        raise ValueError("A must be a matrix.")
    return int(A.shape[0]), int(A.shape[1])


def _normalized_max_rank(max_rank: int | None, m: int, n: int) -> int:
    if max_rank is None:
        return min(m, n)
    return min(max(int(max_rank), 0), m, n)


def fullaca_dense(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompressionResult:
    """
    Simple full-pivot ACA reference implementation on the CPU.

    It materializes the dense residual.  This path remains available as a
    deterministic reference and as a fallback when GPU construction is not
    available.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)

    norm_A = float(np.linalg.norm(A, ord="fro"))
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=np.zeros((m, 0), dtype=np.float64),
            V=np.zeros((0, n), dtype=np.float64),
            rank=0,
            relative_residual=0.0,
            method="fullaca",
        )

    residual = A.copy()
    U_cols = []
    V_rows = []
    relative_residual = 1.0

    for _ in range(max_rank):
        pivot_flat = int(np.argmax(np.abs(residual)))
        i, j = np.unravel_index(pivot_flat, residual.shape)
        pivot = residual[i, j]

        if abs(pivot) <= np.finfo(np.float64).eps * norm_A:
            break

        u = residual[:, j].copy()
        v = residual[i, :].copy() / pivot

        U_cols.append(u)
        V_rows.append(v)

        residual -= np.outer(u, v)
        relative_residual = float(np.linalg.norm(residual, ord="fro") / norm_A)

        if relative_residual <= rtol:
            break

    if not U_cols:
        U = np.zeros((m, 0), dtype=np.float64)
        V = np.zeros((0, n), dtype=np.float64)
    else:
        U = np.ascontiguousarray(np.column_stack(U_cols), dtype=np.float64)
        V = np.ascontiguousarray(np.vstack(V_rows), dtype=np.float64)

    return CompressionResult(
        U=U,
        V=V,
        rank=int(U.shape[1]),
        relative_residual=float(relative_residual),
        method="fullaca",
    )


def truncated_svd(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompressionResult:
    """Reference truncated SVD compressor on the CPU."""
    A = np.asarray(A, dtype=np.float64)
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)

    norm_A = float(np.linalg.norm(A, ord="fro"))
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=np.zeros((m, 0), dtype=np.float64),
            V=np.zeros((0, n), dtype=np.float64),
            rank=0,
            relative_residual=0.0,
            method="svd",
        )

    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    squared_tail = np.cumsum((s[::-1] ** 2))[::-1]

    rank = min(max_rank, len(s))
    for candidate in range(0, min(max_rank, len(s)) + 1):
        residual_norm = (
            0.0 if candidate == len(s) else float(np.sqrt(squared_tail[candidate]))
        )
        if residual_norm / norm_A <= rtol:
            rank = candidate
            break

    if rank == 0:
        U_scaled = np.zeros((m, 0), dtype=np.float64)
        V = np.zeros((0, n), dtype=np.float64)
    else:
        U_scaled = np.ascontiguousarray(U[:, :rank] * s[:rank][None, :])
        V = np.ascontiguousarray(Vh[:rank, :])

    captured = float(np.sum(s[:rank] ** 2))
    residual_sq = max(norm_A * norm_A - captured, 0.0)
    rel = math.sqrt(residual_sq) / norm_A

    return CompressionResult(
        U=U_scaled,
        V=V,
        rank=int(rank),
        relative_residual=float(rel),
        method="svd",
    )


def randomized_svd(
    A: np.ndarray,
    rtol: float = 1e-6,
    max_rank: int | None = None,
    *,
    initial_rank: int = 16,
    oversampling: int = 8,
    power_iterations: int = 1,
    random_seed: int = 12345,
) -> CompressionResult:
    """
    Adaptive randomized SVD on the CPU.

    The exact Frobenius residual of the resulting projected approximation is
    obtained from ||A||_F^2 - sum(s_i^2), avoiding a dense residual matrix.
    This mirrors the GPU algorithm and provides a deterministic fallback.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)
    initial_rank = max(1, int(initial_rank))
    oversampling = max(0, int(oversampling))
    power_iterations = max(0, int(power_iterations))

    norm_sq = float(np.sum(A * A))
    norm_A = math.sqrt(norm_sq)
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=np.zeros((m, 0), dtype=np.float64),
            V=np.zeros((0, n), dtype=np.float64),
            rank=0,
            relative_residual=0.0,
            method="rsvd",
        )

    rng = np.random.default_rng(int(random_seed))
    target = min(initial_rank, max_rank)
    best = None

    while True:
        sketch_rank = min(max_rank, target + oversampling)
        omega = rng.standard_normal((n, sketch_rank))
        Y = A @ omega

        for _ in range(power_iterations):
            Q, _ = np.linalg.qr(Y, mode="reduced")
            Y = A @ (A.T @ Q)

        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ A
        Ub, s, Vh = np.linalg.svd(B, full_matrices=False)

        usable = min(max_rank, int(s.size))
        captured = np.cumsum(s[:usable] ** 2)
        rels = np.sqrt(np.maximum(norm_sq - captured, 0.0)) / norm_A
        acceptable = np.flatnonzero(rels <= rtol)

        rank = int(acceptable[0] + 1) if acceptable.size else usable
        rel = float(rels[rank - 1]) if rank else 1.0
        best = (Q, Ub, s, Vh, rank, rel)

        if acceptable.size or sketch_rank >= max_rank:
            break
        target = min(max_rank, max(target + 1, 2 * target))

    Q, Ub, s, Vh, rank, rel = best
    if rank == 0:
        U_scaled = np.zeros((m, 0), dtype=np.float64)
        V = np.zeros((0, n), dtype=np.float64)
    else:
        U_scaled = np.ascontiguousarray(
            (Q @ Ub[:, :rank]) * s[:rank][None, :], dtype=np.float64
        )
        V = np.ascontiguousarray(Vh[:rank, :], dtype=np.float64)

    return CompressionResult(
        U=U_scaled,
        V=V,
        rank=int(rank),
        relative_residual=float(rel),
        method="rsvd",
    )


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on the GPU container
        raise RuntimeError(
            "CuPy is required for GPU H-matrix compression."
        ) from exc
    return cp


_GPU_RANK1_UPDATE_KERNELS = {}


def _rank1_update_kernel(cp):
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _GPU_RANK1_UPDATE_KERNELS.get(device_id)
    if kernel is not None:
        return kernel

    source = r"""
    extern "C" __global__
    void rank1_residual_update(
        double* __restrict__ residual,
        const double* __restrict__ u,
        const double* __restrict__ v,
        const int m,
        const int n)
    {
        const long long total = ((long long) m) * n;
        const long long stride = ((long long) blockDim.x) * gridDim.x;
        for (long long idx = ((long long) blockDim.x) * blockIdx.x + threadIdx.x;
             idx < total;
             idx += stride) {
            const int i = (int) (idx / n);
            const int j = (int) (idx - ((long long) i) * n);
            residual[idx] -= u[i] * v[j];
        }
    }
    """
    kernel = cp.RawKernel(
        source,
        "rank1_residual_update",
        options=("--std=c++11",),
    )
    _GPU_RANK1_UPDATE_KERNELS[device_id] = kernel
    return kernel


def fullaca_dense_gpu(
    A,
    rtol: float = 1e-6,
    max_rank: int | None = None,
    *,
    residual_check_interval: int = 1,
) -> CompressionResult:
    """
    Full-pivot ACA executed on the active CUDA device.

    This intentionally follows the CPU reference algorithm.  The residual,
    pivot reductions, rank-one updates and Frobenius norms remain on the GPU;
    only small scalar convergence decisions are copied to the host.
    """
    cp = _import_cupy()
    A = cp.asarray(A, dtype=cp.float64, order="C")
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)
    residual_check_interval = max(1, int(residual_check_interval))

    norm_A = float(cp.linalg.norm(A).item())
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=cp.zeros((m, 0), dtype=cp.float64),
            V=cp.zeros((0, n), dtype=cp.float64),
            rank=0,
            relative_residual=0.0,
            method="gpu_fullaca",
        )

    residual = A.copy()
    U_cols = []
    V_rows = []
    relative_residual = 1.0
    update_kernel = _rank1_update_kernel(cp)
    threads = 256
    total = m * n
    blocks = min(65535, max(1, math.ceil(total / threads)))

    for rank_index in range(max_rank):
        pivot_flat = int(cp.argmax(cp.abs(residual)).item())
        i, j = divmod(pivot_flat, n)
        pivot = residual[i, j]
        pivot_abs = float(cp.abs(pivot).item())

        if pivot_abs <= np.finfo(np.float64).eps * norm_A:
            break

        u = residual[:, j].copy()
        v = residual[i, :].copy() / pivot
        U_cols.append(u)
        V_rows.append(v)

        update_kernel(
            (blocks,),
            (threads,),
            (residual, u, v, np.int32(m), np.int32(n)),
        )

        should_check = (
            (rank_index + 1) % residual_check_interval == 0
            or rank_index + 1 == max_rank
        )
        if should_check:
            relative_residual = float((cp.linalg.norm(residual) / norm_A).item())
            if relative_residual <= rtol:
                break

    if not U_cols:
        U = cp.zeros((m, 0), dtype=cp.float64)
        V = cp.zeros((0, n), dtype=cp.float64)
    else:
        U = cp.ascontiguousarray(cp.stack(U_cols, axis=1), dtype=cp.float64)
        V = cp.ascontiguousarray(cp.stack(V_rows, axis=0), dtype=cp.float64)

    # If convergence was not checked at the final accepted rank, report the
    # actual residual rather than the previous checkpoint.
    if U.shape[1] and (len(U_cols) % residual_check_interval != 0):
        relative_residual = float((cp.linalg.norm(residual) / norm_A).item())

    return CompressionResult(
        U=U,
        V=V,
        rank=int(U.shape[1]),
        relative_residual=float(relative_residual),
        method="gpu_fullaca",
    )


def truncated_svd_gpu(
    A,
    rtol: float = 1e-6,
    max_rank: int | None = None,
) -> CompressionResult:
    """Truncated SVD using CuPy/cuSOLVER."""
    cp = _import_cupy()
    A = cp.asarray(A, dtype=cp.float64, order="C")
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)

    norm_sq = float(cp.sum(A * A).item())
    norm_A = math.sqrt(norm_sq)
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=cp.zeros((m, 0), dtype=cp.float64),
            V=cp.zeros((0, n), dtype=cp.float64),
            rank=0,
            relative_residual=0.0,
            method="gpu_svd",
        )

    U, s, Vh = cp.linalg.svd(A, full_matrices=False)
    s_host = cp.asnumpy(s[:max_rank])
    captured = np.cumsum(s_host * s_host)
    rels = np.sqrt(np.maximum(norm_sq - captured, 0.0)) / norm_A
    acceptable = np.flatnonzero(rels <= rtol)
    rank = int(acceptable[0] + 1) if acceptable.size else int(max_rank)
    rel = float(rels[rank - 1]) if rank else 1.0

    U_scaled = cp.ascontiguousarray(
        U[:, :rank] * s[:rank][None, :], dtype=cp.float64
    )
    V = cp.ascontiguousarray(Vh[:rank, :], dtype=cp.float64)

    return CompressionResult(
        U=U_scaled,
        V=V,
        rank=rank,
        relative_residual=rel,
        method="gpu_svd",
    )


def randomized_svd_gpu(
    A,
    rtol: float = 1e-6,
    max_rank: int | None = None,
    *,
    initial_rank: int = 16,
    oversampling: int = 8,
    power_iterations: int = 1,
    random_seed: int = 12345,
) -> CompressionResult:
    """
    Adaptive randomized SVD using GPU GEMM, QR and a small projected SVD.

    This can be faster than full-pivot ACA for large, genuinely low-rank
    far-field blocks.  The method is deterministic for a
    fixed seed and computes the Frobenius residual from captured singular
    energy without materializing A - U@V.
    """
    cp = _import_cupy()
    A = cp.asarray(A, dtype=cp.float64, order="C")
    m, n = _validate_matrix_shape(A)
    max_rank = _normalized_max_rank(max_rank, m, n)
    initial_rank = max(1, int(initial_rank))
    oversampling = max(0, int(oversampling))
    power_iterations = max(0, int(power_iterations))

    norm_sq = float(cp.sum(A * A).item())
    norm_A = math.sqrt(norm_sq)
    if norm_A == 0.0 or max_rank == 0:
        return CompressionResult(
            U=cp.zeros((m, 0), dtype=cp.float64),
            V=cp.zeros((0, n), dtype=cp.float64),
            rank=0,
            relative_residual=0.0,
            method="gpu_rsvd",
        )

    rng = cp.random.RandomState(int(random_seed))
    target = min(initial_rank, max_rank)
    best = None

    while True:
        sketch_rank = min(max_rank, target + oversampling)
        omega = rng.standard_normal((n, sketch_rank)).astype(cp.float64, copy=False)
        Y = A @ omega

        for _ in range(power_iterations):
            Q, _ = cp.linalg.qr(Y, mode="reduced")
            Y = A @ (A.T @ Q)

        Q, _ = cp.linalg.qr(Y, mode="reduced")
        B = Q.T @ A
        Ub, s, Vh = cp.linalg.svd(B, full_matrices=False)

        usable = min(max_rank, int(s.size))
        s_host = cp.asnumpy(s[:usable])
        captured = np.cumsum(s_host * s_host)
        rels = np.sqrt(np.maximum(norm_sq - captured, 0.0)) / norm_A
        acceptable = np.flatnonzero(rels <= rtol)

        rank = int(acceptable[0] + 1) if acceptable.size else usable
        rel = float(rels[rank - 1]) if rank else 1.0
        best = (Q, Ub, s, Vh, rank, rel)

        if acceptable.size or sketch_rank >= max_rank:
            break
        target = min(max_rank, max(target + 1, 2 * target))

    Q, Ub, s, Vh, rank, rel = best
    if rank == 0:
        U_scaled = cp.zeros((m, 0), dtype=cp.float64)
        V = cp.zeros((0, n), dtype=cp.float64)
    else:
        U_scaled = cp.ascontiguousarray(
            (Q @ Ub[:, :rank]) * s[:rank][None, :], dtype=cp.float64
        )
        V = cp.ascontiguousarray(Vh[:rank, :], dtype=cp.float64)

    return CompressionResult(
        U=U_scaled,
        V=V,
        rank=int(rank),
        relative_residual=float(rel),
        method="gpu_rsvd",
    )


def compress_dense_block(
    A: np.ndarray,
    method: str,
    rtol: float,
    max_rank: int | None,
    *,
    random_seed: int = 12345,
    rsvd_initial_rank: int = 16,
    rsvd_oversampling: int = 8,
    rsvd_power_iterations: int = 1,
) -> CompressionResult:
    method_normalized = str(method).strip().lower()
    if method_normalized in {"fullaca", "aca"}:
        return fullaca_dense(A, rtol=rtol, max_rank=max_rank)
    if method_normalized == "svd":
        return truncated_svd(A, rtol=rtol, max_rank=max_rank)
    if method_normalized in {"rsvd", "randomized_svd", "randomized-svd"}:
        return randomized_svd(
            A,
            rtol=rtol,
            max_rank=max_rank,
            initial_rank=rsvd_initial_rank,
            oversampling=rsvd_oversampling,
            power_iterations=rsvd_power_iterations,
            random_seed=random_seed,
        )
    raise ValueError(
        f"Unsupported compressor: {method!r}. "
        "Use 'fullaca', 'svd', or 'rsvd'."
    )


def compress_dense_block_gpu(
    A,
    method: str,
    rtol: float,
    max_rank: int | None,
    *,
    random_seed: int = 12345,
    aca_residual_check_interval: int = 1,
    rsvd_initial_rank: int = 16,
    rsvd_oversampling: int = 8,
    rsvd_power_iterations: int = 1,
) -> CompressionResult:
    method_normalized = str(method).strip().lower()
    if method_normalized in {"fullaca", "aca", "gpu_fullaca"}:
        return fullaca_dense_gpu(
            A,
            rtol=rtol,
            max_rank=max_rank,
            residual_check_interval=aca_residual_check_interval,
        )
    if method_normalized in {"svd", "gpu_svd"}:
        return truncated_svd_gpu(A, rtol=rtol, max_rank=max_rank)
    if method_normalized in {
        "rsvd",
        "gpu_rsvd",
        "randomized_svd",
        "randomized-svd",
    }:
        return randomized_svd_gpu(
            A,
            rtol=rtol,
            max_rank=max_rank,
            initial_rank=rsvd_initial_rank,
            oversampling=rsvd_oversampling,
            power_iterations=rsvd_power_iterations,
            random_seed=random_seed,
        )
    raise ValueError(
        f"Unsupported GPU compressor: {method!r}. "
        "Use 'fullaca', 'svd', or 'rsvd'."
    )


def gpu_workspace_multiplier(method: str) -> float:
    """Conservative dense-block workspace multiplier used before allocation."""
    normalized = str(method).strip().lower()
    if normalized in {"fullaca", "aca", "gpu_fullaca"}:
        # A + dense residual + the temporary absolute-value array used by
        # full-pivot search, plus reduction/factor work buffers.
        return 3.5
    if normalized in {"svd", "gpu_svd"}:
        # cuSOLVER workspace is shape/device dependent; keep a conservative cap.
        return 4.0
    if normalized in {
        "rsvd",
        "gpu_rsvd",
        "randomized_svd",
        "randomized-svd",
    }:
        return 2.0
    return 3.0
