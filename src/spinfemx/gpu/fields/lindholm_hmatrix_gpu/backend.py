from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .storage import HMatrixCPUData


def optional_import_cupy():
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
    except Exception as exc:
        raise RuntimeError(
            "CuPy is required for the GPU backend. Install the CuPy package "
            "matching the CUDA version available in your container."
        ) from exc
    return cp, cpsp


@dataclass
class LowRankBlockGPU:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    U: object
    V: object


class HMatrixGPU:
    """
    Device-resident Stage-A H-matrix MatVec.

    The Python loop over far blocks is kept as a simple validated reference; 
    the fused kernel in backend_fused.py is the production path
    """

    def __init__(self, data: HMatrixCPUData):
        cp, cpsp = optional_import_cupy()
        self.cp = cp
        self.cpsp = cpsp
        self.size = data.size

        self.perm = cp.asarray(data.perm, dtype=cp.int32)
        self.diag_jump_perm = cp.asarray(data.diag_jump[data.perm], dtype=cp.float64)

        near = data.near_csr
        self.near = cpsp.csr_matrix(
            (
                cp.asarray(near.data, dtype=cp.float64),
                cp.asarray(near.indices, dtype=cp.int32),
                cp.asarray(near.indptr, dtype=cp.int32),
            ),
            shape=near.shape,
        )

        self.far_blocks = [
            LowRankBlockGPU(
                row_start=block.row_start,
                row_stop=block.row_stop,
                col_start=block.col_start,
                col_stop=block.col_stop,
                U=cp.asarray(block.U, dtype=cp.float64),
                V=cp.asarray(block.V, dtype=cp.float64),
            )
            for block in data.far_blocks
        ]

        self._x_perm = cp.empty(self.size, dtype=cp.float64)
        self._y_perm = cp.empty(self.size, dtype=cp.float64)

    def matvec(self, x_gpu, out=None):
        cp = self.cp
        x_gpu = cp.asarray(x_gpu, dtype=cp.float64)

        if x_gpu.shape != (self.size,):
            raise ValueError(f"Expected x shape {(self.size,)}, got {x_gpu.shape}.")

        self._x_perm[:] = x_gpu[self.perm]
        self._y_perm[:] = self.diag_jump_perm * self._x_perm
        self._y_perm += self.near @ self._x_perm

        for block in self.far_blocks:
            xj = self._x_perm[block.col_start:block.col_stop]
            self._y_perm[block.row_start:block.row_stop] += block.U @ (block.V @ xj)

        if out is None:
            out = cp.empty_like(x_gpu)
        out[self.perm] = self._y_perm
        return out

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        cp = self.cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        y_gpu = self.matvec(x_gpu)
        return cp.asnumpy(y_gpu)

    def benchmark(self, x: np.ndarray, repeats: int = 20, warmup: int = 3):
        cp = self.cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        out = cp.empty_like(x_gpu)

        for _ in range(warmup):
            self.matvec(x_gpu, out=out)
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        stop = cp.cuda.Event()

        start.record()
        for _ in range(repeats):
            self.matvec(x_gpu, out=out)
        stop.record()
        stop.synchronize()

        elapsed = float(cp.cuda.get_elapsed_time(start, stop)) * 1e-3

        return {
            "timer": "cuda_events",
            "repeats": int(repeats),
            "total_seconds": float(elapsed),
            "average_seconds": float(elapsed / repeats),
        }
