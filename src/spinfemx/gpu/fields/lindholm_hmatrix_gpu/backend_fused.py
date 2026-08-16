from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .backend import optional_import_cupy
from .storage import HMatrixCPUData


def _farfield_kernel_source(real_t: str) -> str:
    """
    Build the fused far-field kernel for a given factor precision.

    ``real_t`` is the C type of the packed low-rank factors U/V ("float" or
    "double").  The input/output vectors x, y and the shared reduction buffer
    stay in double precision regardless, so only the (dominant) factor storage
    and its memory traffic change with the precision choice.

    Layout: factors are stored *transposed* per block,
        U_flatT : (rank, nrows) row-major  -> U[i,k] at u0 + k*nrows + i
        V_flatT : (ncols, rank) row-major  -> V[k,j] at v0 + j*rank  + k
    so that consecutive threads (over k in the V stage, over i in the U stage)
    read consecutive addresses, i.e. coalesced global loads.
    """
    return r"""
extern "C" __device__ __forceinline__
double atomic_add_double_compat(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(
                value + __longlong_as_double(assumed)
            )
        );
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}


extern "C" __global__
void farfield_fused_matvec(
    const REAL_T* __restrict__ U_flatT,
    const REAL_T* __restrict__ V_flatT,
    const int* __restrict__ row_start,
    const int* __restrict__ row_stop,
    const int* __restrict__ col_start,
    const int* __restrict__ col_stop,
    const int* __restrict__ ranks,
    const long long* __restrict__ u_offset,
    const long long* __restrict__ v_offset,
    const double* __restrict__ x,
    double* __restrict__ y,
    const int nblocks
)
{
    const int block_id = blockIdx.x;
    if (block_id >= nblocks) {
        return;
    }

    const int tid = threadIdx.x;
    const int rs = row_start[block_id];
    const int re = row_stop[block_id];
    const int cs = col_start[block_id];
    const int ce = col_stop[block_id];
    const int rank = ranks[block_id];

    const int nrows = re - rs;
    const int ncols = ce - cs;

    const long long u0 = u_offset[block_id];
    const long long v0 = v_offset[block_id];

    extern __shared__ double z_shared[];

    // z[k] = sum_j V[k,j] * x[J_b][j]
    //
    // V is stored transposed as (ncols, rank): V[k,j] lives at v0 + j*rank + k.
    // At a fixed j, threads k, k+1, ... read consecutive addresses (coalesced).
    for (int k = tid; k < rank; k += blockDim.x) {
        double accum = 0.0;

        for (int j = 0; j < ncols; ++j) {
            accum += (double) V_flatT[v0 + ((long long) j) * rank + k] * x[cs + j];
        }

        z_shared[k] = accum;
    }

    __syncthreads();

    // y[I_b] += U_b @ z
    //
    // U is stored transposed as (rank, nrows): U[i,k] lives at u0 + k*nrows + i.
    // At a fixed k, threads i, i+1, ... read consecutive addresses (coalesced).
    //
    // Several admissible blocks can contribute to the same output row, so
    // atomic accumulation avoids races between CUDA thread blocks.
    for (int i = tid; i < nrows; i += blockDim.x) {
        double accum = 0.0;

        for (int k = 0; k < rank; ++k) {
            accum += (double) U_flatT[u0 + ((long long) k) * nrows + i] * z_shared[k];
        }

        atomic_add_double_compat(y + rs + i, accum);
    }
}
""".replace("REAL_T", real_t)




def _is_cupy_oom(cp, exc: Exception) -> bool:
    memory_module = getattr(getattr(cp, "cuda", None), "memory", None)
    oom_type = getattr(memory_module, "OutOfMemoryError", ())
    message = str(exc).lower()
    allocation_failure = any(
        token in message
        for token in (
            "out of memory",
            "memory allocation",
            "alloc failed",
            "status_alloc_failed",
        )
    )
    return isinstance(exc, oom_type) or allocation_failure

def _format_bytes(nbytes: int | float) -> str:
    value = float(nbytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024.0
    return f"{value:.3f} TiB"


@dataclass
class PackedFarFieldCPU:
    U_flat: np.ndarray
    V_flat: np.ndarray
    row_start: np.ndarray
    row_stop: np.ndarray
    col_start: np.ndarray
    col_stop: np.ndarray
    ranks: np.ndarray
    u_offset: np.ndarray
    v_offset: np.ndarray

    @property
    def nblocks(self) -> int:
        return int(self.ranks.size)

    @property
    def max_rank(self) -> int:
        return int(self.ranks.max()) if self.ranks.size else 0

    @property
    def accounted_bytes(self) -> int:
        arrays = (
            self.U_flat,
            self.V_flat,
            self.row_start,
            self.row_stop,
            self.col_start,
            self.col_stop,
            self.ranks,
            self.u_offset,
            self.v_offset,
        )
        return int(sum(arr.nbytes for arr in arrays))


def pack_far_blocks_cpu(
    data: HMatrixCPUData,
    dtype: np.dtype = np.float32,
) -> PackedFarFieldCPU:
    """
    Pack all low-rank factors into two flat host arrays plus compact metadata.

    Factors are stored *transposed* per block so the fused kernel reads them
    with coalesced global loads:
        U -> U^T of shape (rank, nrows), row-major
        V -> V^T of shape (ncols, rank), row-major

    ``dtype`` selects the stored factor precision (float32 by default, which
    halves both VRAM and the dominant memory traffic of the memory-bound
    far-field MatVec).  ACA is built in float64; only the stored copy is cast.

    This avoids thousands of independent CuPy allocations and allows the
    far-field contribution to be evaluated by a single RawKernel launch.
    """
    dtype = np.dtype(dtype)
    blocks = data.far_blocks
    nblocks = len(blocks)

    total_u = int(sum(block.U.size for block in blocks))
    total_v = int(sum(block.V.size for block in blocks))

    U_flat = np.empty(total_u, dtype=dtype)
    V_flat = np.empty(total_v, dtype=dtype)

    row_start = np.empty(nblocks, dtype=np.int32)
    row_stop = np.empty(nblocks, dtype=np.int32)
    col_start = np.empty(nblocks, dtype=np.int32)
    col_stop = np.empty(nblocks, dtype=np.int32)
    ranks = np.empty(nblocks, dtype=np.int32)
    u_offset = np.empty(nblocks, dtype=np.int64)
    v_offset = np.empty(nblocks, dtype=np.int64)

    upos = 0
    vpos = 0

    for block_id, block in enumerate(blocks):
        U = np.ascontiguousarray(block.U, dtype=np.float64)
        V = np.ascontiguousarray(block.V, dtype=np.float64)

        rank = int(U.shape[1])
        if V.shape != (rank, block.col_stop - block.col_start):
            raise ValueError(
                f"Inconsistent V shape for far block {block_id}: {V.shape}."
            )
        if U.shape[0] != block.row_stop - block.row_start:
            raise ValueError(
                f"Inconsistent U shape for far block {block_id}: {U.shape}."
            )

        usize = int(U.size)
        vsize = int(V.size)

        # Transposed, contiguous, cast to the storage precision.
        #   U^T : (rank, nrows) row-major  -> U[i,k] at k*nrows + i
        #   V^T : (ncols, rank) row-major  -> V[k,j] at j*rank  + k
        U_flat[upos:upos + usize] = np.ascontiguousarray(U.T, dtype=dtype).ravel(order="C")
        V_flat[vpos:vpos + vsize] = np.ascontiguousarray(V.T, dtype=dtype).ravel(order="C")

        row_start[block_id] = int(block.row_start)
        row_stop[block_id] = int(block.row_stop)
        col_start[block_id] = int(block.col_start)
        col_stop[block_id] = int(block.col_stop)
        ranks[block_id] = rank
        u_offset[block_id] = upos
        v_offset[block_id] = vpos

        upos += usize
        vpos += vsize

    return PackedFarFieldCPU(
        U_flat=U_flat,
        V_flat=V_flat,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        ranks=ranks,
        u_offset=u_offset,
        v_offset=v_offset,
    )


def pack_far_blocks_gpu(blocks, cp, dtype):
    """Pack transient GPU-built low-rank factors without a host round trip."""
    nblocks = len(blocks)
    total_u = int(sum(int(block[4].size) for block in blocks))
    total_v = int(sum(int(block[5].size) for block in blocks))

    U_flat = cp.empty(total_u, dtype=dtype)
    V_flat = cp.empty(total_v, dtype=dtype)

    row_start = np.empty(nblocks, dtype=np.int32)
    row_stop = np.empty(nblocks, dtype=np.int32)
    col_start = np.empty(nblocks, dtype=np.int32)
    col_stop = np.empty(nblocks, dtype=np.int32)
    ranks = np.empty(nblocks, dtype=np.int32)
    u_offset = np.empty(nblocks, dtype=np.int64)
    v_offset = np.empty(nblocks, dtype=np.int64)

    upos = 0
    vpos = 0
    for block_id, block in enumerate(blocks):
        rs, re, cs, ce, U, V = block
        rank = int(U.shape[1])
        if U.shape != (int(re) - int(rs), rank):
            raise ValueError(
                f"Inconsistent device U shape for far block {block_id}: {U.shape}."
            )
        if V.shape != (rank, int(ce) - int(cs)):
            raise ValueError(
                f"Inconsistent device V shape for far block {block_id}: {V.shape}."
            )

        usize = int(U.size)
        vsize = int(V.size)
        U_flat[upos:upos + usize] = cp.ascontiguousarray(
            U.T, dtype=dtype
        ).ravel(order="C")
        V_flat[vpos:vpos + vsize] = cp.ascontiguousarray(
            V.T, dtype=dtype
        ).ravel(order="C")

        row_start[block_id] = int(rs)
        row_stop[block_id] = int(re)
        col_start[block_id] = int(cs)
        col_stop[block_id] = int(ce)
        ranks[block_id] = rank
        u_offset[block_id] = upos
        v_offset[block_id] = vpos
        upos += usize
        vpos += vsize

    return {
        "U_flat": U_flat,
        "V_flat": V_flat,
        "row_start": cp.asarray(row_start, dtype=cp.int32),
        "row_stop": cp.asarray(row_stop, dtype=cp.int32),
        "col_start": cp.asarray(col_start, dtype=cp.int32),
        "col_stop": cp.asarray(col_stop, dtype=cp.int32),
        "ranks": cp.asarray(ranks, dtype=cp.int32),
        "u_offset": cp.asarray(u_offset, dtype=cp.int64),
        "v_offset": cp.asarray(v_offset, dtype=cp.int64),
        "nblocks": int(nblocks),
        "max_rank": int(ranks.max()) if ranks.size else 0,
    }


class HMatrixGPUPackedFused:
    """
    Packed GPU H-matrix backend with a fused far-field CUDA kernel.

    Near field:
        CSR matrix-vector product through cupyx.scipy.sparse.

    Far field:
        All low-rank blocks are stored in flat GPU arrays and applied using a
        single RawKernel launch. One CUDA thread block processes one low-rank
        H-matrix block.

    The output uses atomic accumulation because multiple H-matrix blocks can
    contribute to the same output interval.
    """

    backend_name = "packed_fused"

    def __init__(
        self,
        data: HMatrixCPUData,
        threads_per_block: int = 128,
        use_fp32: bool = True,
    ):
        cp, cpsp = optional_import_cupy()
        self.cp = cp
        self.cpsp = cpsp
        self.size = int(data.size)

        self.use_fp32 = bool(use_fp32)
        self._factor_dtype = np.float32 if self.use_fp32 else np.float64
        self._cp_factor_dtype = cp.float32 if self.use_fp32 else cp.float64
        self._real_c_type = "float" if self.use_fp32 else "double"

        threads_per_block = int(threads_per_block)
        if threads_per_block < 32 or threads_per_block > 1024:
            raise ValueError("threads_per_block must be between 32 and 1024.")
        self.threads_per_block = threads_per_block

        self.perm = cp.asarray(data.perm, dtype=cp.int32)
        self.diag_jump_perm = cp.asarray(
            data.diag_jump[data.perm],
            dtype=cp.float64,
        )

        near = data.near_csr
        self.near = cpsp.csr_matrix(
            (
                cp.asarray(near.data, dtype=cp.float64),
                cp.asarray(near.indices, dtype=cp.int32),
                cp.asarray(near.indptr, dtype=cp.int32),
            ),
            shape=near.shape,
        )

        device_blocks = getattr(data, "_gpu_far_blocks", None)
        can_direct_pack = (
            device_blocks is not None
            and len(device_blocks) == len(data.far_blocks)
        )
        packed_gpu = None
        if can_direct_pack:
            try:
                packed_gpu = pack_far_blocks_gpu(
                    device_blocks,
                    cp,
                    self._cp_factor_dtype,
                )
            except Exception as exc:
                if not _is_cupy_oom(cp, exc):
                    raise
                # The direct path temporarily needs both the per-block factors
                # and the final packed buffers.  If that peak does not fit,
                # release the transient device factors and rebuild the packed
                # arrays from the host cache copy instead.
                device_blocks.clear()
                cp.get_default_memory_pool().free_all_blocks()
                packed_gpu = None

        if packed_gpu is not None:
            self.n_far_blocks = packed_gpu["nblocks"]
            self.max_rank = packed_gpu["max_rank"]
            self.U_flat = packed_gpu["U_flat"]
            self.V_flat = packed_gpu["V_flat"]
            self.row_start = packed_gpu["row_start"]
            self.row_stop = packed_gpu["row_stop"]
            self.col_start = packed_gpu["col_start"]
            self.col_stop = packed_gpu["col_stop"]
            self.ranks = packed_gpu["ranks"]
            self.u_offset = packed_gpu["u_offset"]
            self.v_offset = packed_gpu["v_offset"]
            self.direct_device_pack = True
        else:
            if device_blocks is not None:
                device_blocks.clear()
                cp.get_default_memory_pool().free_all_blocks()

            packed = pack_far_blocks_cpu(data, dtype=self._factor_dtype)
            self.n_far_blocks = packed.nblocks
            self.max_rank = packed.max_rank

            self.U_flat = cp.asarray(packed.U_flat, dtype=self._cp_factor_dtype)
            self.V_flat = cp.asarray(packed.V_flat, dtype=self._cp_factor_dtype)

            self.row_start = cp.asarray(packed.row_start, dtype=cp.int32)
            self.row_stop = cp.asarray(packed.row_stop, dtype=cp.int32)
            self.col_start = cp.asarray(packed.col_start, dtype=cp.int32)
            self.col_stop = cp.asarray(packed.col_stop, dtype=cp.int32)
            self.ranks = cp.asarray(packed.ranks, dtype=cp.int32)
            self.u_offset = cp.asarray(packed.u_offset, dtype=cp.int64)
            self.v_offset = cp.asarray(packed.v_offset, dtype=cp.int64)
            self.direct_device_pack = False

        if device_blocks is not None:
            try:
                delattr(data, "_gpu_far_blocks")
            except AttributeError:
                pass

        self._x_perm = cp.empty(self.size, dtype=cp.float64)
        self._y_perm = cp.empty(self.size, dtype=cp.float64)

        self._farfield_kernel = cp.RawKernel(
            _farfield_kernel_source(self._real_c_type),
            "farfield_fused_matvec",
            options=("--std=c++11",),
        )

        # z_shared is double regardless of the factor precision.
        self._shared_mem_bytes = max(1, self.max_rank) * np.dtype(np.float64).itemsize

    def _apply_farfield(self):
        if self.n_far_blocks == 0:
            return

        self._farfield_kernel(
            (self.n_far_blocks,),
            (self.threads_per_block,),
            (
                self.U_flat,
                self.V_flat,
                self.row_start,
                self.row_stop,
                self.col_start,
                self.col_stop,
                self.ranks,
                self.u_offset,
                self.v_offset,
                self._x_perm,
                self._y_perm,
                np.int32(self.n_far_blocks),
            ),
            shared_mem=self._shared_mem_bytes,
        )

    def matvec(self, x_gpu, out=None):
        cp = self.cp
        x_gpu = cp.asarray(x_gpu, dtype=cp.float64)

        if x_gpu.shape != (self.size,):
            raise ValueError(f"Expected x shape {(self.size,)}, got {x_gpu.shape}.")

        self._x_perm[:] = x_gpu[self.perm]
        self._y_perm[:] = self.diag_jump_perm * self._x_perm
        self._y_perm += self.near @ self._x_perm
        self._apply_farfield()

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
        """
        Measure the device-resident MatVec using CUDA events.

        The reported interval includes permutation, exact diagonal, near CSR,
        fused far field and scatter to the original ordering. It excludes host
        transfers.
        """
        cp = self.cp
        x_gpu = cp.asarray(x, dtype=cp.float64)
        out = cp.empty_like(x_gpu)

        for _ in range(warmup):
            self.matvec(x_gpu, out=out)
        cp.cuda.get_current_stream().synchronize()

        start = cp.cuda.Event()
        stop = cp.cuda.Event()

        start.record()
        for _ in range(repeats):
            self.matvec(x_gpu, out=out)
        stop.record()
        stop.synchronize()

        elapsed_ms = float(cp.cuda.get_elapsed_time(start, stop))
        elapsed_seconds = elapsed_ms * 1e-3

        return {
            "timer": "cuda_events",
            "repeats": int(repeats),
            "total_seconds": elapsed_seconds,
            "average_seconds": elapsed_seconds / repeats,
        }

    def benchmark_roundtrip(self, x: np.ndarray, repeats: int = 20, warmup: int = 3):
        """
        Measure the current Stage-A CPU -> GPU -> CPU adapter path.
        """
        cp = self.cp

        for _ in range(warmup):
            self.matvec_numpy(x)
        cp.cuda.get_current_stream().synchronize()

        t0 = perf_counter()
        for _ in range(repeats):
            self.matvec_numpy(x)
        cp.cuda.get_current_stream().synchronize()
        elapsed = perf_counter() - t0

        return {
            "timer": "perf_counter_roundtrip",
            "repeats": int(repeats),
            "total_seconds": float(elapsed),
            "average_seconds": float(elapsed / repeats),
        }

    def memory_report(self) -> dict:
        cp = self.cp

        arrays = {
            "perm": self.perm,
            "diag_jump_perm": self.diag_jump_perm,
            "near_data": self.near.data,
            "near_indices": self.near.indices,
            "near_indptr": self.near.indptr,
            "U_flat": self.U_flat,
            "V_flat": self.V_flat,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "col_start": self.col_start,
            "col_stop": self.col_stop,
            "ranks": self.ranks,
            "u_offset": self.u_offset,
            "v_offset": self.v_offset,
            "x_perm_buffer": self._x_perm,
            "y_perm_buffer": self._y_perm,
        }

        item_bytes = {
            name: int(arr.nbytes)
            for name, arr in arrays.items()
        }
        accounted = int(sum(item_bytes.values()))

        pool = cp.get_default_memory_pool()

        return {
            "backend": self.backend_name,
            "n_far_blocks": int(self.n_far_blocks),
            "max_rank": int(self.max_rank),
            "direct_device_pack": bool(self.direct_device_pack),
            "shared_memory_bytes_per_cuda_block": int(self._shared_mem_bytes),
            "accounted_gpu_bytes": accounted,
            "accounted_gpu_human": _format_bytes(accounted),
            "cupy_pool_used_bytes": int(pool.used_bytes()),
            "cupy_pool_used_human": _format_bytes(pool.used_bytes()),
            "cupy_pool_total_bytes": int(pool.total_bytes()),
            "cupy_pool_total_human": _format_bytes(pool.total_bytes()),
            "items": item_bytes,
        }

    def print_memory_report(self, prefix: str = "[HMatrixGPUPackedFused]"):
        report = self.memory_report()

        print(f"{prefix} backend                     : {report['backend']}")
        print(f"{prefix} far low-rank blocks         : {report['n_far_blocks']}")
        print(f"{prefix} maximum far rank            : {report['max_rank']}")
        print(f"{prefix} direct GPU factor pack       : {report['direct_device_pack']}")
        print(
            f"{prefix} shared mem / CUDA block     : "
            f"{_format_bytes(report['shared_memory_bytes_per_cuda_block'])}"
        )
        print(
            f"{prefix} accounted GPU memory        : "
            f"{report['accounted_gpu_human']}"
        )
        print(
            f"{prefix} CuPy pool used memory       : "
            f"{report['cupy_pool_used_human']}"
        )
        print(
            f"{prefix} CuPy pool reserved memory   : "
            f"{report['cupy_pool_total_human']}",
            flush=True,
        )
