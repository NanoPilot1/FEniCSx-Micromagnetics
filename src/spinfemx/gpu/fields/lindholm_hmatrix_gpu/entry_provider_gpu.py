from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import gc
import math
import numpy as np

from .entry_provider import LindholmEntryProviderCPU


LINDHOLM_GPU_KERNEL_VERSION = "lindholm-block-cuda-v2-headerless"


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on the GPU container
        raise RuntimeError(
            "CuPy is required to construct the Lindholm H-matrix on the GPU. "
            "Install the CuPy package matching the CUDA version in the container."
        ) from exc
    return cp


def cupy_gpu_available() -> bool:
    """Return True only when CuPy can see at least one CUDA device."""
    try:
        cp = _import_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


_LINDHOLM_BLOCK_CUDA_SOURCE = r"""
// Deliberately header-free.
//
// CuPy compiles RawModule with NVRTC by default. NVRTC does not automatically
// expose host C-library include directories in every container configuration,
// so including <math.h> can fail before the kernel is compiled. CUDA device
// math functions used below (sqrt, log1p and atan2) are available directly to
// device code and do not require that host header.

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

extern "C" __device__ __forceinline__
double dot3(const double ax, const double ay, const double az,
            const double bx, const double by, const double bz)
{
    return ax * bx + ay * by + az * bz;
}

extern "C" __device__ __forceinline__
double norm3(const double x, const double y, const double z)
{
    return sqrt(x * x + y * y + z * z);
}

extern "C" __device__ __forceinline__
double det3(
    const double ax, const double ay, const double az,
    const double bx, const double by, const double bz,
    const double cx, const double cy, const double cz)
{
    return ax * (by * cz - bz * cy)
         - ay * (bx * cz - bz * cx)
         + az * (bx * cy - by * cx);
}

extern "C" __device__ __forceinline__
double p_log1p(const double ri, const double rj, const double s)
{
    const double tiny = 1.0e-300;
    const double a = ri + rj;
    double denom = a - s;
    if (denom < tiny) {
        denom = tiny;
    }
    return log1p((2.0 * s) / denom);
}

extern "C" __device__ __forceinline__
void lindholm_weights(
    const double x0x, const double x0y, const double x0z,
    const double* __restrict__ tri_pts,
    const double* __restrict__ ntri,
    const double area,
    const double* __restrict__ s_edge,
    const double* __restrict__ eta,
    const double* __restrict__ gamma,
    double* w0,
    double* w1,
    double* w2)
{
    const double tiny = 1.0e-300;
    const double inv8pi = 0.0397887357729738339422209408431285905;

    const double p0x = tri_pts[0];
    const double p0y = tri_pts[1];
    const double p0z = tri_pts[2];
    const double p1x = tri_pts[3];
    const double p1y = tri_pts[4];
    const double p1z = tri_pts[5];
    const double p2x = tri_pts[6];
    const double p2y = tri_pts[7];
    const double p2z = tri_pts[8];

    const double rho0x = p0x - x0x;
    const double rho0y = p0y - x0y;
    const double rho0z = p0z - x0z;
    const double rho1x = p1x - x0x;
    const double rho1y = p1y - x0y;
    const double rho1z = p1z - x0z;
    const double rho2x = p2x - x0x;
    const double rho2y = p2y - x0y;
    const double rho2z = p2z - x0z;

    const double r0 = norm3(rho0x, rho0y, rho0z) + tiny;
    const double r1 = norm3(rho1x, rho1y, rho1z) + tiny;
    const double r2 = norm3(rho2x, rho2y, rho2z) + tiny;

    const double h = dot3(
        ntri[0], ntri[1], ntri[2],
        rho0x, rho0y, rho0z
    );

    const double eta0 = dot3(
        eta[0], eta[1], eta[2],
        rho0x, rho0y, rho0z
    );
    const double eta1 = dot3(
        eta[3], eta[4], eta[5],
        rho1x, rho1y, rho1z
    );
    const double eta2 = dot3(
        eta[6], eta[7], eta[8],
        rho2x, rho2y, rho2z
    );

    const double s0 = s_edge[0];
    const double s1 = s_edge[1];
    const double s2 = s_edge[2];

    const double P0 = p_log1p(r0, r1, s0);
    const double P1 = p_log1p(r1, r2, s1);
    const double P2 = p_log1p(r2, r0, s2);

    const double determinant = det3(
        rho0x, rho0y, rho0z,
        rho1x, rho1y, rho1z,
        rho2x, rho2y, rho2z
    );
    const double d01 = dot3(
        rho0x, rho0y, rho0z,
        rho1x, rho1y, rho1z
    );
    const double d12 = dot3(
        rho1x, rho1y, rho1z,
        rho2x, rho2y, rho2z
    );
    const double d20 = dot3(
        rho2x, rho2y, rho2z,
        rho0x, rho0y, rho0z
    );
    const double omega_denom =
        r0 * r1 * r2 + d01 * r2 + d12 * r0 + d20 * r1;
    const double omega = 2.0 * atan2(determinant, omega_denom);

    const double gP0 = gamma[0] * P0 + gamma[1] * P1 + gamma[2] * P2;
    const double gP1 = gamma[3] * P0 + gamma[4] * P1 + gamma[5] * P2;
    const double gP2 = gamma[6] * P0 + gamma[7] * P1 + gamma[8] * P2;

    const double coefficient = inv8pi / (area + tiny);

    // Same opposite-edge convention as the CPU implementation.
    *w0 = s1 * coefficient * (eta1 * omega - h * gP0);
    *w1 = s2 * coefficient * (eta2 * omega - h * gP1);
    *w2 = s0 * coefficient * (eta0 * omega - h * gP2);
}

extern "C" __global__
void fill_lindholm_block_rows(
    const double* __restrict__ Xb,
    const int* __restrict__ tri_lid,
    const double* __restrict__ tri_pts,
    const double* __restrict__ ntri,
    const double* __restrict__ area,
    const double* __restrict__ s_edge,
    const double* __restrict__ eta,
    const double* __restrict__ gamma,
    const int* __restrict__ rows,
    const int* __restrict__ rel_tri,
    const int* __restrict__ rel_cols,
    const int m,
    const int n,
    const int nrel,
    const int incident_policy,
    double* __restrict__ out)
{
    const int stride = blockDim.x * gridDim.x;

    for (int ii = blockDim.x * blockIdx.x + threadIdx.x;
         ii < m;
         ii += stride) {
        const long long row_offset = ((long long) ii) * n;
        const int gi = rows[ii];
        const double x0x = Xb[3 * gi + 0];
        const double x0y = Xb[3 * gi + 1];
        const double x0z = Xb[3 * gi + 2];

        for (int tt = 0; tt < nrel; ++tt) {
            const int t = rel_tri[tt];
            const int a = tri_lid[3 * t + 0];
            const int b = tri_lid[3 * t + 1];
            const int c = tri_lid[3 * t + 2];

            if (incident_policy == 0 && (gi == a || gi == b || gi == c)) {
                continue;
            }

            double w0;
            double w1;
            double w2;
            lindholm_weights(
                x0x, x0y, x0z,
                tri_pts + ((long long) t) * 9,
                ntri + ((long long) t) * 3,
                area[t],
                s_edge + ((long long) t) * 3,
                eta + ((long long) t) * 9,
                gamma + ((long long) t) * 9,
                &w0, &w1, &w2
            );

            const int c0 = rel_cols[3 * tt + 0];
            const int c1 = rel_cols[3 * tt + 1];
            const int c2 = rel_cols[3 * tt + 2];

            if (c0 >= 0) {
                out[row_offset + c0] += w0;
            }
            if (c1 >= 0) {
                out[row_offset + c1] += w1;
            }
            if (c2 >= 0) {
                out[row_offset + c2] += w2;
            }
        }
    }
}

extern "C" __global__
void fill_lindholm_block_atomic(
    const double* __restrict__ Xb,
    const int* __restrict__ tri_lid,
    const double* __restrict__ tri_pts,
    const double* __restrict__ ntri,
    const double* __restrict__ area,
    const double* __restrict__ s_edge,
    const double* __restrict__ eta,
    const double* __restrict__ gamma,
    const int* __restrict__ rows,
    const int* __restrict__ rel_tri,
    const int* __restrict__ rel_cols,
    const int m,
    const int n,
    const int nrel,
    const int incident_policy,
    double* __restrict__ out)
{
    const long long total = ((long long) m) * nrel;
    const long long stride = ((long long) blockDim.x) * gridDim.x;

    for (long long pair = ((long long) blockDim.x) * blockIdx.x + threadIdx.x;
         pair < total;
         pair += stride) {
        const int ii = (int) (pair / nrel);
        const int tt = (int) (pair - ((long long) ii) * nrel);
        const int gi = rows[ii];
        const int t = rel_tri[tt];

        const int a = tri_lid[3 * t + 0];
        const int b = tri_lid[3 * t + 1];
        const int c = tri_lid[3 * t + 2];

        if (incident_policy == 0 && (gi == a || gi == b || gi == c)) {
            continue;
        }

        const double x0x = Xb[3 * gi + 0];
        const double x0y = Xb[3 * gi + 1];
        const double x0z = Xb[3 * gi + 2];

        double w0;
        double w1;
        double w2;
        lindholm_weights(
            x0x, x0y, x0z,
            tri_pts + ((long long) t) * 9,
            ntri + ((long long) t) * 3,
            area[t],
            s_edge + ((long long) t) * 3,
            eta + ((long long) t) * 9,
            gamma + ((long long) t) * 9,
            &w0, &w1, &w2
        );

        const int c0 = rel_cols[3 * tt + 0];
        const int c1 = rel_cols[3 * tt + 1];
        const int c2 = rel_cols[3 * tt + 2];
        const long long row_offset = ((long long) ii) * n;

        if (c0 >= 0) {
            atomic_add_double_compat(out + row_offset + c0, w0);
        }
        if (c1 >= 0) {
            atomic_add_double_compat(out + row_offset + c1, w1);
        }
        if (c2 >= 0) {
            atomic_add_double_compat(out + row_offset + c2, w2);
        }
    }
}
"""


@dataclass
class LindholmEntryProviderGPU:
    """
    Device-resident Lindholm block provider.

    Geometry is copied to the selected CUDA device only on the first block
    evaluation, so loading an existing H-matrix cache has no construction-side
    GPU allocation.  The CPU provider is retained as a correctness/fallback
    path and to build compact source-node/triangle adjacency.
    """

    Xb: np.ndarray
    tri_lid: np.ndarray
    tri_pts: np.ndarray
    ntri: np.ndarray
    area: np.ndarray
    s_edge: np.ndarray
    eta: np.ndarray
    gamma: np.ndarray
    incident_policy: int = 0
    device_id: int = 0
    threads_per_block: int = 128
    kernel_mode: str = "auto"
    fast_math: bool = False
    index_cache_max_bytes: int = 128 * 1024**2

    def __post_init__(self):
        self.Xb = np.ascontiguousarray(self.Xb, dtype=np.float64)
        self.tri_lid = np.ascontiguousarray(self.tri_lid, dtype=np.int32)
        self.tri_pts = np.ascontiguousarray(self.tri_pts, dtype=np.float64)
        self.ntri = np.ascontiguousarray(self.ntri, dtype=np.float64)
        self.area = np.ascontiguousarray(self.area, dtype=np.float64)
        self.s_edge = np.ascontiguousarray(self.s_edge, dtype=np.float64)
        self.eta = np.ascontiguousarray(self.eta, dtype=np.float64)
        self.gamma = np.ascontiguousarray(self.gamma, dtype=np.float64)
        self.incident_policy = int(self.incident_policy)
        self.device_id = int(self.device_id)
        self.threads_per_block = int(self.threads_per_block)
        self.fast_math = bool(self.fast_math)
        self.index_cache_max_bytes = int(self.index_cache_max_bytes)

        if self.device_id < 0:
            raise ValueError("device_id must be non-negative.")
        if self.threads_per_block < 32 or self.threads_per_block > 1024:
            raise ValueError("threads_per_block must be between 32 and 1024.")
        if self.index_cache_max_bytes < 0:
            raise ValueError("index_cache_max_bytes must be non-negative.")

        self.kernel_mode = str(self.kernel_mode).strip().lower()
        if self.kernel_mode not in {"auto", "row", "atomic"}:
            raise ValueError("kernel_mode must be 'auto', 'row', or 'atomic'.")

        self.cpu_provider = LindholmEntryProviderCPU(
            Xb=self.Xb,
            tri_lid=self.tri_lid,
            tri_pts=self.tri_pts,
            ntri=self.ntri,
            area=self.area,
            s_edge=self.s_edge,
            eta=self.eta,
            gamma=self.gamma,
            incident_policy=self.incident_policy,
        )

        self._cp = None
        self._device_ready = False
        self._index_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._index_cache_bytes = 0

        self._stats = {
            "gpu_fill_calls": 0,
            "gpu_fill_entries": 0,
            "gpu_fill_target_triangle_pairs": 0,
            "gpu_row_kernel_calls": 0,
            "gpu_atomic_kernel_calls": 0,
            "gpu_index_cache_hits": 0,
            "gpu_index_cache_misses": 0,
            "gpu_index_cache_evictions": 0,
        }

    @classmethod
    def from_opB(
        cls,
        opB,
        *,
        device_id: int = 0,
        threads_per_block: int = 128,
        kernel_mode: str = "auto",
        fast_math: bool = False,
        index_cache_max_bytes: int = 128 * 1024**2,
    ):
        return cls(
            Xb=opB.Xb,
            tri_lid=opB.tri_lid,
            tri_pts=opB.tri_pts,
            ntri=opB.ntri,
            area=opB.area,
            s_edge=opB.s_edge,
            eta=opB.eta,
            gamma=opB.gamma,
            incident_policy=opB.incident_policy,
            device_id=device_id,
            threads_per_block=threads_per_block,
            kernel_mode=kernel_mode,
            fast_math=fast_math,
            index_cache_max_bytes=index_cache_max_bytes,
        )

    @property
    def size(self) -> int:
        return int(self.Xb.shape[0])

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def cp(self):
        self._ensure_device()
        return self._cp

    @property
    def cache_signature(self) -> dict:
        return {
            "provider": type(self).__name__,
            "kernel_version": LINDHOLM_GPU_KERNEL_VERSION,
            "incident_policy": int(self.incident_policy),
            "kernel_mode": self.kernel_mode,
            "threads_per_block": int(self.threads_per_block),
            "fast_math": bool(self.fast_math),
        }

    def cache_signature_for_backend(self, backend: str) -> dict:
        if str(backend).strip().lower() == "cpu":
            return self.cpu_provider.cache_signature
        return self.cache_signature

    def gpu_available(self) -> bool:
        try:
            cp = _import_cupy()
            return int(cp.cuda.runtime.getDeviceCount()) > self.device_id
        except Exception:
            return False

    def _clear_device_state(
        self,
        *,
        free_pool: bool,
        clear_cupy_module: bool,
    ):
        """Clear complete or partially initialized CUDA construction state."""
        self.clear_device_index_cache()
        cp = self._cp

        for name in (
            "_Xb_gpu",
            "_tri_lid_gpu",
            "_tri_pts_gpu",
            "_ntri_gpu",
            "_area_gpu",
            "_s_edge_gpu",
            "_eta_gpu",
            "_gamma_gpu",
            "_row_kernel",
            "_atomic_kernel",
            "_raw_module",
        ):
            if hasattr(self, name):
                delattr(self, name)

        self._device_ready = False
        if clear_cupy_module:
            self._cp = None

        gc.collect()
        if free_pool and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def _ensure_device(self):
        if self._device_ready:
            return

        cp = _import_cupy()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= self.device_id:
            raise RuntimeError(
                f"CUDA device {self.device_id} is not available. "
                f"Detected device count: {device_count}."
            )

        cp.cuda.Device(self.device_id).use()
        self._cp = cp

        options = ["--std=c++11"]
        if self.fast_math:
            options.append("--use_fast_math")

        # RawModule compilation is lazy: get_function() invokes NVRTC. Compile
        # before copying the potentially large boundary geometry so a compiler
        # error cannot leave duplicate device arrays allocated after a retry.
        try:
            module = cp.RawModule(
                code=_LINDHOLM_BLOCK_CUDA_SOURCE,
                options=tuple(options),
                backend="nvrtc",
            )
            row_kernel = module.get_function("fill_lindholm_block_rows")
            atomic_kernel = module.get_function("fill_lindholm_block_atomic")
        except Exception:
            self._clear_device_state(
                free_pool=True,
                clear_cupy_module=True,
            )
            raise

        # Allocate into a temporary mapping first. If one transfer fails, clear
        # all successfully allocated arrays before releasing CuPy's free pool.
        device_arrays = {}
        try:
            device_arrays["_Xb_gpu"] = cp.asarray(self.Xb, dtype=cp.float64)
            device_arrays["_tri_lid_gpu"] = cp.asarray(
                self.tri_lid, dtype=cp.int32
            )
            device_arrays["_tri_pts_gpu"] = cp.asarray(
                self.tri_pts, dtype=cp.float64
            )
            device_arrays["_ntri_gpu"] = cp.asarray(
                self.ntri, dtype=cp.float64
            )
            device_arrays["_area_gpu"] = cp.asarray(
                self.area, dtype=cp.float64
            )
            device_arrays["_s_edge_gpu"] = cp.asarray(
                self.s_edge, dtype=cp.float64
            )
            device_arrays["_eta_gpu"] = cp.asarray(
                self.eta, dtype=cp.float64
            )
            device_arrays["_gamma_gpu"] = cp.asarray(
                self.gamma, dtype=cp.float64
            )
        except Exception:
            device_arrays.clear()
            gc.collect()
            self._clear_device_state(
                free_pool=True,
                clear_cupy_module=True,
            )
            raise

        self._raw_module = module
        self._row_kernel = row_kernel
        self._atomic_kernel = atomic_kernel
        for name, value in device_arrays.items():
            setattr(self, name, value)
        self._device_ready = True

    def prepare_device(self):
        """Compile kernels and upload immutable boundary geometry once."""
        self._ensure_device()

    def _cache_get(self, key):
        if key is None:
            return None
        value = self._index_cache.get(key)
        if value is None:
            self._stats["gpu_index_cache_misses"] += 1
            return None
        self._index_cache.move_to_end(key)
        self._stats["gpu_index_cache_hits"] += 1
        return value[0]

    def _cache_put(self, key, value):
        if key is None or self.index_cache_max_bytes == 0:
            return value

        nbytes = 0
        arrays = value if isinstance(value, tuple) else (value,)
        for array in arrays:
            nbytes += int(array.nbytes)

        if nbytes > self.index_cache_max_bytes:
            return value

        old = self._index_cache.pop(key, None)
        if old is not None:
            self._index_cache_bytes -= int(old[1])

        while (
            self._index_cache
            and self._index_cache_bytes + nbytes > self.index_cache_max_bytes
        ):
            _, evicted = self._index_cache.popitem(last=False)
            self._index_cache_bytes -= int(evicted[1])
            self._stats["gpu_index_cache_evictions"] += 1

        self._index_cache[key] = (value, nbytes)
        self._index_cache_bytes += nbytes
        return value

    def _rows_gpu(self, rows: np.ndarray, row_key=None):
        key = None if row_key is None else ("rows", row_key)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        rows_gpu = self._cp.asarray(rows, dtype=self._cp.int32)
        return self._cache_put(key, rows_gpu)

    def _relevant_gpu(self, cols: np.ndarray, col_key=None):
        key = None if col_key is None else ("relevant", col_key)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        rel_tri, rel_cols = self.cpu_provider._relevant_triangles(cols)
        value = (
            self._cp.asarray(rel_tri, dtype=self._cp.int32),
            self._cp.asarray(rel_cols, dtype=self._cp.int32),
        )
        return self._cache_put(key, value)

    def _select_kernel_mode(self, m: int, nrel: int) -> str:
        if self.kernel_mode != "auto":
            return self.kernel_mode

        # The atomic path exposes m*nrel independent target/triangle pairs and
        # is much faster for the common case of many relevant panels per row.
        # For tiny blocks the deterministic row kernel avoids atomic overhead.
        pair_work = int(m) * int(nrel)
        if nrel >= 64 and pair_work >= 8192:
            return "atomic"
        return "row"

    def fill_block_device(
        self,
        rows,
        cols,
        *,
        row_key=None,
        col_key=None,
    ):
        self._ensure_device()
        cp = self._cp

        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        m = int(rows.size)
        n = int(cols.size)

        out = cp.zeros((m, n), dtype=cp.float64)
        if m == 0 or n == 0:
            return out

        rows_gpu = self._rows_gpu(rows, row_key=row_key)
        rel_tri_gpu, rel_cols_gpu = self._relevant_gpu(cols, col_key=col_key)
        nrel = int(rel_tri_gpu.size)
        if nrel == 0:
            return out

        args = (
            self._Xb_gpu,
            self._tri_lid_gpu,
            self._tri_pts_gpu,
            self._ntri_gpu,
            self._area_gpu,
            self._s_edge_gpu,
            self._eta_gpu,
            self._gamma_gpu,
            rows_gpu,
            rel_tri_gpu,
            rel_cols_gpu,
            np.int32(m),
            np.int32(n),
            np.int32(nrel),
            np.int32(self.incident_policy),
            out,
        )

        mode = self._select_kernel_mode(m, nrel)
        threads = self.threads_per_block
        if mode == "atomic":
            total = m * nrel
            blocks = max(1, math.ceil(total / threads))
            # Limit grid size; both kernels use grid-stride loops.
            blocks = min(blocks, 65535)
            self._atomic_kernel((blocks,), (threads,), args)
            self._stats["gpu_atomic_kernel_calls"] += 1
        else:
            blocks = max(1, math.ceil(m / threads))
            blocks = min(blocks, 65535)
            self._row_kernel((blocks,), (threads,), args)
            self._stats["gpu_row_kernel_calls"] += 1

        self._stats["gpu_fill_calls"] += 1
        self._stats["gpu_fill_entries"] += m * n
        self._stats["gpu_fill_target_triangle_pairs"] += m * nrel
        return out

    def fill_block(self, rows, cols):
        """Compatibility entry point used by the generic GPU builder."""
        return self.fill_block_device(rows, cols)

    def fill_block_cpu(self, rows, cols) -> np.ndarray:
        return self.cpu_provider.fill_block(rows, cols)

    def clear_device_index_cache(self):
        self._index_cache.clear()
        self._index_cache_bytes = 0

    def release_construction_resources(self):
        """Release construction data, including partial initialization state."""
        self._clear_device_state(
            free_pool=True,
            clear_cupy_module=True,
        )

    def stats(self) -> dict:
        return {
            **self._stats,
            "gpu_index_cache_bytes": int(self._index_cache_bytes),
            "gpu_index_cache_entries": int(len(self._index_cache)),
            "kernel_mode": self.kernel_mode,
            "kernel_version": LINDHOLM_GPU_KERNEL_VERSION,
        }
