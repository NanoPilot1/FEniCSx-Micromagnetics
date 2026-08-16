from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .backend import HMatrixGPU
from .backend_fused import HMatrixGPUPackedFused
from .builder import HMatrixBuildConfig, build_or_load_hmatrix, print_hmatrix_summary
from .entry_provider import LindholmEntryProviderCPU
from .entry_provider_gpu import LindholmEntryProviderGPU


_GPU_BACKENDS = {
    "legacy": HMatrixGPU,
    "packed_fused": HMatrixGPUPackedFused,
}


@dataclass
class HMatrixGPUReplacementSingleRank:
    """
    Single-rank adapter that replaces only the boundary double-layer MatVec.

    Available GPU backends:
      - legacy: Python loop over low-rank blocks. Kept as a validated reference.
      - packed_fused: flat U/V storage and one fused RawKernel launch for the
        complete far field.
    """

    cpu_data: object
    gpu_backend: object
    provider: object
    gpu_backend_name: str

    @classmethod
    def from_opB(
        cls,
        opB,
        cache_path: str | Path,
        epsilon: float = 1e-6,
        eta: float = 2.0,
        leaf_size: int = 64,
        compressor: str = "fullaca",
        max_rank: int | None = None,
        max_temporary_block_bytes: int = 256 * 1024 * 1024,
        force_rebuild: bool = False,
        cache_compressed: bool = False,
        verbose: bool = True,
        build_backend: str = "gpu",
        build_device_id: int = 0,
        build_threads_per_block: int = 128,
        gpu_entry_kernel: str = "auto",
        gpu_fast_math: bool = False,
        gpu_index_cache_max_bytes: int = 128 * 1024**2,
        gpu_memory_fraction: float = 0.75,
        gpu_fallback_to_cpu: bool = True,
        gpu_aca_residual_check_interval: int = 1,
        gpu_rsvd_initial_rank: int = 16,
        gpu_rsvd_oversampling: int = 8,
        gpu_rsvd_power_iterations: int = 1,
        gpu_progress_interval_seconds: float = 5.0,
        gpu_backend: str = "packed_fused",
        threads_per_block: int = 128,
        use_fp32: bool = True,
    ):
        comm = getattr(opB, "comm", None)
        if comm is not None and int(comm.size) != 1:
            raise RuntimeError(
                "HMatrixGPUReplacementSingleRank is intentionally single-rank. "
                "Run with one MPI rank for validation."
            )

        gpu_backend = str(gpu_backend).strip().lower()
        if gpu_backend not in _GPU_BACKENDS:
            raise ValueError(
                f"Unsupported GPU backend {gpu_backend!r}. "
                f"Use one of {sorted(_GPU_BACKENDS)}."
            )

        requested_build_backend = str(build_backend).strip().lower()
        if requested_build_backend == "cpu":
            provider = LindholmEntryProviderCPU.from_opB(opB)
        else:
            provider = LindholmEntryProviderGPU.from_opB(
                opB,
                device_id=int(build_device_id),
                threads_per_block=int(build_threads_per_block),
                kernel_mode=str(gpu_entry_kernel),
                fast_math=bool(gpu_fast_math),
                index_cache_max_bytes=int(gpu_index_cache_max_bytes),
            )

        config = HMatrixBuildConfig(
            epsilon=epsilon,
            eta=eta,
            leaf_size=leaf_size,
            compressor=compressor,
            max_rank=max_rank,
            max_temporary_block_bytes=max_temporary_block_bytes,
            build_backend=requested_build_backend,
            gpu_device_id=int(build_device_id),
            gpu_memory_fraction=float(gpu_memory_fraction),
            gpu_fallback_to_cpu=bool(gpu_fallback_to_cpu),
            gpu_aca_residual_check_interval=int(
                gpu_aca_residual_check_interval
            ),
            gpu_rsvd_initial_rank=int(gpu_rsvd_initial_rank),
            gpu_rsvd_oversampling=int(gpu_rsvd_oversampling),
            gpu_rsvd_power_iterations=int(gpu_rsvd_power_iterations),
            gpu_progress_interval_seconds=float(
                gpu_progress_interval_seconds
            ),
        )

        cpu_data = build_or_load_hmatrix(
            cache_path=cache_path,
            Xb=opB.Xb,
            diag_jump=opB.diag_jump,
            provider=provider,
            config=config,
            force_rebuild=force_rebuild,
            verbose=verbose,
            cache_compressed=cache_compressed,
        )

        if verbose:
            print_hmatrix_summary(cpu_data, prefix="[HMatrixGPU]")

        backend_class = _GPU_BACKENDS[gpu_backend]

        if gpu_backend == "packed_fused":
            backend = backend_class(
                cpu_data,
                threads_per_block=threads_per_block,
                use_fp32=use_fp32,
            )
        else:
            backend = backend_class(cpu_data)

        if verbose and hasattr(backend, "print_memory_report"):
            backend.print_memory_report()

        return cls(
            cpu_data=cpu_data,
            gpu_backend=backend,
            provider=provider,
            gpu_backend_name=gpu_backend,
        )

    def apply_numpy(self, x_boundary: np.ndarray) -> np.ndarray:
        return self.gpu_backend.matvec_numpy(x_boundary)

    def benchmark_roundtrip(self, x_boundary: np.ndarray, repeats: int, warmup: int):
        if hasattr(self.gpu_backend, "benchmark_roundtrip"):
            return self.gpu_backend.benchmark_roundtrip(
                x_boundary,
                repeats=repeats,
                warmup=warmup,
            )

        # Backwards-compatible fallback for the legacy backend.
        from time import perf_counter

        for _ in range(warmup):
            self.apply_numpy(x_boundary)

        t0 = perf_counter()
        for _ in range(repeats):
            self.apply_numpy(x_boundary)
        elapsed = perf_counter() - t0

        return {
            "timer": "perf_counter_roundtrip",
            "repeats": int(repeats),
            "total_seconds": float(elapsed),
            "average_seconds": float(elapsed / repeats),
        }

    def validate_against_dense(self, seed: int = 1234, trials: int = 3):
        rng = np.random.default_rng(seed)
        nb = self.cpu_data.size
        rows = np.arange(nb, dtype=np.int32)
        fill_cpu = getattr(self.provider, "fill_block_cpu", None)
        if fill_cpu is None:
            dense_offdiag = np.asarray(self.provider.fill_block(rows, rows))
        else:
            dense_offdiag = fill_cpu(rows, rows)
        dense = np.asarray(dense_offdiag, dtype=np.float64).copy()
        dense[np.arange(nb), np.arange(nb)] += self.cpu_data.diag_jump

        errors = []
        for _ in range(trials):
            x = rng.standard_normal(nb)
            y_ref = dense @ x
            y_gpu = self.apply_numpy(x)
            rel = np.linalg.norm(y_gpu - y_ref) / max(np.linalg.norm(y_ref), 1e-300)
            errors.append(float(rel))

        return {
            "trials": int(trials),
            "maximum_relative_error": max(errors, default=0.0),
            "relative_errors": errors,
        }


def apply_B_to_u1_boundary_single_rank(demag):
    """
    Requirements:
      - demag.B_gpu is an HMatrixGPUReplacementSingleRank instance.
      - demag._set_boundary_x_from_u1() already fills demag._x_global.
      - demag.opB contains the same boundary mappings used by the HTool class.
    """
    if int(demag.comm.size) != 1:
        raise RuntimeError("This helper only supports a single MPI rank.")

    demag._set_boundary_x_from_u1()
    demag._y_global[:] = demag.B_gpu.apply_numpy(demag._x_global)

    demag.g_u2_fun.x.array[:] = 0.0
    if demag.opB.bdofs_owned.size > 0:
        demag.g_u2_fun.x.array[demag.opB.bdofs_owned] = (
            demag._y_global[demag.opB.bidx_owned]
        )

    demag.g_u2_fun.x.scatter_forward()
