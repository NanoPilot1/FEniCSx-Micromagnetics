from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import gc
import json
import math
import numpy as np

from .compressors import (
    compress_dense_block,
    compress_dense_block_gpu,
    gpu_workspace_multiplier,
)
from .storage import HMatrixCPUData, HMatrixStorageBuilder
from .tree import ClusterNode, admissible, build_cluster_tree


HMATRIX_CONSTRUCTION_VERSION = "lindholm-hmatrix-build-v2"


@dataclass
class HMatrixBuildConfig:
    epsilon: float = 1e-6
    eta: float = 2.0
    leaf_size: int = 64
    compressor: str = "fullaca"
    max_rank: int | None = None
    max_temporary_block_bytes: int = 256 * 1024 * 1024
    low_rank_storage_factor: float = 0.95

    # Construction backend.  CPU is the reference path.  GPU evaluates
    # Lindholm blocks and compresses them with CuPy.  Auto selects GPU only when
    # the provider supports it and a CUDA device is visible.
    build_backend: str = "cpu"
    gpu_device_id: int = 0
    gpu_memory_fraction: float = 0.75
    gpu_fallback_to_cpu: bool = True
    gpu_aca_residual_check_interval: int = 1
    gpu_rsvd_initial_rank: int = 16
    gpu_rsvd_oversampling: int = 8
    gpu_rsvd_power_iterations: int = 1
    gpu_progress_interval_seconds: float = 5.0

    def __post_init__(self):
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be positive.")
        if self.max_temporary_block_bytes < 8:
            raise ValueError("max_temporary_block_bytes is too small.")
        if not (0.0 < self.low_rank_storage_factor <= 1.0):
            raise ValueError("low_rank_storage_factor must be in (0, 1].")

        self.build_backend = str(self.build_backend).strip().lower()
        if self.build_backend not in {"cpu", "gpu", "auto"}:
            raise ValueError("build_backend must be 'cpu', 'gpu', or 'auto'.")

        compressor = str(self.compressor).strip().lower()
        if compressor not in {
            "fullaca",
            "aca",
            "svd",
            "rsvd",
            "randomized_svd",
            "randomized-svd",
        }:
            raise ValueError(
                "compressor must be 'fullaca', 'svd', or 'rsvd'."
            )
        if compressor == "aca":
            compressor = "fullaca"
        elif compressor in {"randomized_svd", "randomized-svd"}:
            compressor = "rsvd"
        self.compressor = compressor

        self.gpu_device_id = int(self.gpu_device_id)
        if self.gpu_device_id < 0:
            raise ValueError("gpu_device_id must be non-negative.")
        if not (0.0 < float(self.gpu_memory_fraction) <= 1.0):
            raise ValueError("gpu_memory_fraction must be in (0, 1].")
        if self.gpu_aca_residual_check_interval < 1:
            raise ValueError("gpu_aca_residual_check_interval must be positive.")
        if self.gpu_rsvd_initial_rank < 1:
            raise ValueError("gpu_rsvd_initial_rank must be positive.")
        if self.gpu_rsvd_oversampling < 0:
            raise ValueError("gpu_rsvd_oversampling must be non-negative.")
        if self.gpu_rsvd_power_iterations < 0:
            raise ValueError("gpu_rsvd_power_iterations must be non-negative.")
        if self.gpu_progress_interval_seconds < 0.0:
            raise ValueError("gpu_progress_interval_seconds must be non-negative.")


def geometry_hash(Xb: np.ndarray, tri_lid: np.ndarray | None = None) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(Xb, dtype=np.float64).view(np.uint8))
    if tri_lid is not None:
        digest.update(np.ascontiguousarray(tri_lid, dtype=np.int32).view(np.uint8))
    return digest.hexdigest()


def _provider_signature(provider, backend: str | None = None) -> dict:
    backend_getter = getattr(provider, "cache_signature_for_backend", None)
    if backend is not None and callable(backend_getter):
        signature = backend_getter(backend)
    else:
        signature = getattr(provider, "cache_signature", None)
        if callable(signature):
            signature = signature()

    if signature is None:
        signature = {
            "provider": type(provider).__name__,
            "incident_policy": int(getattr(provider, "incident_policy", -1)),
        }
    return json.loads(json.dumps(signature, sort_keys=True))


def _provider_gpu_available(provider) -> bool:
    if not bool(getattr(provider, "supports_gpu", False)):
        return False
    checker = getattr(provider, "gpu_available", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _resolve_build_backend(provider, config: HMatrixBuildConfig) -> str:
    requested = config.build_backend
    if requested == "cpu":
        return "cpu"

    if _provider_gpu_available(provider):
        return "gpu"

    if requested == "auto" or config.gpu_fallback_to_cpu:
        return "cpu"

    raise RuntimeError(
        "GPU H-matrix construction was requested, but the entry provider cannot "
        "access a CUDA device and gpu_fallback_to_cpu=False."
    )


def _release_provider_construction_resources(provider) -> bool:
    release = getattr(provider, "release_construction_resources", None)
    if not callable(release):
        return False
    release()
    return True


def _cache_build_config(config: HMatrixBuildConfig, resolved_backend: str) -> dict:
    value = asdict(config)
    # Reporting cadence does not alter the numerical matrix and must not force a
    # costly rebuild when users only change log verbosity.
    value.pop("gpu_progress_interval_seconds", None)
    value["resolved_build_backend"] = str(resolved_backend)
    value["construction_version"] = HMATRIX_CONSTRUCTION_VERSION
    return value


def _block_random_seed(
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> int:
    # Stable integer mixing independent of Python's randomized hash seed.
    value = (
        int(row_start) * 73856093
        ^ int(row_stop) * 19349663
        ^ int(col_start) * 83492791
        ^ int(col_stop) * 2654435761
    )
    return int(value & 0x7FFFFFFF)


class HMatrixBuilder:
    """Build an H-matrix on the CPU in permuted coordinates."""

    resolved_build_backend = "cpu"

    def __init__(
        self,
        Xb: np.ndarray,
        diag_jump: np.ndarray,
        provider,
        config: HMatrixBuildConfig | None = None,
        *,
        verbose: bool = False,
    ):
        self.Xb = np.ascontiguousarray(Xb, dtype=np.float64)
        self.diag_jump = np.ascontiguousarray(diag_jump, dtype=np.float64)
        self.provider = provider
        self.config = config or HMatrixBuildConfig()
        self.verbose = bool(verbose)

        if self.Xb.ndim != 2 or self.Xb.shape[1] != 3:
            raise ValueError("Xb must have shape (Nb, 3).")
        if self.diag_jump.shape != (self.Xb.shape[0],):
            raise ValueError("diag_jump must have shape (Nb,).")
        if int(provider.size) != int(self.Xb.shape[0]):
            raise ValueError("provider.size must match Xb.shape[0].")

        self.Nb = int(self.Xb.shape[0])
        self.tree = build_cluster_tree(self.Xb, leaf_size=self.config.leaf_size)
        self.storage = HMatrixStorageBuilder(self.Nb)

        self.stats = {
            "dense_blocks": 0,
            "low_rank_blocks": 0,
            "rejected_low_rank_blocks": 0,
            "forced_subdivisions_memory": 0,
            "compression_tolerance_subdivisions": 0,
            "block_entry_evaluations": 0,
            "max_observed_rank": 0,
            "sum_observed_rank": 0,
            "cpu_fallback_blocks": 0,
        }

    def _indices(self, node: ClusterNode):
        return np.ascontiguousarray(
            self.tree.perm[node.start:node.stop], dtype=np.int32
        )

    def _dense_block(self, row_node: ClusterNode, col_node: ClusterNode):
        rows = self._indices(row_node)
        cols = self._indices(col_node)
        fill_cpu = getattr(self.provider, "fill_block_cpu", None)
        if fill_cpu is not None:
            A = fill_cpu(rows, cols)
        else:
            A = self.provider.fill_block(rows, cols)
        self.stats["block_entry_evaluations"] += int(A.size)
        return A

    def _add_dense(self, row_node: ClusterNode, col_node: ClusterNode):
        A = self._dense_block(row_node, col_node)
        self.storage.add_dense(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            A,
        )
        self.stats["dense_blocks"] += 1

    @staticmethod
    def _can_split(node: ClusterNode) -> bool:
        return not node.is_leaf

    def _subdivide(self, row_node: ClusterNode, col_node: ClusterNode):
        row_can_split = self._can_split(row_node)
        col_can_split = self._can_split(col_node)

        if not row_can_split and not col_can_split:
            self._add_dense(row_node, col_node)
            return

        if row_can_split and (
            not col_can_split or row_node.diameter >= col_node.diameter
        ):
            self._build_block(row_node.left, col_node)
            self._build_block(row_node.right, col_node)
        else:
            self._build_block(row_node, col_node.left)
            self._build_block(row_node, col_node.right)

    def _record_low_rank(self, row_node, col_node, result):
        self.storage.add_low_rank(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            result.U,
            result.V,
            relative_residual=result.relative_residual,
        )
        self.stats["low_rank_blocks"] += 1
        self.stats["max_observed_rank"] = max(
            self.stats["max_observed_rank"], result.rank
        )
        self.stats["sum_observed_rank"] += result.rank

    def _record_dense_rejection(self, row_node, col_node, A):
        self.storage.add_dense(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            A,
        )
        self.stats["dense_blocks"] += 1
        self.stats["rejected_low_rank_blocks"] += 1

    def _compress_cpu(self, A, row_node, col_node):
        config = self.config
        return compress_dense_block(
            A,
            method=config.compressor,
            rtol=config.epsilon,
            max_rank=config.max_rank,
            random_seed=_block_random_seed(
                row_node.start,
                row_node.stop,
                col_node.start,
                col_node.stop,
            ),
            rsvd_initial_rank=config.gpu_rsvd_initial_rank,
            rsvd_oversampling=config.gpu_rsvd_oversampling,
            rsvd_power_iterations=config.gpu_rsvd_power_iterations,
        )

    def _compression_meets_tolerance(self, result) -> bool:
        residual = float(result.relative_residual)
        if not np.isfinite(residual):
            return False
        # Allow only a roundoff-sized margin around the requested Frobenius
        # tolerance.  In particular, reaching max_rank is not enough by itself.
        margin = 1.0 + 64.0 * np.finfo(np.float64).eps
        return residual <= float(self.config.epsilon) * margin

    def _build_block(self, row_node: ClusterNode, col_node: ClusterNode):
        config = self.config
        dense_bytes = (
            row_node.size * col_node.size * np.dtype(np.float64).itemsize
        )
        is_admissible = admissible(row_node, col_node, eta=config.eta)

        if is_admissible and dense_bytes > config.max_temporary_block_bytes:
            self.stats["forced_subdivisions_memory"] += 1
            self._subdivide(row_node, col_node)
            return

        if is_admissible:
            A = self._dense_block(row_node, col_node)
            result = self._compress_cpu(A, row_node, col_node)

            dense_entries = int(A.size)
            low_rank_entries = int(result.storage_entries)

            meets_tolerance = self._compression_meets_tolerance(result)
            if (
                result.rank > 0
                and meets_tolerance
                and low_rank_entries
                < config.low_rank_storage_factor * dense_entries
            ):
                self._record_low_rank(row_node, col_node, result)
                return

            if (
                not meets_tolerance
                and (self._can_split(row_node) or self._can_split(col_node))
            ):
                self.stats["compression_tolerance_subdivisions"] += 1
                del A, result
                self._subdivide(row_node, col_node)
                return

            self._record_dense_rejection(row_node, col_node, A)
            return

        if row_node.is_leaf and col_node.is_leaf:
            self._add_dense(row_node, col_node)
            return

        self._subdivide(row_node, col_node)

    def build(self, extra_metadata: dict | None = None) -> HMatrixCPUData:
        t0 = perf_counter()
        self._build_block(self.tree.root, self.tree.root)
        elapsed = perf_counter() - t0

        low_rank_blocks = self.stats["low_rank_blocks"]
        avg_rank = (
            self.stats["sum_observed_rank"] / low_rank_blocks
            if low_rank_blocks
            else 0.0
        )

        metadata = {
            "Nb": self.Nb,
            "geometry_hash": geometry_hash(
                self.Xb,
                getattr(self.provider, "tri_lid", None),
            ),
            "build_config": asdict(self.config),
            "build_time_seconds": elapsed,
            "average_far_rank": avg_rank,
            "resolved_build_backend": self.resolved_build_backend,
            "construction_version": HMATRIX_CONSTRUCTION_VERSION,
            **self.stats,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        data = self.storage.finalize(
            perm=self.tree.perm,
            diag_jump=self.diag_jump,
            metadata=metadata,
        )
        data.metadata["near_nnz"] = int(data.near_csr.nnz)
        data.metadata["far_storage_entries"] = int(
            sum(block.storage_entries for block in data.far_blocks)
        )
        data.metadata["stored_entries_total"] = int(data.stored_entries)
        data.metadata["compression_ratio_entries"] = float(
            data.compression_ratio_entries
        )
        return data


class HMatrixBuilderGPU(HMatrixBuilder):
    """
    GPU construction path for the Lindholm H-matrix.

    The cluster tree and admissibility logic remain on the CPU.  Dense block
    entries, ACA/SVD compression and residual calculations are performed on the
    selected CUDA device.  Only accepted low-rank factors and terminal near
    blocks are copied back to host memory for the persistent cache and the
    existing packed MatVec backend.
    """

    resolved_build_backend = "gpu"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_progress = perf_counter()
        self._processed_terminal_blocks = 0
        # Transient device factors are attached to the completed CPU data so
        # HMatrixGPUPackedFused can pack them device-to-device instead of
        # copying the just-built factors CPU -> GPU again.
        self._far_device_blocks = []
        self.stats.update(
            {
                "gpu_blocks": 0,
                "gpu_oom_subdivisions": 0,
                "gpu_host_transfer_bytes": 0,
                "gpu_workspace_subdivisions": 0,
            }
        )

    @property
    def cp(self):
        # Accessing provider.cp invokes a lazy property that compiles the CUDA
        # kernels. Use the private state directly here so exception handling can
        # inspect an existing CuPy module without recursively recompiling.
        cp_value = getattr(self.provider, "_cp", None)
        if cp_value is not None and getattr(self.provider, "_device_ready", False):
            return cp_value

        ensure = getattr(self.provider, "_ensure_device", None)
        if ensure is None:
            raise RuntimeError(
                "The selected provider does not expose a device-resident block path."
            )
        ensure()
        cp_value = getattr(self.provider, "_cp", None)
        if cp_value is None:
            raise RuntimeError("GPU provider initialization did not expose CuPy.")
        return cp_value

    def _report_progress(self):
        interval = float(self.config.gpu_progress_interval_seconds)
        if not self.verbose or interval <= 0.0:
            return
        now = perf_counter()
        if now - self._last_progress < interval:
            return
        self._last_progress = now
        print(
            "[HMatrix GPU build] "
            f"terminal_blocks={self._processed_terminal_blocks}, "
            f"far={self.stats['low_rank_blocks']}, "
            f"near={self.stats['dense_blocks']}, "
            f"entries={self.stats['block_entry_evaluations']:,}, "
            f"host_transfer={self.stats['gpu_host_transfer_bytes'] / 1024**2:.3f} MiB",
            flush=True,
        )

    def _device_workspace_fits(self, dense_bytes: int) -> bool:
        config = self.config
        if dense_bytes > config.max_temporary_block_bytes:
            return False

        cp = self.cp
        driver_free_bytes, _ = cp.cuda.runtime.memGetInfo()

        # CuPy's memory pool keeps released blocks reserved from the CUDA
        # driver.  Those bytes are reusable by this process even though
        # memGetInfo() reports them as unavailable.  Ignoring them causes
        # unnecessary H-tree subdivisions after the first large block.
        pool = cp.get_default_memory_pool()
        pool_reusable_bytes = max(
            int(pool.total_bytes()) - int(pool.used_bytes()),
            0,
        )
        available_bytes = int(driver_free_bytes) + pool_reusable_bytes

        estimated = int(
            math.ceil(
                dense_bytes * gpu_workspace_multiplier(config.compressor)
            )
        )
        return estimated <= int(
            float(available_bytes) * config.gpu_memory_fraction
        )

    def _dense_block_gpu(self, row_node: ClusterNode, col_node: ClusterNode):
        rows = self._indices(row_node)
        cols = self._indices(col_node)
        fill_device = getattr(self.provider, "fill_block_device", None)
        if fill_device is None:
            A = self.provider.fill_block(rows, cols)
        else:
            A = fill_device(
                rows,
                cols,
                row_key=(row_node.start, row_node.stop),
                col_key=(col_node.start, col_node.stop),
            )
        self.stats["block_entry_evaluations"] += int(A.size)
        self.stats["gpu_blocks"] += 1
        return A

    def _dense_block_cpu_fallback(
        self, row_node: ClusterNode, col_node: ClusterNode
    ) -> np.ndarray:
        rows = self._indices(row_node)
        cols = self._indices(col_node)
        fill_cpu = getattr(self.provider, "fill_block_cpu", None)
        if fill_cpu is None:
            raise RuntimeError(
                "GPU construction failed and the provider has no CPU fallback."
            )
        A = fill_cpu(rows, cols)
        self.stats["block_entry_evaluations"] += int(A.size)
        self.stats["cpu_fallback_blocks"] += 1
        return A

    def _to_host(self, array):
        host = self.cp.asnumpy(array)
        self.stats["gpu_host_transfer_bytes"] += int(host.nbytes)
        return host

    def _add_dense_gpu(self, row_node: ClusterNode, col_node: ClusterNode):
        try:
            A_gpu = self._dense_block_gpu(row_node, col_node)
            A = self._to_host(A_gpu)
        except Exception as exc:
            if not self._is_gpu_oom(exc) or not self.config.gpu_fallback_to_cpu:
                raise
            # OOM tracebacks retain the failed device allocation frame.  Drop
            # those references before attempting the CPU fallback.
            try:
                exc.__traceback__ = None
            except Exception:
                pass
            try:
                del A_gpu
            except UnboundLocalError:
                pass
            gc.collect()
            self._release_free_pool_blocks()
            A = self._dense_block_cpu_fallback(row_node, col_node)

        self.storage.add_dense(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            A,
        )
        self.stats["dense_blocks"] += 1
        self._processed_terminal_blocks += 1
        self._report_progress()

    def _release_free_pool_blocks(self):
        # Cleanup must never initialize or compile the lazy GPU provider.
        cp = getattr(self.provider, "_cp", None)
        if cp is None:
            return
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _is_gpu_oom(self, exc: Exception) -> bool:
        """Classify allocation failures without triggering CUDA compilation."""
        cp = getattr(self.provider, "_cp", None)
        oom_type = ()
        if cp is not None:
            memory_module = getattr(getattr(cp, "cuda", None), "memory", None)
            oom_type = getattr(memory_module, "OutOfMemoryError", ())

        current = exc
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if oom_type and isinstance(current, oom_type):
                return True

            message = str(current).lower()
            if any(
                token in message
                for token in (
                    "out of memory",
                    "memory allocation",
                    "alloc failed",
                    "status_alloc_failed",
                    "cuda_error_memory_allocation",
                    "cuda_error_out_of_memory",
                )
            ):
                return True

            current = current.__cause__ or current.__context__

        return False

    def _subdivide_or_cpu_fallback(
        self,
        row_node: ClusterNode,
        col_node: ClusterNode,
        *,
        count_oom: bool,
    ):
        if self._can_split(row_node) or self._can_split(col_node):
            if count_oom:
                self.stats["gpu_oom_subdivisions"] += 1
            else:
                self.stats["gpu_workspace_subdivisions"] += 1
            self._subdivide(row_node, col_node)
            return

        if not self.config.gpu_fallback_to_cpu:
            reason = "GPU OOM" if count_oom else "GPU workspace limit"
            raise MemoryError(
                f"{reason} for an unsplittable H-matrix block "
                f"{row_node.size}x{col_node.size}."
            )

        A = self._dense_block_cpu_fallback(row_node, col_node)
        self.storage.add_dense(
            row_node.start,
            row_node.stop,
            col_node.start,
            col_node.stop,
            A,
        )
        self.stats["dense_blocks"] += 1
        self._processed_terminal_blocks += 1
        self._report_progress()

    def _compress_gpu(self, A_gpu, row_node, col_node):
        config = self.config
        return compress_dense_block_gpu(
            A_gpu,
            method=config.compressor,
            rtol=config.epsilon,
            max_rank=config.max_rank,
            random_seed=_block_random_seed(
                row_node.start,
                row_node.stop,
                col_node.start,
                col_node.stop,
            ),
            aca_residual_check_interval=config.gpu_aca_residual_check_interval,
            rsvd_initial_rank=config.gpu_rsvd_initial_rank,
            rsvd_oversampling=config.gpu_rsvd_oversampling,
            rsvd_power_iterations=config.gpu_rsvd_power_iterations,
        )

    def _build_block(self, row_node: ClusterNode, col_node: ClusterNode):
        config = self.config
        dense_bytes = (
            row_node.size * col_node.size * np.dtype(np.float64).itemsize
        )
        is_admissible = admissible(row_node, col_node, eta=config.eta)

        if is_admissible and not self._device_workspace_fits(dense_bytes):
            self.stats["forced_subdivisions_memory"] += 1
            self._subdivide_or_cpu_fallback(
                row_node,
                col_node,
                count_oom=False,
            )
            return

        if is_admissible:
            try:
                A_gpu = self._dense_block_gpu(row_node, col_node)
                result = self._compress_gpu(A_gpu, row_node, col_node)

                dense_entries = int(A_gpu.size)
                low_rank_entries = int(result.storage_entries)

                meets_tolerance = self._compression_meets_tolerance(result)
                if (
                    result.rank > 0
                    and meets_tolerance
                    and low_rank_entries
                    < config.low_rank_storage_factor * dense_entries
                ):
                    U_gpu = result.U
                    V_gpu = result.V
                    self._far_device_blocks.append(
                        (
                            int(row_node.start),
                            int(row_node.stop),
                            int(col_node.start),
                            int(col_node.stop),
                            U_gpu,
                            V_gpu,
                        )
                    )
                    U = self._to_host(U_gpu)
                    V = self._to_host(V_gpu)
                    result.U = U
                    result.V = V
                    self._record_low_rank(row_node, col_node, result)
                elif (
                    not meets_tolerance
                    and (self._can_split(row_node) or self._can_split(col_node))
                ):
                    self.stats["compression_tolerance_subdivisions"] += 1
                    del A_gpu, result
                    self._subdivide(row_node, col_node)
                    return
                else:
                    A = self._to_host(A_gpu)
                    self._record_dense_rejection(
                        row_node,
                        col_node,
                        A,
                    )

                self._processed_terminal_blocks += 1
                self._report_progress()
                return
            except Exception as exc:
                if not self._is_gpu_oom(exc):
                    raise
                # Clear the traceback and local references before recursively
                # building child blocks; otherwise the failed dense block and
                # compressor workspace can remain live during the retry.
                try:
                    exc.__traceback__ = None
                except Exception:
                    pass
                try:
                    del A_gpu
                except UnboundLocalError:
                    pass
                try:
                    del result
                except UnboundLocalError:
                    pass
                gc.collect()
                self._release_free_pool_blocks()
                self._subdivide_or_cpu_fallback(
                    row_node,
                    col_node,
                    count_oom=True,
                )
                return

        if row_node.is_leaf and col_node.is_leaf:
            self._add_dense_gpu(row_node, col_node)
            return

        self._subdivide(row_node, col_node)

    def build(self, extra_metadata: dict | None = None) -> HMatrixCPUData:
        # Compile NVRTC kernels and upload immutable geometry before traversing
        # the H-tree. Configuration/compiler errors then fail with a short,
        # direct traceback rather than after a deep recursive descent.
        prepare = getattr(self.provider, "prepare_device", None)
        prepare_elapsed = 0.0
        if callable(prepare):
            prepare_t0 = perf_counter()
            prepare()
            prepare_elapsed = perf_counter() - prepare_t0

        data = super().build(extra_metadata=extra_metadata)
        data.metadata["gpu_prepare_seconds"] = float(prepare_elapsed)
        data.metadata["build_time_seconds"] = float(
            data.metadata.get("build_time_seconds", 0.0) + prepare_elapsed
        )
        provider_stats = getattr(self.provider, "stats", None)
        if callable(provider_stats):
            data.metadata.update(provider_stats())

        if len(self._far_device_blocks) == len(data.far_blocks):
            # Deliberately transient and not serialized by HMatrixCPUData.save().
            # The packed backend consumes and clears this payload.
            data._gpu_far_blocks = self._far_device_blocks
            data.metadata["gpu_direct_far_pack_available"] = True
        else:
            data.metadata["gpu_direct_far_pack_available"] = False
        return data



def build_or_load_hmatrix(
    cache_path: str | Path,
    Xb: np.ndarray,
    diag_jump: np.ndarray,
    provider,
    config: HMatrixBuildConfig | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
    cache_compressed: bool = True,
) -> HMatrixCPUData:
    cache_path = Path(cache_path)
    config = config or HMatrixBuildConfig()
    resolved_backend = _resolve_build_backend(provider, config)
    expected_hash = geometry_hash(Xb, getattr(provider, "tri_lid", None))
    expected_config = _cache_build_config(config, resolved_backend)
    expected_provider_signature = _provider_signature(provider, resolved_backend)

    if cache_path.exists() and not force_rebuild:
        data = HMatrixCPUData.load(cache_path)
        cached_hash = data.metadata.get("geometry_hash")
        cached_config = data.metadata.get("build_config")
        cached_provider_signature = data.metadata.get("provider_signature")

        if (
            cached_hash == expected_hash
            and cached_config == expected_config
            and cached_provider_signature == expected_provider_signature
        ):
            data.metadata["cache_loaded"] = True
            data.metadata["cache_file_bytes"] = int(cache_path.stat().st_size)
            if verbose:
                print(f"[HMatrix] loaded cache: {cache_path}", flush=True)
            return data

        if verbose:
            print(
                "[HMatrix] cache metadata changed; rebuilding. "
                f"resolved_build_backend={resolved_backend}",
                flush=True,
            )

    if verbose:
        print(
            "[HMatrix] construction backend="
            f"{resolved_backend}, compressor={config.compressor}",
            flush=True,
        )

    def run_builder(backend_name: str):
        cls = HMatrixBuilderGPU if backend_name == "gpu" else HMatrixBuilder
        instance = cls(
            Xb=Xb,
            diag_jump=diag_jump,
            provider=provider,
            config=config,
            verbose=verbose,
        )
        result = instance.build(
            extra_metadata={
                "provider_signature": expected_provider_signature,
                "requested_build_backend": config.build_backend,
                "resolved_build_backend": backend_name,
            }
        )
        return result

    try:
        data = run_builder(resolved_backend)
    except Exception as exc:
        if resolved_backend != "gpu" or not config.gpu_fallback_to_cpu:
            raise
        if verbose:
            print(
                "[HMatrix] GPU construction failed; falling back to CPU. "
                f"Reason: {type(exc).__name__}: {exc}",
                flush=True,
            )
        _release_provider_construction_resources(provider)
        resolved_backend = "cpu"
        expected_config = _cache_build_config(config, resolved_backend)
        expected_provider_signature = _provider_signature(
            provider,
            resolved_backend,
        )
        data = run_builder(resolved_backend)
        data.metadata["gpu_build_fallback_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Cache equality must use the resolved backend, not only the requested one.
    data.metadata["build_config"] = expected_config
    data.metadata["resolved_build_backend"] = resolved_backend
    data.metadata["cache_compressed"] = bool(cache_compressed)
    data.metadata["cache_loaded"] = False
    data.metadata["gpu_construction_resources_released"] = (
        _release_provider_construction_resources(provider)
    )

    cache_t0 = perf_counter()
    data.save(cache_path, compressed=bool(cache_compressed))
    cache_elapsed = perf_counter() - cache_t0
    data.metadata["cache_write_seconds"] = float(cache_elapsed)
    data.metadata["cache_file_bytes"] = int(cache_path.stat().st_size)

    if verbose:
        print_hmatrix_summary(data, prefix="[HMatrix build]")
        cache_mode = "compressed" if cache_compressed else "uncompressed"
        print(
            f"[HMatrix] saved {cache_mode} cache in {cache_elapsed:.6f} s: "
            f"{cache_path}",
            flush=True,
        )

    return data



def print_hmatrix_summary(data: HMatrixCPUData, prefix: str = "[HMatrix]"):
    meta = data.metadata
    print(f"{prefix} Nb                         : {data.size}")
    print(
        f"{prefix} construction backend       : "
        f"{meta.get('resolved_build_backend', 'cpu')}"
    )
    print(
        f"{prefix} compressor                 : "
        f"{meta.get('build_config', {}).get('compressor', 'unknown')}"
    )
    print(f"{prefix} near CSR nnz               : {data.near_csr.nnz}")
    print(f"{prefix} far low-rank blocks        : {len(data.far_blocks)}")
    print(f"{prefix} average far rank           : {meta.get('average_far_rank', 0.0):.3f}")
    print(f"{prefix} maximum far rank           : {meta.get('max_observed_rank', 0)}")
    print(f"{prefix} stored entries             : {data.stored_entries}")
    print(f"{prefix} compression ratio entries  : {data.compression_ratio_entries:.3f}")
    print(
        f"{prefix} tolerance subdivisions    : "
        f"{meta.get('compression_tolerance_subdivisions', 0)}"
    )
    if meta.get("resolved_build_backend") == "gpu":
        print(
            f"{prefix} GPU-generated blocks       : {meta.get('gpu_blocks', 0)}"
        )
        print(
            f"{prefix} CPU fallback blocks        : {meta.get('cpu_fallback_blocks', 0)}"
        )
        print(
            f"{prefix} GPU workspace subdivisions: "
            f"{meta.get('gpu_workspace_subdivisions', 0)}"
        )
        print(
            f"{prefix} GPU OOM subdivisions      : "
            f"{meta.get('gpu_oom_subdivisions', 0)}"
        )
        print(
            f"{prefix} Lindholm target/tri pairs : "
            f"{meta.get('gpu_fill_target_triangle_pairs', 0):,}"
        )
        print(
            f"{prefix} entry kernels row/atomic  : "
            f"{meta.get('gpu_row_kernel_calls', 0)}/"
            f"{meta.get('gpu_atomic_kernel_calls', 0)}"
        )
        print(
            f"{prefix} GPU->host construction I/O : "
            f"{meta.get('gpu_host_transfer_bytes', 0) / 1024**2:.3f} MiB"
        )
    print(
        f"{prefix} build time                 : "
        f"{meta.get('build_time_seconds', 0.0):.6f} s"
    )
    if "cache_write_seconds" in meta:
        print(
            f"{prefix} cache write time           : "
            f"{meta.get('cache_write_seconds', 0.0):.6f} s",
            flush=True,
        )
    elif meta.get("cache_loaded"):
        print(f"{prefix} cache source               : loaded", flush=True)
