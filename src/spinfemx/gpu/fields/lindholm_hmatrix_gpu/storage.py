from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp


@dataclass
class LowRankBlockCPU:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    U: np.ndarray
    V: np.ndarray
    relative_residual: float = 0.0

    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    @property
    def storage_entries(self) -> int:
        return int(self.U.size + self.V.size)


@dataclass
class HMatrixCPUData:
    perm: np.ndarray
    diag_jump: np.ndarray
    near_csr: sp.csr_matrix
    far_blocks: list[LowRankBlockCPU]
    metadata: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.perm.size)

    @property
    def inverse_perm(self) -> np.ndarray:
        inv = np.empty_like(self.perm)
        inv[self.perm] = np.arange(self.perm.size, dtype=self.perm.dtype)
        return inv

    @property
    def dense_theoretical_entries(self) -> int:
        return int(self.size * self.size)

    @property
    def stored_entries(self) -> int:
        far = sum(block.storage_entries for block in self.far_blocks)
        return int(self.near_csr.nnz + far + self.diag_jump.size)

    @property
    def compression_ratio_entries(self) -> float:
        return self.dense_theoretical_entries / max(self.stored_entries, 1)

    def matvec_cpu(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.size,):
            raise ValueError(f"Expected x.shape == {(self.size,)}, got {x.shape}.")

        x_perm = x[self.perm]
        y_perm = self.diag_jump[self.perm] * x_perm
        y_perm = y_perm + self.near_csr @ x_perm

        for block in self.far_blocks:
            xj = x_perm[block.col_start:block.col_stop]
            y_perm[block.row_start:block.row_stop] += block.U @ (block.V @ xj)

        y = np.empty_like(x)
        y[self.perm] = y_perm
        return y

    def save(self, path: str | Path, *, compressed: bool = True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        far_meta = []
        U_parts = []
        V_parts = []
        u_offset = 0
        v_offset = 0

        for block in self.far_blocks:
            U_flat = np.ascontiguousarray(block.U, dtype=np.float64).ravel()
            V_flat = np.ascontiguousarray(block.V, dtype=np.float64).ravel()

            far_meta.append(
                (
                    block.row_start,
                    block.row_stop,
                    block.col_start,
                    block.col_stop,
                    block.rank,
                    u_offset,
                    U_flat.size,
                    v_offset,
                    V_flat.size,
                    float(block.relative_residual),
                )
            )
            U_parts.append(U_flat)
            V_parts.append(V_flat)
            u_offset += U_flat.size
            v_offset += V_flat.size

        far_meta_dtype = np.dtype(
            [
                ("row_start", np.int64),
                ("row_stop", np.int64),
                ("col_start", np.int64),
                ("col_stop", np.int64),
                ("rank", np.int64),
                ("u_offset", np.int64),
                ("u_size", np.int64),
                ("v_offset", np.int64),
                ("v_size", np.int64),
                ("relative_residual", np.float64),
            ]
        )

        far_meta_array = np.asarray(far_meta, dtype=far_meta_dtype)
        U_flat_all = np.concatenate(U_parts) if U_parts else np.empty(0, dtype=np.float64)
        V_flat_all = np.concatenate(V_parts) if V_parts else np.empty(0, dtype=np.float64)

        # The matrix is already numerically compressed.  ZIP/DEFLATE often
        # saves little on floating-point factors while consuming noticeable CPU
        # time, so callers that prioritize startup time can request an
        # uncompressed NPZ archive.
        save_npz = np.savez_compressed if compressed else np.savez
        # Passing a file object prevents NumPy from silently appending ".npz"
        # when a caller intentionally supplies a cache path without that suffix.
        with path.open("wb") as archive_file:
            save_npz(
                archive_file,
                perm=np.asarray(self.perm, dtype=np.int32),
                diag_jump=np.asarray(self.diag_jump, dtype=np.float64),
                near_data=np.asarray(self.near_csr.data, dtype=np.float64),
                near_indices=np.asarray(self.near_csr.indices, dtype=np.int32),
                near_indptr=np.asarray(self.near_csr.indptr, dtype=np.int32),
                near_shape=np.asarray(self.near_csr.shape, dtype=np.int64),
                far_meta=far_meta_array,
                far_U_flat=U_flat_all,
                far_V_flat=V_flat_all,
                metadata_json=np.asarray(json.dumps(self.metadata)),
            )

    @classmethod
    def load(cls, path: str | Path):
        archive = np.load(Path(path), allow_pickle=False)
        perm = np.asarray(archive["perm"], dtype=np.int32)
        diag_jump = np.asarray(archive["diag_jump"], dtype=np.float64)
        shape = tuple(np.asarray(archive["near_shape"], dtype=np.int64).tolist())

        near_csr = sp.csr_matrix(
            (
                np.asarray(archive["near_data"], dtype=np.float64),
                np.asarray(archive["near_indices"], dtype=np.int32),
                np.asarray(archive["near_indptr"], dtype=np.int32),
            ),
            shape=shape,
        )

        U_flat_all = np.asarray(archive["far_U_flat"], dtype=np.float64)
        V_flat_all = np.asarray(archive["far_V_flat"], dtype=np.float64)
        far_blocks = []

        for meta in archive["far_meta"]:
            row_start = int(meta["row_start"])
            row_stop = int(meta["row_stop"])
            col_start = int(meta["col_start"])
            col_stop = int(meta["col_stop"])
            rank = int(meta["rank"])

            m = row_stop - row_start
            n = col_stop - col_start

            u0 = int(meta["u_offset"])
            u1 = u0 + int(meta["u_size"])
            v0 = int(meta["v_offset"])
            v1 = v0 + int(meta["v_size"])

            U = np.ascontiguousarray(U_flat_all[u0:u1].reshape(m, rank))
            V = np.ascontiguousarray(V_flat_all[v0:v1].reshape(rank, n))

            far_blocks.append(
                LowRankBlockCPU(
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                    U=U,
                    V=V,
                    relative_residual=float(meta["relative_residual"]),
                )
            )

        metadata_raw = str(np.asarray(archive["metadata_json"]).item())
        metadata = json.loads(metadata_raw) if metadata_raw else {}

        return cls(
            perm=perm,
            diag_jump=diag_jump,
            near_csr=near_csr,
            far_blocks=far_blocks,
            metadata=metadata,
        )


class HMatrixStorageBuilder:
    """Accumulates dense near-field blocks and low-rank far-field blocks."""

    def __init__(self, size: int):
        self.size = int(size)
        self._near_rows = []
        self._near_cols = []
        self._near_data = []
        self.far_blocks: list[LowRankBlockCPU] = []

    def add_dense(self, row_start: int, row_stop: int, col_start: int, col_stop: int, A):
        A = np.asarray(A, dtype=np.float64)
        expected_shape = (row_stop - row_start, col_stop - col_start)
        if A.shape != expected_shape:
            raise ValueError(f"Dense block shape {A.shape} != expected {expected_shape}.")

        local_rows, local_cols = np.nonzero(A)
        if local_rows.size == 0:
            return

        self._near_rows.append(local_rows.astype(np.int64) + row_start)
        self._near_cols.append(local_cols.astype(np.int64) + col_start)
        self._near_data.append(A[local_rows, local_cols])

    def add_low_rank(
        self,
        row_start: int,
        row_stop: int,
        col_start: int,
        col_stop: int,
        U,
        V,
        relative_residual: float,
    ):
        self.far_blocks.append(
            LowRankBlockCPU(
                row_start=int(row_start),
                row_stop=int(row_stop),
                col_start=int(col_start),
                col_stop=int(col_stop),
                U=np.ascontiguousarray(U, dtype=np.float64),
                V=np.ascontiguousarray(V, dtype=np.float64),
                relative_residual=float(relative_residual),
            )
        )

    def finalize(self, perm: np.ndarray, diag_jump: np.ndarray, metadata: dict):
        if self._near_rows:
            rows = np.concatenate(self._near_rows)
            cols = np.concatenate(self._near_cols)
            data = np.concatenate(self._near_data)
            near = sp.coo_matrix((data, (rows, cols)), shape=(self.size, self.size)).tocsr()
            near.sum_duplicates()
        else:
            near = sp.csr_matrix((self.size, self.size), dtype=np.float64)

        return HMatrixCPUData(
            perm=np.asarray(perm, dtype=np.int32),
            diag_jump=np.asarray(diag_jump, dtype=np.float64),
            near_csr=near,
            far_blocks=self.far_blocks,
            metadata=dict(metadata),
        )
