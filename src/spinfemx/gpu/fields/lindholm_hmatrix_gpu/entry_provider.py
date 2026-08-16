from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit, prange

from .lindholm_kernel import lindholm_weights_precomp


LINDHOLM_CPU_KERNEL_VERSION = "lindholm-block-numba-v2"


def build_node_incident_triangle_lists(tri_lid: np.ndarray, nb: int):
    """Build a compact source-node -> incident-triangle adjacency."""
    tri_lid = np.asarray(tri_lid, dtype=np.int32)
    counts = np.zeros(nb, dtype=np.int32)

    for t in range(tri_lid.shape[0]):
        counts[tri_lid[t, 0]] += 1
        counts[tri_lid[t, 1]] += 1
        counts[tri_lid[t, 2]] += 1

    ptr = np.zeros(nb + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)

    incident_tri = np.empty(ptr[-1], dtype=np.int32)
    incident_loc = np.empty(ptr[-1], dtype=np.int32)

    cursor = ptr[:-1].copy()
    for t in range(tri_lid.shape[0]):
        for loc in range(3):
            source_node = tri_lid[t, loc]
            pos = cursor[source_node]
            incident_tri[pos] = t
            incident_loc[pos] = loc
            cursor[source_node] += 1

    return ptr, incident_tri, incident_loc


@njit(parallel=True, fastmath=True, nogil=True)
def _fill_lindholm_block_triangle_numba(
    Xb,
    tri_lid,
    tri_pts,
    ntri,
    area,
    s_edge,
    eta,
    gamma,
    rows,
    rel_tri,
    rel_cols,
    incident_policy,
    out,
):
    """
    Evaluate the off-diagonal Lindholm block B[rows, cols] by source triangle.

    For every target row the loop visits each *relevant* triangle (a triangle
    incident to at least one block column) exactly once, evaluates the three
    Lindholm weights (w0, w1, w2) of that triangle a single time, and scatters
    each weight to the in-block column carried by the corresponding triangle
    vertex.  This replaces the previous incidence loop, which re-evaluated the
    same (target, triangle) weight tuple once per shared column (up to three
    times for an interior triangle).  Result is bit-for-bit the same
    accumulation; only the evaluation count drops.

    rel_tri  : (T,) global triangle indices touching the column set.
    rel_cols : (T, 3) local column index of each triangle vertex, or -1 if that
               vertex is not part of this block.

    The exact diagonal jump term is deliberately excluded and must be applied
    separately during MatVec.
    """
    m = rows.shape[0]
    n = out.shape[1]
    T = rel_tri.shape[0]

    for ii in prange(m):
        for jj in range(n):
            out[ii, jj] = 0.0

    for ii in prange(m):
        gi = rows[ii]
        x0 = Xb[gi]

        for tt in range(T):
            t = rel_tri[tt]

            a = tri_lid[t, 0]
            b = tri_lid[t, 1]
            c = tri_lid[t, 2]

            if incident_policy == 0 and (gi == a or gi == b or gi == c):
                continue

            c0 = rel_cols[tt, 0]
            c1 = rel_cols[tt, 1]
            c2 = rel_cols[tt, 2]

            w0, w1, w2 = lindholm_weights_precomp(
                x0,
                tri_pts[t, 0],
                tri_pts[t, 1],
                tri_pts[t, 2],
                ntri[t],
                area[t],
                s_edge[t],
                eta[t],
                gamma[t],
            )

            if c0 >= 0:
                out[ii, c0] += w0
            if c1 >= 0:
                out[ii, c1] += w1
            if c2 >= 0:
                out[ii, c2] += w2


@dataclass
class LindholmEntryProviderCPU:
    "CPU entry provider operating on the precomputed boundary geometry arrays (Xb, tri_lid, tri_pts, ...)"
    Xb: np.ndarray
    tri_lid: np.ndarray
    tri_pts: np.ndarray
    ntri: np.ndarray
    area: np.ndarray
    s_edge: np.ndarray
    eta: np.ndarray
    gamma: np.ndarray
    incident_policy: int = 0

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

        self.incident_ptr, self.incident_tri, self.incident_loc = (
            build_node_incident_triangle_lists(self.tri_lid, self.Xb.shape[0])
        )

        # Persistent scratch for mapping a global source node to its local
        # column index inside the current block.  Kept clean (all -1) between
        # fill_block calls so it never costs an O(Nb) reset per block.
        self._col_of_node = np.full(self.Xb.shape[0], -1, dtype=np.int32)

    @classmethod
    def from_opB(cls, opB):
        """Create a provider directly from DoubleLayerBoundaryDataMPI."""
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
        )

    @property
    def size(self) -> int:
        return int(self.Xb.shape[0])

    @property
    def supports_gpu(self) -> bool:
        return False

    @property
    def cache_signature(self) -> dict:
        return {
            "provider": type(self).__name__,
            "kernel_version": LINDHOLM_CPU_KERNEL_VERSION,
            "incident_policy": int(self.incident_policy),
        }

    def cache_signature_for_backend(self, backend: str) -> dict:
        return self.cache_signature

    def _relevant_triangles(self, cols: np.ndarray):
        """
        Triangles incident to at least one block column, each listed once,
        together with the local column index carried by each of their vertices
        (or -1 when that vertex does not belong to the block).

        Fully vectorized; touches only O(incidences + T) memory rather than the
        whole node set.
        """
        starts = self.incident_ptr[cols]
        stops = self.incident_ptr[cols + 1]
        counts = stops - starts
        total = int(counts.sum())

        if total == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.int32),
            )

        # Gather the incident triangles of every column without a Python loop.
        seg = np.repeat(np.arange(cols.size, dtype=np.int64), counts)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        positions = starts[seg] + within
        rel_tri = np.unique(self.incident_tri[positions]).astype(np.int32)

        # Map the three global vertices of each relevant triangle to local
        # block columns using the persistent (all -1) scratch buffer.
        self._col_of_node[cols] = np.arange(cols.size, dtype=np.int32)
        rel_cols = np.ascontiguousarray(
            self._col_of_node[self.tri_lid[rel_tri]], dtype=np.int32
        )
        self._col_of_node[cols] = -1  # restore clean state for the next block

        return rel_tri, rel_cols

    def fill_block(self, rows, cols) -> np.ndarray:
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        out = np.empty((rows.size, cols.size), dtype=np.float64)

        rel_tri, rel_cols = self._relevant_triangles(cols)

        _fill_lindholm_block_triangle_numba(
            self.Xb,
            self.tri_lid,
            self.tri_pts,
            self.ntri,
            self.area,
            self.s_edge,
            self.eta,
            self.gamma,
            rows,
            rel_tri,
            rel_cols,
            self.incident_policy,
            out,
        )
        return out
