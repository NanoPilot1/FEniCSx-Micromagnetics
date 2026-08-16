from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ClusterNode:
    start: int
    stop: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: "ClusterNode | None" = None
    right: "ClusterNode | None" = None

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))


@dataclass
class ClusterTree:
    points: np.ndarray
    perm: np.ndarray
    root: ClusterNode
    leaf_size: int


def _bbox(points: np.ndarray):
    return points.min(axis=0), points.max(axis=0)


def build_cluster_tree(points: np.ndarray, leaf_size: int = 64) -> ClusterTree:
    """
    Construct a binary spatial tree and a permutation such that each cluster
    occupies a contiguous interval in permuted coordinates.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    n = int(points.shape[0])

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if n == 0:
        raise ValueError("At least one point is required.")
    if leaf_size < 1:
        raise ValueError("leaf_size must be positive.")

    ordered_indices = []

    def recurse(indices: np.ndarray) -> ClusterNode:
        local_points = points[indices]
        bbox_min, bbox_max = _bbox(local_points)

        if indices.size <= leaf_size:
            start = len(ordered_indices)
            ordered_indices.extend(indices.tolist())
            stop = len(ordered_indices)
            return ClusterNode(start, stop, bbox_min, bbox_max)

        extent = bbox_max - bbox_min
        split_axis = int(np.argmax(extent))

        sort_order = np.argsort(local_points[:, split_axis], kind="mergesort")
        sorted_indices = indices[sort_order]
        midpoint = sorted_indices.size // 2

        # A defensive fallback for degenerate point clouds.
        if midpoint == 0 or midpoint == sorted_indices.size:
            midpoint = max(1, sorted_indices.size // 2)

        left = recurse(sorted_indices[:midpoint])
        right = recurse(sorted_indices[midpoint:])
        return ClusterNode(
            start=left.start,
            stop=right.stop,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            left=left,
            right=right,
        )

    root = recurse(np.arange(n, dtype=np.int32))
    perm = np.asarray(ordered_indices, dtype=np.int32)

    if perm.size != n or np.unique(perm).size != n:
        raise RuntimeError("Invalid cluster-tree permutation.")

    return ClusterTree(points=points, perm=perm, root=root, leaf_size=int(leaf_size))


def bbox_distance(a: ClusterNode, b: ClusterNode) -> float:
    """Euclidean distance between two axis-aligned bounding boxes."""
    gap_left = a.bbox_min - b.bbox_max
    gap_right = b.bbox_min - a.bbox_max
    gap = np.maximum(np.maximum(gap_left, gap_right), 0.0)
    return float(np.linalg.norm(gap))


def admissible(a: ClusterNode, b: ClusterNode, eta: float) -> bool:
    """
    Standard geometric admissibility condition:
        max(diam(a), diam(b)) <= eta * dist(a, b)
    """
    distance = bbox_distance(a, b)
    if distance <= 0.0:
        return False
    return max(a.diameter, b.diameter) <= float(eta) * distance
