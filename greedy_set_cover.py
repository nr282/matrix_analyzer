"""
Greedy Set Cover via Discretization for delta-coverings of matrix spaces.

Each input matrix is treated as a candidate center. Its "set" consists of all
matrices within Frobenius distance delta of it. The greedy algorithm iteratively
picks the candidate whose set covers the most currently-uncovered matrices,
achieving an O(log n) approximation ratio relative to the minimum cover size.
"""

import numpy as np
from typing import List, Set, Tuple


def _build_coverage(flat: np.ndarray, delta: float) -> List[Set[int]]:
    """
    For each index i, returns the set of indices j with ||flat[i] - flat[j]||_2 <= delta.
    flat has shape (n, d) where d is the flattened matrix dimension.
    """
    n = flat.shape[0]
    coverage: List[Set[int]] = []
    for i in range(n):
        dists = np.linalg.norm(flat - flat[i], axis=1)
        coverage.append(set(np.where(dists <= delta)[0].tolist()))
    return coverage


def greedy_set_cover_delta_covering(
    matrices: List[np.ndarray],
    delta: float,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Computes a delta-covering of `matrices` under the Frobenius norm via greedy set cover.

    Algorithm:
      1. Discretize: each matrix is a candidate center whose "set" is all matrices
         within Frobenius distance delta.
      2. Greedily pick the candidate covering the most uncovered matrices.
      3. Repeat until every matrix is covered.

    Parameters
    ----------
    matrices : list of np.ndarray
        The finite collection of matrices to cover. All must have the same shape.
    delta : float
        Covering radius (ball diameter is 2*delta).

    Returns
    -------
    centers : list of np.ndarray
        The chosen center matrices forming the delta-covering.
    indices : list of int
        Positions of the chosen centers in the original `matrices` list.
    """
    if not matrices:
        return [], []

    flat = np.array([M.ravel() for M in matrices], dtype=float)
    n = flat.shape[0]

    coverage = _build_coverage(flat, delta)

    uncovered: Set[int] = set(range(n))
    chosen: List[int] = []

    while uncovered:
        best = max(range(n), key=lambda i: len(coverage[i] & uncovered))
        chosen.append(best)
        uncovered -= coverage[best]

    return [matrices[i] for i in chosen], chosen


def verify_delta_covering(
    matrices: List[np.ndarray],
    centers: List[np.ndarray],
    delta: float,
) -> bool:
    """
    Returns True if every matrix in `matrices` is within Frobenius distance
    delta of at least one center.
    """
    for M in matrices:
        if not any(np.linalg.norm((M - C).ravel()) <= delta for C in centers):
            return False
    return True

