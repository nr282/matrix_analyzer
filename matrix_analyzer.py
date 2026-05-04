"""
Module aims to provide tools to analyze the matricies that are found in backpropgation.

The core ideas, philosophy and goals of this codebase is to help AI researchers better
analyze matricies, and hence exploit their structure.

With any algorithm (A) aimed at solving a problem (P), the more structure that one can
discover about the problem (P), the better an algorithm (A) can become.

Currently, backpropogation relies on the solution of the multiplication of two matricies
A and B to find solution S, concretely, S = A * B.

GEMM (General Matrix Multiplication) algorithm is the algorithm chosen for this solution.

GEMM is not fine-tuned toward matricies that are found in backpropogation, but it is a general
tool aimed at solving a general mathematical problem, without taking into account the structure
of backpropogation. Structure in this case could be the low-rank nature of the A and B matricies.
Likewise, structure could mean being bounded, or concentrated in a probabilistic sense
on some lower dimensional manifold. Alternatively, the space of backpropogation matricies can be
covered with a delta covering.

To stress again, GEMM is not fine-tuned for deep learning. It is fine-tuned to any matrix-matrix multiplication.

In order to fine tune, the multiplication of A and B, we must first investigate the structure
of A and B.

This is the goal of this repository.

####################################################################################################

One might ask what structure will this repository investigate?
    1. Given a matrix A, what is the closest norm circulant, toeplitz, or s-sparse matrix.
    2. What is the metric entropy of the space, X = M(NxN), with a metric p.
    3. What is a covering of a space of X = M(NxN) matricies.
    4. The application of (1),(2),(3) to backpropogation matricies.

"""

import numpy as np
from typing import List, Optional, Tuple, Set


def project_matrix_to_circulant(A: np.ndarray) -> np.ndarray:
    """
    Projects square matrix A to the nearest circulant matrix in the Frobenius norm.

    A circulant matrix C satisfies C[i,j] = c[(j-i) % n].  The optimal c is
    obtained by averaging each wrapped diagonal of A:
        c[k] = mean{ A[i, (i+k) % n]  for i = 0..n-1 }
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    n = A.shape[0]
    i = np.arange(n)
    c = np.array([A[i, (i + k) % n].mean() for k in range(n)])
    row, col = np.ogrid[:n, :n]
    return c[(col - row) % n]


def project_matrix_to_nearest_sparse(A: np.ndarray, s: int) -> np.ndarray:
    """
    Projects A to the nearest s-sparse matrix in the Frobenius norm.

    Keeps the s entries of largest absolute value and zeros the rest.
    """
    if s <= 0:
        return np.zeros_like(A)
    if s >= A.size:
        return A.copy()
    B = np.zeros_like(A)
    top_s = np.argpartition(np.abs(A).ravel(), -s)[-s:]
    B.ravel()[top_s] = A.ravel()[top_s]
    return B


def project_matrix_to_nearest_toeplitz(A: np.ndarray) -> np.ndarray:
    """
    Projects A to the nearest Toeplitz matrix in the Frobenius norm.

    A Toeplitz matrix T satisfies T[i,j] = t[j-i].  The optimal t[d] is the
    mean of A[i,j] over all (i,j) on diagonal d (where d = j - i).
    """
    m, n = A.shape
    T = np.zeros_like(A)
    for d in range(-(m - 1), n):
        rows = np.arange(max(0, -d), min(m, n - d))
        cols = rows + d
        avg = A[rows, cols].mean()
        T[rows, cols] = avg
    return T


def create_covering_for_matricies(
    matricies: List[np.ndarray],
    delta: float = 1.0,
) -> List[np.ndarray]:
    """
    Returns a delta-covering of the given matrices under the Frobenius norm.

    Uses a greedy algorithm: repeatedly pick an uncovered matrix as a new
    center and remove all matrices within Frobenius distance delta of it.
    """
    centers: List[np.ndarray] = []
    uncovered = list(matricies)
    while uncovered:
        center = uncovered[0]
        centers.append(center)
        uncovered = [
            M for M in uncovered[1:]
            if np.linalg.norm(M - center, "fro") > delta
        ]
    return centers


