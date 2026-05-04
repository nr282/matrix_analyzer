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

def project_matrix_to_circulant(A: np.array[np.array[float]]):
    """
    Projects matrix A to the nearest in 2 norm circulant matrix.

    :param A:
    :return:
    """

    pass

def project_matrix_to_nearest_sparse(A: np.array[np.array[float]]):
    pass


def project_matrix_to_nearest_toeplitz(A: np.array[np.array[float]]):
    pass

def create_covering_for_matricies(matricies: List[np.array[np.array[float]]]):
    """
    The goal is to apply rigorously apply

    """

    pass


