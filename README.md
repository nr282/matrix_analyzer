## Philosophy

The core ideas, philosophy and goals of this codebase is to help AI researchers better
analyze matricies, and hence exploit their structure.

## Structure Improves Algorithm

With any algorithm (A) aimed at solving a problem (P), the more structure that one can
discover about the problem (P), the better an algorithm (A) can become.

## Problem

Currently, backpropogation relies on the solution of the multiplication of two matricies
A and B to find solution S, concretely, S = A * B.

## GEMM is not Fine Tuned to Backpropogation

GEMM (General Matrix Multiplication) algorithm is the algorithm chosen for this solution.

GEMM is not fine-tuned toward matricies that are found in backpropogation, but it is a general
tool aimed at solving a general mathematical problem, without taking into account the structure
of backpropogation. Structure in this case could be the low-rank nature of the A and B matricies.
Likewise, structure could mean being bounded, or concentrated in a probabilistic sense
on some lower dimensional manifold. Alternatively, the space of backpropogation matricies can be
covered with a delta covering.

To stress again, GEMM is not fine-tuned for deep learning. It is fine-tuned to any matrix-matrix multiplication.

## Investigate Backpropogation Matricies

In order to fine tune, the multiplication of A and B, we must first investigate the structure
of A and B.
