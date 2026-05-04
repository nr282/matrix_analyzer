import unittest
import numpy as np

from matrix_analyzer import (
    project_matrix_to_circulant,
    project_matrix_to_nearest_sparse,
    project_matrix_to_nearest_toeplitz,
    create_covering_for_matricies,
)


class TestProjectToCirculant(unittest.TestCase):

    def test_output_is_circulant(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 5))
        C = project_matrix_to_circulant(A)
        n = C.shape[0]
        i = np.arange(n)
        for k in range(n):
            vals = C[i, (i + k) % n]
            self.assertTrue(np.allclose(vals, vals[0]),
                            f"Wrapped diagonal {k} is not constant")

    def test_circulant_input_unchanged(self):
        # A circulant matrix should project to itself
        c = np.array([1.0, 2.0, 3.0, 4.0])
        n = len(c)
        row, col = np.ogrid[:n, :n]
        A = c[(col - row) % n]
        C = project_matrix_to_circulant(A)
        np.testing.assert_allclose(C, A)

    def test_frobenius_optimality(self):
        # The projection must be closer (or equal) to A than any other circulant
        rng = np.random.default_rng(1)
        A = rng.standard_normal((4, 4))
        C = project_matrix_to_circulant(A)
        proj_dist = np.linalg.norm(A - C, "fro")

        for _ in range(50):
            c_rand = rng.standard_normal(4)
            row, col = np.ogrid[:4, :4]
            C_rand = c_rand[(col - row) % 4]
            self.assertLessEqual(proj_dist, np.linalg.norm(A - C_rand, "fro") + 1e-10)

    def test_shape_preserved(self):
        A = np.eye(6)
        C = project_matrix_to_circulant(A)
        self.assertEqual(C.shape, A.shape)

    def test_non_square_raises(self):
        A = np.ones((3, 4))
        with self.assertRaises(AssertionError):
            project_matrix_to_circulant(A)

    def test_1x1_matrix(self):
        A = np.array([[7.0]])
        C = project_matrix_to_circulant(A)
        np.testing.assert_allclose(C, A)


class TestProjectToSparse(unittest.TestCase):

    def test_exact_sparsity(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        S = project_matrix_to_nearest_sparse(A, s=2)
        self.assertEqual(np.count_nonzero(S), 2)

    def test_largest_entries_kept(self):
        A = np.array([[1.0, 5.0], [3.0, 2.0]])
        S = project_matrix_to_nearest_sparse(A, s=2)
        # 5.0 and 3.0 are the two largest
        self.assertAlmostEqual(S[0, 1], 5.0)
        self.assertAlmostEqual(S[1, 0], 3.0)
        self.assertAlmostEqual(S[0, 0], 0.0)
        self.assertAlmostEqual(S[1, 1], 0.0)

    def test_s_equals_size_returns_copy(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        S = project_matrix_to_nearest_sparse(A, s=4)
        np.testing.assert_array_equal(S, A)

    def test_s_exceeds_size_returns_copy(self):
        A = np.ones((3, 3))
        S = project_matrix_to_nearest_sparse(A, s=100)
        np.testing.assert_array_equal(S, A)

    def test_negative_entries(self):
        A = np.array([[-10.0, 1.0], [2.0, 0.5]])
        S = project_matrix_to_nearest_sparse(A, s=1)
        # -10 has the largest absolute value
        self.assertAlmostEqual(S[0, 0], -10.0)
        self.assertEqual(np.count_nonzero(S), 1)

    def test_frobenius_optimality(self):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((4, 4))
        s = 5
        S = project_matrix_to_nearest_sparse(A, s=s)
        proj_dist = np.linalg.norm(A - S, "fro")

        # Any other s-sparse matrix formed by zeroing different entries
        # must have Frobenius distance >= proj_dist
        flat = A.ravel()
        indices = np.argsort(np.abs(flat))  # ascending
        # Force a different s-sparse pattern (keep the s smallest instead)
        alt = np.zeros_like(A)
        alt.ravel()[indices[:s]] = flat[indices[:s]]
        self.assertLessEqual(proj_dist, np.linalg.norm(A - alt, "fro") + 1e-10)

class TestProjectToToeplitz(unittest.TestCase):

    def test_output_is_toeplitz(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((5, 5))
        T = project_matrix_to_nearest_toeplitz(A)
        n = T.shape[0]
        for d in range(-(n - 1), n):
            diag = np.diag(T, d)
            self.assertTrue(np.allclose(diag, diag[0]),
                            f"Diagonal {d} is not constant")

    def test_toeplitz_input_unchanged(self):
        # A Toeplitz matrix should project to itself
        A = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 1.0, 2.0],
            [5.0, 4.0, 1.0],
        ])
        T = project_matrix_to_nearest_toeplitz(A)
        np.testing.assert_allclose(T, A)

    def test_frobenius_optimality(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((4, 4))
        T = project_matrix_to_nearest_toeplitz(A)
        proj_dist = np.linalg.norm(A - T, "fro")

        # Perturb the Toeplitz projection slightly and verify it's worse
        for _ in range(30):
            noise = rng.standard_normal(T.shape) * 0.01
            n = T.shape[0]
            for d in range(-(n - 1), n):
                rows = np.arange(max(0, -d), min(n, n - d))
                noise[rows, rows + d] = noise[rows[0], rows[0] + d]
            T_perturbed = T + noise
            self.assertLessEqual(proj_dist,
                                 np.linalg.norm(A - T_perturbed, "fro") + 1e-8)

    def test_shape_preserved(self):
        A = np.ones((4, 4))
        T = project_matrix_to_nearest_toeplitz(A)
        self.assertEqual(T.shape, A.shape)

    def test_non_square_matrix(self):
        A = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        T = project_matrix_to_nearest_toeplitz(A)
        self.assertEqual(T.shape, A.shape)
        m, n = T.shape
        for d in range(-(m - 1), n):
            diag = np.diag(T, d)
            self.assertTrue(np.allclose(diag, diag[0]))

    def test_1x1_matrix(self):
        A = np.array([[5.0]])
        T = project_matrix_to_nearest_toeplitz(A)
        np.testing.assert_allclose(T, A)


class TestCreateCovering(unittest.TestCase):

    def test_every_matrix_is_covered(self):
        rng = np.random.default_rng(5)
        matrices = [rng.standard_normal((3, 3)) for _ in range(30)]
        delta = 2.0
        centers = create_covering_for_matricies(matrices, delta=delta)
        for M in matrices:
            dists = [np.linalg.norm(M - c, "fro") for c in centers]
            self.assertLessEqual(min(dists), delta + 1e-10,
                                 "A matrix is not covered by any center")

    def test_centers_are_subset_of_input(self):
        rng = np.random.default_rng(6)
        matrices = [rng.standard_normal((2, 2)) for _ in range(10)]
        centers = create_covering_for_matricies(matrices, delta=1.0)
        for c in centers:
            self.assertTrue(any(np.array_equal(c, M) for M in matrices))

    def test_single_matrix(self):
        A = np.eye(3)
        centers = create_covering_for_matricies([A], delta=1.0)
        self.assertEqual(len(centers), 1)
        np.testing.assert_array_equal(centers[0], A)

    def test_large_delta_gives_one_center(self):
        rng = np.random.default_rng(7)
        # Cluster all matrices near the origin so one center covers all
        matrices = [rng.standard_normal((2, 2)) * 0.01 for _ in range(20)]
        centers = create_covering_for_matricies(matrices, delta=10.0)
        self.assertEqual(len(centers), 1)

    def test_zero_delta_each_matrix_is_own_center(self):
        # With delta=0 every distinct matrix must be its own center
        matrices = [np.eye(2) * k for k in range(1, 6)]
        centers = create_covering_for_matricies(matrices, delta=0.0)
        self.assertEqual(len(centers), len(matrices))

    def test_empty_input(self):
        centers = create_covering_for_matricies([], delta=1.0)
        self.assertEqual(centers, [])


if __name__ == "__main__":
    unittest.main()
