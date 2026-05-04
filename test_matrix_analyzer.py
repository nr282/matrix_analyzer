import unittest
import numpy as np

from deep_learning_implementation import Linear, ReLU, Sequential
from matrix_analyzer import (
    project_matrix_to_circulant,
    project_matrix_to_nearest_sparse,
    project_matrix_to_nearest_toeplitz,
    create_covering_for_matricies,
)
from greedy_set_cover import greedy_set_cover_delta_covering, verify_delta_covering


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


class TestGreedySetCover(unittest.TestCase):

    RNG = np.random.default_rng(42)

    def _cluster(self, center, n, noise):
        return [center + self.RNG.normal(0, noise, center.shape) for _ in range(n)]

    def _three_cluster_matrices(self):
        c1 = np.zeros((3, 3))
        c2 = np.eye(3) * 5
        c3 = np.ones((3, 3)) * 10
        return self._cluster(c1, 6, 0.3) + self._cluster(c2, 6, 0.3) + self._cluster(c3, 6, 0.3)

    def test_empty_input(self):
        centers, indices = greedy_set_cover_delta_covering([], delta=1.0)
        self.assertEqual(centers, [])
        self.assertEqual(indices, [])

    def test_single_matrix(self):
        M = np.eye(3)
        centers, indices = greedy_set_cover_delta_covering([M], delta=1.0)
        self.assertEqual(len(centers), 1)
        self.assertEqual(indices, [0])

    def test_identical_matrices_need_one_center(self):
        mats = [np.eye(4).copy() for _ in range(5)]
        centers, _ = greedy_set_cover_delta_covering(mats, delta=0.1)
        self.assertEqual(len(centers), 1)

    def test_covering_is_valid_for_all_deltas(self):
        mats = self._three_cluster_matrices()
        for delta in [0.5, 1.0, 2.0, 5.0]:
            centers, _ = greedy_set_cover_delta_covering(mats, delta)
            self.assertTrue(
                verify_delta_covering(mats, centers, delta),
                f"Covering invalid at delta={delta}",
            )

    def test_three_clusters_recovered_at_delta_2(self):
        mats = self._three_cluster_matrices()
        centers, _ = greedy_set_cover_delta_covering(mats, delta=2.0)
        self.assertEqual(len(centers), 3)

    def test_tight_delta_needs_all_centers(self):
        mats = [np.eye(3) * k for k in range(5)]
        centers, _ = greedy_set_cover_delta_covering(mats, delta=0.1)
        self.assertEqual(len(centers), len(mats))

    def test_indices_are_valid_and_unique(self):
        mats = self._three_cluster_matrices()
        _, indices = greedy_set_cover_delta_covering(mats, delta=2.0)
        n = len(mats)
        self.assertTrue(all(0 <= i < n for i in indices))
        self.assertEqual(len(set(indices)), len(indices))

    def test_centers_match_indexed_matrices(self):
        mats = self._three_cluster_matrices()
        centers, indices = greedy_set_cover_delta_covering(mats, delta=2.0)
        for center, idx in zip(centers, indices):
            np.testing.assert_array_equal(center, mats[idx])

    def test_larger_delta_fewer_or_equal_centers(self):
        mats = self._three_cluster_matrices()
        small, _ = greedy_set_cover_delta_covering(mats, delta=1.0)
        large, _ = greedy_set_cover_delta_covering(mats, delta=2.0)
        self.assertLessEqual(len(large), len(small))

    def test_verify_detects_invalid_cover(self):
        mats = [np.eye(3) * k for k in range(3)]
        bad_centers = [np.eye(3) * 100]
        self.assertFalse(verify_delta_covering(mats, bad_centers, delta=1.0))

    def test_non_square_matrices(self):
        mats = [self.RNG.standard_normal((2, 5)) for _ in range(10)]
        centers, _ = greedy_set_cover_delta_covering(mats, delta=3.0)
        self.assertTrue(verify_delta_covering(mats, centers, delta=3.0))


class TestSetCoverOnWeightMatrices(unittest.TestCase):
    """
    Tests for applying greedy set cover to the columns of neural-network
    weight matrices. Each column is the weight vector for one output neuron;
    the covering size measures how many distinct prototype directions are
    needed to represent all neurons at a given radius.
    """

    def _make_model(self):
        return Sequential([
            Linear(10, 16, seed=0),
            ReLU(),
            Linear(16, 8, seed=1),
            ReLU(),
            Linear(8, 2, seed=2),
        ])

    @staticmethod
    def _columns(W: np.ndarray) -> list:
        return [W[:, j] for j in range(W.shape[1])]

    def test_covering_valid_for_every_layer_and_delta(self):
        model = self._make_model()
        for W in model.weight_matrices():
            cols = self._columns(W)
            for delta in np.arange(0.5, 6.0, 0.5):
                centers, _ = greedy_set_cover_delta_covering(cols, delta)
                self.assertTrue(
                    verify_delta_covering(cols, centers, delta),
                    f"Invalid covering: W.shape={W.shape}, delta={delta:.1f}",
                )

    def test_center_count_bounded_by_column_count(self):
        model = self._make_model()
        for W in model.weight_matrices():
            cols = self._columns(W)
            for delta in [0.5, 2.0, 5.0]:
                centers, _ = greedy_set_cover_delta_covering(cols, delta)
                self.assertLessEqual(len(centers), W.shape[1])

    def test_monotone_in_delta(self):
        model = self._make_model()
        for W in model.weight_matrices():
            cols = self._columns(W)
            deltas = np.arange(0.5, 6.0, 0.5)
            counts = [
                len(greedy_set_cover_delta_covering(cols, d)[0])
                for d in deltas
            ]
            for a, b in zip(counts, counts[1:]):
                self.assertLessEqual(b, a,
                    f"Center count increased as delta grew: W.shape={W.shape}")

    def test_large_delta_collapses_to_one_center(self):
        # At a sufficiently large radius every column is within one ball
        model = self._make_model()
        for W in model.weight_matrices():
            cols = self._columns(W)
            max_norm = max(np.linalg.norm(c) for c in cols)
            centers, _ = greedy_set_cover_delta_covering(cols, delta=max_norm * 3)
            self.assertEqual(len(centers), 1,
                f"Expected 1 center at huge delta, got {len(centers)} for W.shape={W.shape}")

    def test_zero_noise_identical_columns_need_one_center(self):
        # A weight matrix whose columns are all identical compresses to 1 center
        W = np.tile(np.array([1.0, 2.0, 3.0, 4.0])[:, None], (1, 12))
        cols = self._columns(W)
        centers, _ = greedy_set_cover_delta_covering(cols, delta=0.01)
        self.assertEqual(len(centers), 1)

    def test_two_cluster_columns_recovered(self):
        # Columns split into two tight clusters well-separated in Frobenius distance
        rng = np.random.default_rng(7)
        cluster_a = [rng.normal(0, 0.05, 8) for _ in range(10)]
        cluster_b = [rng.normal(10, 0.05, 8) for _ in range(10)]
        cols = cluster_a + cluster_b
        centers, _ = greedy_set_cover_delta_covering(cols, delta=0.5)
        self.assertEqual(len(centers), 2)

    def test_column_indices_reference_original_list(self):
        model = self._make_model()
        W = model.weight_matrices()[0]
        cols = self._columns(W)
        centers, indices = greedy_set_cover_delta_covering(cols, delta=2.0)
        for center, idx in zip(centers, indices):
            np.testing.assert_array_equal(center, cols[idx])

    def test_all_24_deltas_produce_valid_coverings(self):
        model = self._make_model()
        W = model.weight_matrices()[0]
        cols = self._columns(W)
        for delta in np.arange(0.5, 12.5, 0.5):
            centers, _ = greedy_set_cover_delta_covering(cols, delta)
            self.assertTrue(
                verify_delta_covering(cols, centers, delta),
                f"Invalid covering at delta={delta:.1f}",
            )


if __name__ == "__main__":
    unittest.main()
