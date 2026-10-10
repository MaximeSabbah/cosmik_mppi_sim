import unittest
import numpy as np
from capsule_prediction import Observation, predict


class PredictionTests(unittest.TestCase):
    def obs(self, stamp=0, ids=('arm',), ends=None, frame='world'):
        if ends is None:
            ends = np.zeros((len(ids), 2, 3))
        return Observation(stamp, frame, ids, ends, np.full(len(ids), 0.08))

    def test_hold_is_stationary_at_all_rollout_times(self):
        current = self.obs(1, ends=[[[0, 0, 0], [0, 0, 1]]])
        result = predict(current, np.linspace(1, 2, 50))
        self.assertEqual(result.endpoints.shape, (50, 1, 2, 3))
        np.testing.assert_equal(result.endpoints, np.repeat(current.endpoints[None], 50, axis=0))

    def test_cv_preserves_geometry_and_matches_reordered_ids(self):
        old = self.obs(0, ('a', 'b'), np.array([[[0, 0, 0], [0, 0, 1]], [[5, 0, 0], [5, 0, 1]]]))
        current = self.obs(1, ('b', 'a'), np.array([[[3, 0, 0], [3, 0, 1]], [[1, 0, 0], [1, 0, 1]]]))
        result = predict(current, [1, 2], old)
        np.testing.assert_allclose(result.endpoints[1, :, 0, 0], [1, 2])
        np.testing.assert_allclose(result.endpoints[:, :, 1] - result.endpoints[:, :, 0], [[[0, 0, 1]] * 2] * 2)

    def test_new_id_holds(self):
        result = predict(self.obs(1, ('new',)), [2], self.obs(0, ('old',)))
        np.testing.assert_equal(result.endpoints, 0)

    def test_bad_history_or_times_rejected(self):
        current = self.obs(1)
        for times, old in [([0], None), ([2, 1], None), ([2], self.obs(1)), ([2], self.obs(0, frame='camera'))]:
            with self.assertRaises(ValueError):
                predict(current, times, old)

    def test_nonfinite_and_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            self.obs(ids=('a', 'a'))
        with self.assertRaises(ValueError):
            self.obs(stamp=float('nan'))


if __name__ == '__main__':
    unittest.main()
