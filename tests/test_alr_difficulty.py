import unittest

import torch

from modeling.alr_difficulty import DifficultyEstimator


class DifficultyEstimatorTest(unittest.TestCase):
    def test_mean_pool_hidden_states_respects_mask(self):
        hidden_states = torch.tensor(
            [
                [[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]],
                [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
            ]
        )
        attention_mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        pooled = DifficultyEstimator.mean_pool_hidden_states(
            hidden_states,
            attention_mask,
        )
        expected = torch.tensor([[2.0, 3.0], [5.0, 10.0]])
        torch.testing.assert_close(pooled, expected)

    def test_length_label_round_trip(self):
        estimator = DifficultyEstimator(
            hidden_size=4,
            latent_trajectory_lengths=[256, 64, 128],
            mlp_hidden_size=8,
        )
        self.assertEqual(estimator.latent_trajectory_lengths, [64, 128, 256])
        self.assertEqual(estimator.length_to_label(128), 1)
        self.assertEqual(estimator.label_to_length(2), 256)

    def test_invalid_length_raises(self):
        estimator = DifficultyEstimator(
            hidden_size=4,
            latent_trajectory_lengths=[64, 128],
            mlp_hidden_size=8,
        )
        with self.assertRaises(ValueError):
            estimator.length_to_label(256)
        with self.assertRaises(ValueError):
            estimator.label_to_length(3)


if __name__ == "__main__":
    unittest.main()
