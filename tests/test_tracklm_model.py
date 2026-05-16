import unittest

import torch

from generative_tracking.config import load_config
from generative_tracking.model import TrackLMRS


def _batch(batch_size=2, n_current=4, n_history=3):
    return {
        "window_boxes": torch.zeros(batch_size, 3, max(n_current, 1), 7),
        "window_class_ids": torch.zeros(batch_size, 3, max(n_current, 1), dtype=torch.long),
        "window_track_ids": torch.zeros(batch_size, 3, max(n_current, 1), dtype=torch.long),
        "window_valid_mask": torch.ones(batch_size, 3, max(n_current, 1), dtype=torch.bool),
        "current_boxes": torch.zeros(batch_size, n_current, 7),
        "current_class_ids": torch.zeros(batch_size, n_current, dtype=torch.long),
        "current_track_ids": torch.arange(n_current).repeat(batch_size, 1),
        "current_valid_mask": torch.ones(batch_size, n_current, dtype=torch.bool),
        "history_boxes": torch.zeros(batch_size, n_history, 7),
        "history_class_ids": torch.zeros(batch_size, n_history, dtype=torch.long),
        "history_track_ids": torch.arange(n_history).repeat(batch_size, 1) if n_history else torch.zeros(batch_size, 0, dtype=torch.long),
        "history_valid_mask": torch.ones(batch_size, n_history, dtype=torch.bool),
        "pointer_labels": torch.full((batch_size, n_current), n_history, dtype=torch.long),
    }


class TrackLMModelTest(unittest.TestCase):
    def test_mock_model_forward_shape(self):
        cfg = load_config(
            overrides={
                "dataset": {"max_history_tracks": 3},
                "model": {"visual_dim": 32, "qformer_hidden_size": 32, "mock_llm_hidden_size": 32, "num_queries": 4, "mock_llm_layers": 1},
            }
        )
        model = TrackLMRS(cfg)
        out = model(_batch())
        self.assertEqual(tuple(out["logits"].shape), (2, 4, 4))
        self.assertTrue(bool(torch.isfinite(out["loss"])))


    def test_mock_model_no_history_only_new_logit(self):
        cfg = load_config(
            overrides={
                "dataset": {"max_history_tracks": 0},
                "model": {"visual_dim": 32, "qformer_hidden_size": 32, "mock_llm_hidden_size": 32, "num_queries": 4, "mock_llm_layers": 1},
            }
        )
        model = TrackLMRS(cfg)
        batch = _batch(batch_size=1, n_current=2, n_history=0)
        batch["pointer_labels"] = torch.zeros(1, 2, dtype=torch.long)
        out = model(batch)
        self.assertEqual(tuple(out["logits"].shape), (1, 2, 1))
        self.assertTrue(bool(torch.isfinite(out["loss"])))


if __name__ == "__main__":
    unittest.main()
