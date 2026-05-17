import unittest

import torch

from generative_tracking.config import load_config
from generative_tracking.model import TrackLMRS, hungarian_min_cost_match


def _batch(batch_size=2, n_objects=4):
    return {
        "window_boxes": torch.zeros(batch_size, 3, max(n_objects, 1), 7),
        "window_class_ids": torch.zeros(batch_size, 3, max(n_objects, 1), dtype=torch.long),
        "window_track_ids": torch.zeros(batch_size, 3, max(n_objects, 1), dtype=torch.long),
        "window_valid_mask": torch.ones(batch_size, 3, max(n_objects, 1), dtype=torch.bool),
        "target_boxes": torch.zeros(batch_size, n_objects, 7),
        "target_class_ids": torch.zeros(batch_size, n_objects, dtype=torch.long),
        "target_track_ids": torch.arange(n_objects).repeat(batch_size, 1),
        "target_valid_mask": torch.ones(batch_size, n_objects, dtype=torch.bool),
    }


class TrackLMModelTest(unittest.TestCase):
    def test_mock_model_forward_shape(self):
        cfg = load_config(
            overrides={
                "dataset": {"max_objects": 8},
                "model": {
                    "visual_dim": 32,
                    "qformer_hidden_size": 32,
                    "mock_llm_hidden_size": 32,
                    "num_queries": 4,
                    "num_track_queries": 6,
                    "track_embed_dim": 16,
                    "mock_llm_layers": 1,
                },
            }
        )
        model = TrackLMRS(cfg)
        out = model(_batch())
        self.assertEqual(tuple(out["pred_boxes"].shape), (2, 6, 7))
        self.assertEqual(tuple(out["pred_boxes_normalized"].shape), (2, 6, 7))
        self.assertEqual(tuple(out["pred_logits"].shape), (2, 6, len(cfg.dataset.class_names) + 1))
        self.assertEqual(tuple(out["pred_track_embeds"].shape), (2, 6, 16))
        self.assertTrue(bool(torch.isfinite(out["loss"])))
        self.assertTrue(bool(out["pred_boxes_normalized"][..., :6].ge(0).all()))
        self.assertTrue(bool(out["pred_boxes_normalized"][..., :6].le(1).all()))

    def test_mock_model_no_targets_loss_is_finite(self):
        cfg = load_config(
            overrides={
                "model": {
                    "visual_dim": 32,
                    "qformer_hidden_size": 32,
                    "mock_llm_hidden_size": 32,
                    "num_queries": 4,
                    "num_track_queries": 6,
                    "track_embed_dim": 16,
                    "mock_llm_layers": 1,
                }
            }
        )
        model = TrackLMRS(cfg)
        batch = _batch(batch_size=1, n_objects=0)
        out = model(batch)
        self.assertTrue(bool(torch.isfinite(out["loss"])))
        self.assertEqual(float(out["match_count"]), 0.0)

    def test_hungarian_min_cost_match(self):
        cost = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
        pred_idx, tgt_idx = hungarian_min_cost_match(cost)
        self.assertEqual(pred_idx.tolist(), [0, 1])
        self.assertEqual(tgt_idx.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
