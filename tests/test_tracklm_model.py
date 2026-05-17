import unittest

import torch

from generative_tracking.chapter3_features import (
    DETECTOR_BOX_CODE_DIM,
    DETECTOR_POS_ENC_DIM,
    DETECTOR_SCORE_DIM,
    build_detector_tokens,
    extract_detector_frame_tokens,
)
from generative_tracking.config import load_config
from generative_tracking.model import TrackLMRS, generative_track_loss, hungarian_min_cost_match


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
        self.assertEqual(tuple(out["pred_logits"].shape), (2, 6, 2))
        self.assertEqual(tuple(out["pred_track_embeds"].shape), (2, 6, 16))
        self.assertTrue(bool(torch.isfinite(out["loss"])))
        self.assertIn("L_obj", out)
        self.assertNotIn("L_det", out)
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

    def test_detector_token_binds_query_box_and_score(self):
        query = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        boxes = torch.tensor(
            [
                [10.0, 1.0, 0.5, 4.0, 2.0, 1.5, 0.25],
                [20.0, 2.0, 1.0, 5.0, 2.5, 2.0, -0.5],
            ]
        )
        scores = torch.tensor([0.9, 0.2])
        tokens = build_detector_tokens(query, boxes, scores)
        self.assertEqual(
            tuple(tokens.shape),
            (2, query.shape[-1] + DETECTOR_BOX_CODE_DIM + DETECTOR_SCORE_DIM + DETECTOR_POS_ENC_DIM),
        )
        self.assertTrue(torch.equal(tokens[:, : query.shape[-1]], query))
        self.assertTrue(torch.equal(tokens[:, query.shape[-1] + DETECTOR_BOX_CODE_DIM: query.shape[-1] + DETECTOR_BOX_CODE_DIM + 1], scores.unsqueeze(-1)))

    def test_extract_detector_frame_tokens_orders_by_score(self):
        batch_dict = {
            "batch_size": 1,
            "query_voxel_features": torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]),
            "query_boxes": torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]]
            ),
            "query_scores": torch.tensor([[0.2, 0.9, 0.5]]),
        }
        tokens = extract_detector_frame_tokens(batch_dict, max_tokens=2)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].shape[0], 2)
        self.assertEqual(tokens[0][0, 0], 2.0)
        self.assertEqual(tokens[0][1, 0], 3.0)

    def test_objectness_loss_and_metrics_are_finite(self):
        cfg = load_config(
            overrides={
                "model": {
                    "visual_dim": 32,
                    "qformer_hidden_size": 32,
                    "mock_llm_hidden_size": 32,
                    "num_queries": 4,
                    "num_track_queries": 3,
                    "track_embed_dim": 16,
                    "mock_llm_layers": 1,
                }
            }
        )
        outputs = {
            "pred_logits": torch.tensor([[[0.1, 1.2], [1.3, 0.2], [0.0, 0.4]]], dtype=torch.float32),
            "pred_boxes": torch.zeros(1, 3, 7),
            "pred_boxes_normalized": torch.full((1, 3, 7), 0.5),
            "pred_track_embeds": torch.randn(1, 3, 16),
        }
        batch = {
            "target_boxes": torch.zeros(1, 2, 7),
            "target_class_ids": torch.zeros(1, 2, dtype=torch.long),
            "target_track_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "target_valid_mask": torch.tensor([[True, False]]),
        }
        losses, metrics = generative_track_loss(outputs, batch, cfg)
        self.assertIn("L_obj", losses)
        self.assertNotIn("L_cls", losses)
        self.assertTrue(bool(torch.isfinite(losses["L_obj"])))
        self.assertTrue(bool(torch.isfinite(losses["L_center"])))
        self.assertTrue(bool(torch.isfinite(metrics["objectness_acc"])))

    def test_hungarian_min_cost_match(self):
        cost = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
        pred_idx, tgt_idx = hungarian_min_cost_match(cost)
        self.assertEqual(pred_idx.tolist(), [0, 1])
        self.assertEqual(tgt_idx.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
