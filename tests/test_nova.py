import pickle
import tempfile
import unittest

import numpy as np
import torch

from generative_tracking.config import load_config
from generative_tracking.nova_data import (
    DetectionFrame,
    NOVAAssociationDataset,
    filter_detection_frame,
    nova_collate,
    validate_detection_frame,
)
from generative_tracking.nova_model import NOVAAssociationModel
from generative_tracking.nova_runtime import NOVAOnlineTracker, associate_by_scores


def _box(x, y=0.0):
    return np.asarray([x, y, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)


def _info(frame_idx, tracks):
    return {
        "sequence_id": "seq0",
        "frame_id": f"{frame_idx:06d}",
        "frame_idx": frame_idx,
        "annos": {
            "name": np.asarray(["Car"] * len(tracks)),
            "track_id": np.asarray([track_id for track_id, _box_value in tracks], dtype=np.int64),
            "gt_boxes_lidar": np.stack([box for _track_id, box in tracks]).astype(np.float32) if tracks else np.zeros((0, 7), dtype=np.float32),
            "score": np.ones((len(tracks),), dtype=np.float32),
        },
    }


def _cfg(info_path):
    return load_config(
        overrides={
            "device": "cpu",
            "dataset": {
                "name": "tmp",
                "info_paths": {"tmp": {"train": info_path, "val": info_path}},
                "class_names": ["Car"],
                "box_center_range": [-20.0, -20.0, -5.0, 40.0, 20.0, 5.0],
            },
            "nova": {"history_len": 2, "det_gt_iou_threshold": 0.5, "association_threshold": 0.5},
            "model": {
                "use_mock_llm": True,
                "mock_llm_hidden_size": 32,
                "mock_llm_layers": 1,
                "num_attention_heads": 4,
            },
        }
    )


class NOVATest(unittest.TestCase):
    def test_detection_cache_schema_and_car_filter(self):
        frame = {
            "sequence_id": "seq0",
            "frame_id": "000001",
            "frame_idx": 1,
            "pred_boxes": np.stack([_box(0.0), _box(1.0), _box(2.0)]),
            "pred_scores": np.asarray([0.9, 0.01, 0.8], dtype=np.float32),
            "pred_labels": np.asarray([0, 0, 1], dtype=np.int64),
        }
        filtered = filter_detection_frame(frame, class_id=0, score_thresh=0.05, max_dets=10)
        det = validate_detection_frame(filtered)
        self.assertEqual(det.sequence_id, "seq0")
        self.assertEqual(tuple(det.pred_boxes.shape), (1, 7))
        self.assertEqual(det.pred_scores.dtype, np.float32)
        self.assertEqual(det.pred_labels.tolist(), [0])

    def test_pair_label_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = f"{tmpdir}/tracking_infos_train.pkl"
            infos = [
                _info(0, [(1, _box(0.0))]),
                _info(1, [(1, _box(1.0)), (2, _box(10.0))]),
            ]
            with open(info_path, "wb") as f:
                pickle.dump(infos, f)
            cfg = _cfg(info_path)
            detections = {
                ("seq0", "000000"): DetectionFrame("seq0", "000000", 0, np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
                ("seq0", "000001"): DetectionFrame(
                    "seq0",
                    "000001",
                    1,
                    np.stack([_box(1.0), _box(10.0)]).astype(np.float32),
                    np.asarray([0.9, 0.8], dtype=np.float32),
                    np.asarray([0, 0], dtype=np.int64),
                ),
            }
            dataset = NOVAAssociationDataset(cfg, split="train", detection_cache=detections)
            labels_by_det = {dataset[idx]["det_idx"]: int(dataset[idx]["match_label"]) for idx in range(len(dataset))}
            self.assertEqual(labels_by_det[0], 1)
            self.assertEqual(labels_by_det[1], 0)
            batch = nova_collate([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["track_history_boxes"].shape), (2, 2, 7))
            self.assertIn("Task: 3D Association", batch["prompt_texts"][0])
            self.assertIn("Question: Is this the same object?", batch["prompt_texts"][0])
            self.assertIn("Frame -2: Observation: Missing", batch["prompt_texts"][0])
            self.assertNotIn("Frame -2: Observation: Missing, Box:", batch["prompt_texts"][0])
            self.assertEqual(batch["prompt_texts"][0].count("<box>"), 2)
            self.assertEqual(tuple(batch["box_token_mask"].shape), (2, 3))
            self.assertEqual(batch["box_token_mask"][0].tolist(), [False, True, True])

    def test_model_forward_shapes_and_loss(self):
        cfg = load_config(
            overrides={
                "device": "cpu",
                "dataset": {"box_center_range": [-20.0, -20.0, -5.0, 40.0, 20.0, 5.0]},
                "nova": {"history_len": 3, "geometry_hidden_size": 32},
                "model": {
                    "use_mock_llm": True,
                    "mock_llm_hidden_size": 32,
                    "mock_llm_layers": 1,
                    "num_attention_heads": 4,
                },
            }
        )
        model = NOVAAssociationModel(cfg)
        batch = {
            "track_history_boxes": torch.zeros(4, 3, 7),
            "track_history_scores": torch.ones(4, 3),
            "track_history_class_ids": torch.zeros(4, 3, dtype=torch.long),
            "track_history_mask": torch.ones(4, 3, dtype=torch.bool),
            "candidate_box": torch.zeros(4, 7),
            "candidate_score": torch.ones(4),
            "candidate_class_id": torch.zeros(4, dtype=torch.long),
            "match_label": torch.tensor([1, 0, 0, 1], dtype=torch.long),
            "target_iou": torch.tensor([0.9, 0.0, 0.0, 0.8]),
            "quality_valid": torch.tensor([True, False, False, True]),
        }
        out = model(batch)
        self.assertEqual(tuple(out["match_logits"].shape), (4, 2))
        self.assertEqual(tuple(out["quality"].shape), (4,))
        self.assertTrue(bool(torch.isfinite(out["loss"])))

    def test_hungarian_and_tracker_lifecycle(self):
        scores = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        matches, unmatched_tracks, unmatched_dets = associate_by_scores(scores, threshold=0.5)
        self.assertEqual([(t, d) for t, d, _s in matches], [(0, 0), (1, 1)])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_dets, [])

        cfg = load_config(
            overrides={
                "dataset": {"class_names": ["Car"]},
                "nova": {"history_len": 2, "association_threshold": 0.5, "max_lost_frames": 1},
            }
        )
        tracker = NOVAOnlineTracker(cfg, torch.device("cpu"))
        frame0 = DetectionFrame(
            "seq0",
            "000000",
            0,
            np.stack([_box(0.0), _box(10.0)]),
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
        )
        out0 = tracker.update(frame0, _DummyScores([]))
        self.assertEqual([track["id"] for track in out0["tracks"]], [0, 1])
        frame1 = DetectionFrame(
            "seq0",
            "000001",
            1,
            np.stack([_box(1.0), _box(11.0)]),
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
        )
        out1 = tracker.update(frame1, _DummyScores([0.9, 0.1, 0.2, 0.8]))
        self.assertEqual([track["id"] for track in out1["tracks"]], [0, 1])
        frame2 = DetectionFrame("seq0", "000002", 2, np.stack([_box(2.0)]), np.asarray([1.0], dtype=np.float32), np.asarray([0], dtype=np.int64))
        out2 = tracker.update(frame2, _DummyScores([0.9, 0.1]))
        self.assertEqual([track["id"] for track in out2["tracks"]], [0])
        self.assertEqual(tracker.tracks[1].lost_frames, 1)


class _DummyScores(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.as_tensor(scores, dtype=torch.float32)

    def forward(self, batch):
        return {"match_prob": self.scores[: batch["candidate_box"].shape[0]]}


if __name__ == "__main__":
    unittest.main()
