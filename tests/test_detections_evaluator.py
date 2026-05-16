import json
import pickle
import tempfile
import unittest

import numpy as np

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset
from generative_tracking.evaluator import evaluate_tracking_json


def _info(frame_idx, track_ids):
    track_ids = np.asarray(track_ids, dtype=np.int64)
    return {
        "sequence_id": "seq0",
        "frame_id": f"{frame_idx:06d}",
        "frame_idx": frame_idx,
        "annos": {
            "name": np.asarray(["Car"] * len(track_ids)),
            "track_id": track_ids,
            "gt_boxes_lidar": np.asarray([[float(i) * 10.0, 0, 0, 4, 2, 1.5, 0] for i in range(len(track_ids))], dtype=np.float32),
            "score": np.ones((len(track_ids),), dtype=np.float32),
        },
    }


class DetectionEvaluatorTest(unittest.TestCase):
    def test_detection_source_assigns_pointer_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = f"{tmpdir}/infos.pkl"
            det_path = f"{tmpdir}/dets.pkl"
            infos = [_info(0, [5]), _info(1, [5, 8])]
            with open(info_path, "wb") as f:
                pickle.dump(infos, f)
            detections = {
                "frames": [
                    {"sequence_id": "seq0", "frame_id": "000000", "boxes_lidar": infos[0]["annos"]["gt_boxes_lidar"], "name": ["Car"], "score": [0.9]},
                    {"sequence_id": "seq0", "frame_id": "000001", "boxes_lidar": infos[1]["annos"]["gt_boxes_lidar"], "name": ["Car", "Car"], "score": [0.9, 0.8]},
                ]
            }
            with open(det_path, "wb") as f:
                pickle.dump(detections, f)
            cfg = load_config(
                overrides={
                    "dataset": {
                        "name": "tmp",
                        "K": 2,
                        "object_source": "detections",
                        "info_paths": {"tmp": {"train": info_path, "val": info_path}},
                        "detection_paths": {"train": det_path, "val": det_path},
                        "max_history_tracks": 4,
                        "max_current_objects": 8,
                    }
                }
            )
            sample = SequenceWindowDataset(cfg, split="train")[1]
            self.assertEqual(sample["history_track_ids"].tolist(), [5])
            self.assertEqual(sample["pointer_labels"].tolist(), [0, 4])

    def test_builtin_evaluator_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = f"{tmpdir}/infos.pkl"
            result_path = f"{tmpdir}/result.json"
            infos = [_info(0, [1])]
            with open(info_path, "wb") as f:
                pickle.dump(infos, f)
            result = [{"sequence_id": "seq0", "frame_id": "000000", "tracks": [{"id": 7, "class": "Car", "box3d": infos[0]["annos"]["gt_boxes_lidar"][0].tolist(), "score": 1.0}]}]
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
            metrics = evaluate_tracking_json(result_path, info_path, ["Car"], iou_threshold=0.5)
            self.assertEqual(metrics["matches"], 1.0)
            self.assertEqual(metrics["false_positive"], 0.0)


if __name__ == "__main__":
    unittest.main()
