import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generative_tracking.ab3dmot_evaluator import evaluate_ab3dmot_json


def _box(x):
    return [float(x), 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]


class AB3DMOTEvaluatorTest(unittest.TestCase):
    def test_ab3dmot_metrics_are_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            infos = []
            results = []
            for frame_idx in range(2):
                frame_id = f"{frame_idx:06d}"
                infos.append(
                    {
                        "sequence_id": "seq0",
                        "frame_id": frame_id,
                        "frame_idx": frame_idx,
                        "annos": {
                            "name": np.asarray(["Car"]),
                            "track_id": np.asarray([7], dtype=np.int64),
                            "gt_boxes_lidar": np.asarray([_box(frame_idx)], dtype=np.float32),
                            "score": np.asarray([1.0], dtype=np.float32),
                        },
                    }
                )
                results.append(
                    {
                        "sequence_id": "seq0",
                        "frame_id": frame_id,
                        "tracks": [
                            {
                                "id": 3,
                                "class": "Car",
                                "box3d": _box(frame_idx),
                                "score": 0.9 - frame_idx * 0.1,
                            }
                        ],
                    }
                )
            info_path = root / "infos.pkl"
            result_path = root / "results.json"
            output_path = root / "ab3dmot.json"
            with info_path.open("wb") as f:
                pickle.dump(infos, f)
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(results, f)
            metrics = evaluate_ab3dmot_json(
                result_path,
                info_path,
                class_names=["Car"],
                iou_threshold=0.5,
                recall_points=4,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())
            self.assertIn("sAMOTA", metrics)
            self.assertIn("AMOTA", metrics)
            self.assertIn("AMOTP", metrics)
            self.assertEqual(float(metrics["MOTA"]), 1.0)
            self.assertEqual(float(metrics["IDS"]), 0.0)
            self.assertEqual(len(metrics["curve"]), 4)

    def test_ab3dmot_bev_range_filters_gt_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            infos = [
                {
                    "sequence_id": "seq0",
                    "frame_id": "000000",
                    "frame_idx": 0,
                    "annos": {
                        "name": np.asarray(["Car", "Car"]),
                        "track_id": np.asarray([1, 2], dtype=np.int64),
                        "gt_boxes_lidar": np.asarray([_box(0.0), _box(100.0)], dtype=np.float32),
                        "score": np.asarray([1.0, 1.0], dtype=np.float32),
                    },
                }
            ]
            results = [
                {
                    "sequence_id": "seq0",
                    "frame_id": "000000",
                    "tracks": [
                        {"id": 1, "class": "Car", "box3d": _box(0.0), "score": 1.0},
                        {"id": 2, "class": "Car", "box3d": _box(100.0), "score": 1.0},
                    ],
                }
            ]
            info_path = root / "infos.pkl"
            result_path = root / "results.json"
            with info_path.open("wb") as f:
                pickle.dump(infos, f)
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(results, f)

            metrics = evaluate_ab3dmot_json(
                result_path,
                info_path,
                class_names=["Car"],
                iou_threshold=0.5,
                recall_points=4,
                bev_range=[-1.0, -1.0, 1.0, 1.0],
            )

            self.assertEqual(float(metrics["num_gt"]), 1.0)
            self.assertEqual(float(metrics["num_pred"]), 1.0)
            self.assertEqual(float(metrics["MOTA"]), 1.0)


if __name__ == "__main__":
    unittest.main()
