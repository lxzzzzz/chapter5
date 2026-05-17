import json
import pickle
import tempfile
import unittest

import numpy as np

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


class EvaluatorTest(unittest.TestCase):
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
