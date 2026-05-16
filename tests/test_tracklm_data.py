import pickle
import tempfile
import unittest

import numpy as np

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset, build_history_candidates, build_pointer_labels, frame_to_objects


def _info(frame_idx, track_ids):
    track_ids = np.asarray(track_ids, dtype=np.int64)
    return {
        "sequence_id": "seq0",
        "frame_id": f"{frame_idx:06d}",
        "frame_idx": frame_idx,
        "annos": {
            "name": np.asarray(["Car"] * len(track_ids)),
            "track_id": track_ids,
            "gt_boxes_lidar": np.zeros((len(track_ids), 7), dtype=np.float32),
            "score": np.ones((len(track_ids),), dtype=np.float32),
        },
    }


class TrackLMDataTest(unittest.TestCase):
    def test_pointer_labels_recent_history_order(self):
        class_to_id = {"Car": 0}
        history = [frame_to_objects(_info(0, [5, 8, 12]), class_to_id)]
        candidates = build_history_candidates(history, max_history_tracks=3)
        labels = build_pointer_labels(np.asarray([5, 8, 20, 12]), candidates["track_ids"], max_history_tracks=3)
        self.assertEqual(candidates["track_ids"].tolist(), [5, 8, 12])
        self.assertEqual(labels.tolist(), [0, 1, 3, 2])


    def test_padding_mask_for_early_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = f"{tmpdir}/tracking_infos_train.pkl"
            with open(info_path, "wb") as f:
                pickle.dump([_info(0, [1]), _info(1, [1])], f)
            cfg = load_config(
                overrides={
                    "dataset": {
                        "name": "tmp",
                        "K": 3,
                        "stride": 1,
                        "info_paths": {"tmp": {"train": info_path, "val": info_path}},
                        "max_history_tracks": 4,
                        "max_current_objects": 8,
                    }
                }
            )
            dataset = SequenceWindowDataset(cfg, split="train")
            sample = dataset[0]
            self.assertEqual(sample["frame_valid_mask"].tolist(), [False, False, True])
            self.assertEqual(sample["pointer_labels"].tolist(), [4])


if __name__ == "__main__":
    unittest.main()
