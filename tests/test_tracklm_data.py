import pickle
import tempfile
import unittest

import numpy as np

from generative_tracking.config import load_config
from generative_tracking.data import SequenceWindowDataset, frame_to_objects, tracklm_collate


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
    def test_frame_to_objects_reads_gt_tracks(self):
        objects = frame_to_objects(_info(0, [5, 8]), {"Car": 0})
        self.assertEqual(objects.track_ids.tolist(), [5, 8])
        self.assertEqual(objects.class_ids.tolist(), [0, 0])

    def test_frame_to_objects_filters_non_target_classes(self):
        info = _info(0, [5, 8])
        info["annos"]["name"] = np.asarray(["Car", "Pedestrian"])
        objects = frame_to_objects(info, {"Car": 0})
        self.assertEqual(objects.track_ids.tolist(), [5])
        self.assertEqual(objects.class_names.tolist(), ["Car"])

    def test_padding_mask_and_targets_for_early_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info_path = f"{tmpdir}/tracking_infos_train.pkl"
            with open(info_path, "wb") as f:
                pickle.dump([_info(0, [1]), _info(1, [1, 2])], f)
            cfg = load_config(
                overrides={
                    "dataset": {
                        "name": "tmp",
                        "K": 3,
                        "stride": 1,
                        "info_paths": {"tmp": {"train": info_path, "val": info_path}},
                        "max_objects": 8,
                    }
                }
            )
            dataset = SequenceWindowDataset(cfg, split="train")
            sample = dataset[0]
            self.assertEqual(sample["frame_valid_mask"].tolist(), [False, False, True])
            self.assertEqual(sample["target_track_ids"].tolist(), [1])
            batch = tracklm_collate([dataset[1]])
            self.assertEqual(tuple(batch["target_boxes"].shape), (1, 2, 7))
            self.assertTrue(batch["target_valid_mask"][0, :2].all())


if __name__ == "__main__":
    unittest.main()
