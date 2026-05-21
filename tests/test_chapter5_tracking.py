import unittest

import numpy as np

from generative_tracking.nova_data import DetectionFrame
from tools.run_chapter5_tracking import Chapter5Tracker


def _box(x):
    return np.asarray([x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)


class Chapter5TrackingTest(unittest.TestCase):
    def test_fmca_new_tracks_are_output_on_birth_frame(self):
        tracker = Chapter5Tracker(
            class_names=["Car"],
            variant="full_5_5",
            max_lost_frames=2,
            min_hits=1,
            high_score_thresh=0.6,
            low_score_thresh=0.3,
            iou_threshold=0.001,
            center_distance=4.0,
            tlom_threshold=0.5,
            dt_hypotheses=[1.0],
        )
        frame = DetectionFrame(
            "seq0",
            "000000",
            0,
            np.stack([_box(0.0), _box(10.0)]),
            np.asarray([0.9, 0.8], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
        )

        out = tracker.update(frame)

        self.assertEqual([track["id"] for track in out["tracks"]], [0, 1])
        self.assertEqual([tracker.tracks[track_id].lost_frames for track_id in (0, 1)], [0, 0])

    def test_fmca_recovers_confirmed_track_with_low_score_detection(self):
        tracker = Chapter5Tracker(
            class_names=["Car"],
            variant="full_5_5",
            max_lost_frames=2,
            min_hits=1,
            high_score_thresh=0.6,
            low_score_thresh=0.3,
            iou_threshold=0.001,
            center_distance=4.0,
            tlom_threshold=0.5,
            dt_hypotheses=[1.0],
        )
        frame0 = DetectionFrame(
            "seq0",
            "000000",
            0,
            np.stack([_box(0.0)]),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
        )
        tracker.update(frame0)
        frame1 = DetectionFrame(
            "seq0",
            "000001",
            1,
            np.stack([_box(0.5)]),
            np.asarray([0.4], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
        )

        out = tracker.update(frame1)

        self.assertEqual([track["id"] for track in out["tracks"]], [0])
        self.assertEqual(tracker.tracks[0].lost_frames, 0)
        self.assertEqual(tracker.tracks[0].hits, 2)


if __name__ == "__main__":
    unittest.main()
