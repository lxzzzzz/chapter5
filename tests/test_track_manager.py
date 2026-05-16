import unittest

import numpy as np

from generative_tracking.track_manager import TrackManager


class TrackManagerTest(unittest.TestCase):
    def test_track_manager_new_reuse_lost_delete(self):
        manager = TrackManager(max_lost_frames=1)
        boxes = np.zeros((2, 7), dtype=np.float32)
        out0 = manager.update("seq", "0", boxes, ["Car", "Car"], np.ones(2), np.asarray([0, 0]), np.asarray([], dtype=np.int64))
        ids0 = [track["id"] for track in out0["tracks"]]
        self.assertEqual(ids0, [0, 1])

        out1 = manager.update(
            "seq",
            "1",
            boxes[:1],
            ["Car"],
            np.ones(1),
            np.asarray([0]),
            np.asarray([ids0[0]], dtype=np.int64),
            np.asarray([True]),
        )
        self.assertEqual(out1["tracks"][0]["id"], ids0[0])
        self.assertEqual(manager.tracks[ids0[1]].state, "lost")

        manager.update("seq", "2", boxes[:1], ["Car"], np.ones(1), np.asarray([0]), np.asarray([ids0[0]], dtype=np.int64), np.asarray([True]))
        self.assertNotIn(ids0[1], manager.tracks)


if __name__ == "__main__":
    unittest.main()
