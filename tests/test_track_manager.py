import unittest

import numpy as np

from generative_tracking.track_manager import TrackEmbeddingManager


class TrackEmbeddingManagerTest(unittest.TestCase):
    def test_embedding_manager_new_reuse_lost_delete(self):
        manager = TrackEmbeddingManager(max_lost_frames=1, match_threshold=0.5)
        boxes = np.zeros((2, 7), dtype=np.float32)
        embeds = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out0 = manager.update("seq", "0", boxes, np.asarray([0, 0]), ["Car"], np.ones(2), embeds)
        ids0 = [track["id"] for track in out0["tracks"]]
        self.assertEqual(ids0, [0, 1])

        out1 = manager.update("seq", "1", boxes[:1], np.asarray([0]), ["Car"], np.ones(1), embeds[:1])
        self.assertEqual(out1["tracks"][0]["id"], ids0[0])
        self.assertEqual(manager.tracks[ids0[1]].state, "lost")

        manager.update("seq", "2", boxes[:1], np.asarray([0]), ["Car"], np.ones(1), embeds[:1])
        self.assertNotIn(ids0[1], manager.tracks)


if __name__ == "__main__":
    unittest.main()
