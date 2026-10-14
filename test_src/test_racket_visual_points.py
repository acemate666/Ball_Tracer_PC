import unittest
from pathlib import Path

import numpy as np

from test_src.racket_visual_points import VisualPoints, feature, red_marker


class RacketVisualPointsTest(unittest.TestCase):
    def test_red_paint_separates_from_white_wrap_and_broad_skin_region(self):
        image = np.full((100, 100, 3), 15, dtype=np.uint8)
        image[32:43, 46:51] = [160, 160, 160]
        image[50:53, 48:51] = [20, 25, 90]
        image[36:60, 57:68] = [55, 80, 130]
        found = red_marker(image, np.array([49., 48.]))
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0]["xy"], [49., 51.], atol=.1)

    def test_small_imu_label_is_not_confused_with_large_white_grip_band(self):
        image = np.full((80, 80), 20, dtype=np.uint8)
        image[23:38, 36:43] = 180
        image[44:47, 39:42] = 140
        found = feature(image, np.array([40., 40.]), True)
        self.assertTrue(found)
        self.assertTrue(all(np.linalg.norm(np.array(p["xy"]) - [40, 45]) < 2 for p in found))

    def test_multiview_geometry_rejects_one_wrong_camera_point(self):
        path = Path(__file__).resolve().parents[1] / "src/config/four_camera_calib_18.json"
        geometry = VisualPoints(path)
        point = np.array([.2, 5., 1.6])
        pixels = {sn: geometry.project(point, sn) for sn in geometry.cameras}
        np.testing.assert_allclose(geometry.triangulate(pixels), point, atol=1e-9)
        candidates = {sn: [{"xy": uv.tolist(), "score": 80.}] for sn, uv in pixels.items()}
        bad = list(candidates)[-1]
        candidates[bad][0]["xy"] = (pixels[bad] + [70, -50]).tolist()
        located = geometry.locate(candidates)
        self.assertIsNotNone(located)
        self.assertNotIn(bad, located["pixels"])
        np.testing.assert_allclose(located["xyz_m"], point, atol=1e-9)
