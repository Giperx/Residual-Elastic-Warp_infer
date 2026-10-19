"""CPU-only unit tests for the histogram-matched sparse-GT metrics.

No torch and no images are required. Predictions, GT, and masks are built in
memory so the metric contract can be checked in isolation.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rewarp_drive import metrics  # noqa: E402


def _gradient(height, width):
    yy = np.linspace(0.05, 0.95, height, dtype=np.float64)[:, None]
    xx = np.linspace(0.05, 0.95, width, dtype=np.float64)[None, :]
    base = (yy * 0.5 + xx * 0.5)[..., None]
    return np.repeat(base, 3, axis=-1)


class FrameHMTests(unittest.TestCase):
    def test_identical_is_perfect(self):
        image = _gradient(12, 15)
        mask = np.ones(image.shape[:2], dtype=bool)
        row = metrics.frame_hm(image, image.copy(), mask)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["hm_mae"], 0.0, places=12)
        self.assertAlmostEqual(row["hm_rmse"], 0.0, places=12)
        self.assertEqual(row["hm_psnr"], float("inf"))
        self.assertAlmostEqual(row["hm_ssim"], 1.0, places=12)

    def test_empty_mask_returns_none(self):
        image = _gradient(8, 8)
        mask = np.zeros(image.shape[:2], dtype=bool)
        self.assertIsNone(metrics.frame_hm(image, image.copy(), mask))

    def test_small_color_shift_is_finite(self):
        gt = _gradient(16, 24)
        # A spatially localised shift: a perfectly monotonic global shift would
        # be undone by histogram matching, so shift only the left half.
        pred = gt.copy()
        pred[:, : gt.shape[1] // 2] += 0.1
        pred = np.clip(pred, 0.0, 1.0)
        mask = np.ones(gt.shape[:2], dtype=bool)
        row = metrics.frame_hm(pred, gt, mask)
        self.assertIsNotNone(row)
        self.assertTrue(np.isfinite(row["hm_psnr"]))
        self.assertGreater(row["hm_mae"], 0.0)
        self.assertGreater(row["hm_ssim"], 0.0)
        self.assertLessEqual(row["hm_ssim"], 1.0)


class RegionSliceTests(unittest.TestCase):
    def test_1554_thirds(self):
        self.assertEqual(
            metrics.region_slices(1554),
            {"Left": (0, 518), "Center": (518, 1036), "Right": (1036, 1554)},
        )

    def test_indivisible_width_is_error(self):
        with self.assertRaises(SystemExit):
            metrics.region_slices(1553)


class AggregationTests(unittest.TestCase):
    def test_mean_rows_skips_non_finite_psnr(self):
        rows = [
            {"hm_psnr": 10.0, "hm_ssim": 0.5, "hm_mae": 0.1, "hm_rmse": 0.2},
            {"hm_psnr": float("inf"), "hm_ssim": 1.0, "hm_mae": 0.0, "hm_rmse": 0.0},
        ]
        stats = metrics.mean_rows(rows)
        self.assertAlmostEqual(stats["hm_psnr"], 10.0, places=6)
        self.assertEqual(stats["n"], 2)

    def test_mean_of_is_region_mean(self):
        left = {"hm_psnr": 20.0, "hm_ssim": 0.8, "hm_mae": 0.2, "hm_rmse": 0.4}
        right = {"hm_psnr": 30.0, "hm_ssim": 0.6, "hm_mae": 0.4, "hm_rmse": 0.2}
        combined = metrics.mean_of([left, right])
        self.assertAlmostEqual(combined["hm_psnr"], 25.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
