import unittest

from vidgoclip.reframe import TrackPoint, smooth_tracking_points


class ReframeTests(unittest.TestCase):
    def test_tracking_smoothing_reduces_large_jump(self):
        raw = [
            TrackPoint(0.0, 0.20, "face"),
            TrackPoint(0.25, 0.22, "face"),
            TrackPoint(0.50, 0.90, "motion"),
        ]
        smooth = smooth_tracking_points(raw, alpha=0.25)

        raw_jump = raw[-1].x_ratio - raw[-2].x_ratio
        smooth_jump = smooth[-1].x_ratio - smooth[-2].x_ratio

        self.assertLess(abs(smooth_jump), abs(raw_jump))
        self.assertGreaterEqual(smooth[-1].x_ratio, 0.0)
        self.assertLessEqual(smooth[-1].x_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
