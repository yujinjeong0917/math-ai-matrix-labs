import math
import unittest

from effect_reading import (GUIDE_ROWS, STRENGTH, cohen_d, coverage, d_confidence_interval, share_at_least, simulate,
                            valid_guide_row)


class Reading(unittest.TestCase):
    def test_hand_example(self):
        self.assertAlmostEqual(cohen_d([5, 6, 7], [3, 4, 5]), 2.0)

    def test_width_scales(self):
        w1 = d_confidence_interval(0.0, 100, 100)
        w2 = d_confidence_interval(0.0, 400, 400)
        self.assertAlmostEqual((w1[1] - w1[0]) / (w2[1] - w2[0]), 2.0, places=6)

    def test_small_n_includes_zero(self):
        a, b = simulate(20, 0.4, 11)
        d = cohen_d(a, b)
        lo, hi = d_confidence_interval(d, 20, 20)
        self.assertLess(lo, 0)

    def test_reproducible(self):
        self.assertEqual(simulate(30, 0.4, 5), simulate(30, 0.4, 5))

    def test_coverage_near_95(self):
        rate, mean_d = coverage(100, 0.4)
        self.assertGreater(rate, 0.90)
        self.assertLess(rate, 1.0)
        self.assertAlmostEqual(mean_d, 0.4, delta=0.05)

    def test_sample_d_often_exceeds_058(self):
        self.assertEqual(share_at_least(100, 0.4, 0.58), 28)

    def test_guide_rows_all_valid(self):
        self.assertEqual(len(GUIDE_ROWS), 7)
        self.assertTrue(all(valid_guide_row(r) for r in GUIDE_ROWS))

    def test_low_trust_rows_marked(self):
        for r in GUIDE_ROWS:
            if r["근거"] in ("Wang 외 실험 3", "Wang 외 실험 4", "Fang 외 실험 1"):
                self.assertEqual(r["강도"], STRENGTH[2])

    def test_guide_row(self):
        ok = {"조건": "x", "대상": "y", "이유": "z", "근거": "w", "강도": STRENGTH[1]}
        self.assertTrue(valid_guide_row(ok))
        self.assertFalse(valid_guide_row({**ok, "강도": "확실"}))
        self.assertFalse(valid_guide_row({**ok, "대상": ""}))


if __name__ == "__main__":
    unittest.main()
