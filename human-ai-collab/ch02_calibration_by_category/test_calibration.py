import unittest

from calibration import CATEGORIES, LOG, accuracy, acceptance, gap_points, overall, range_ratio


class Calibration(unittest.TestCase):
    def test_each_cell_has_20(self):
        self.assertTrue(all(v[0] == 20 for v in LOG.values()))
        self.assertEqual(len(LOG), 8)

    def test_hand_calc_delivery(self):
        self.assertAlmostEqual(accuracy("배송", 1), 0.80)
        self.assertAlmostEqual(acceptance("배송", 1), 0.70)
        self.assertEqual(gap_points("배송", 1), -10.0)
        self.assertAlmostEqual(accuracy("배송", 3), 0.50)
        self.assertAlmostEqual(acceptance("배송", 3), 0.75)
        self.assertEqual(gap_points("배송", 3), 25.0)

    def test_same_overall_acceptance_both_months(self):
        self.assertAlmostEqual(overall(1)[1], 0.70)
        self.assertAlmostEqual(overall(3)[1], 0.70)

    def test_overall_accuracy(self):
        self.assertAlmostEqual(overall(1)[0], 51 / 80)
        self.assertAlmostEqual(overall(3)[0], 49 / 80)

    def test_range_ratio(self):
        self.assertAlmostEqual(range_ratio(1), 0.10 / 0.45)
        self.assertAlmostEqual(range_ratio(3), 0.20 / 0.45)

    def test_overall_gap_hides_category_gaps(self):
        gaps = [gap_points(c, 1) for c in CATEGORIES]
        self.assertLess(min(gaps), 0)
        self.assertGreater(max(gaps), 20)
        self.assertLess(abs((overall(1)[1] - overall(1)[0]) * 100), 7)


if __name__ == "__main__":
    unittest.main()
