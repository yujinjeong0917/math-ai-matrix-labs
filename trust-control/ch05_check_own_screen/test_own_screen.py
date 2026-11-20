import statistics
import unittest

from own_screen import (ARM_A, ARM_B, COLUMNS, GUIDE_ROWS, STRENGTHS, arm_values, bootstrap_ci, diff_of_means,
                        person_measures, valid_guide_row)


class OwnScreen(unittest.TestCase):
    def test_hand_person(self):
        m = person_measures(6, 4)
        self.assertAlmostEqual(m["overtrust"], 0.5)
        self.assertAlmostEqual(m["undertrust"], 0.25)
        self.assertAlmostEqual(m["appropriate"], 0.625)

    def test_means(self):
        self.assertEqual(len(ARM_A), 12)
        self.assertEqual(len(ARM_B), 12)
        a, b = arm_values(ARM_A, "appropriate"), arm_values(ARM_B, "appropriate")
        self.assertAlmostEqual(statistics.fmean(a), 0.625)
        self.assertAlmostEqual(statistics.fmean(b), 0.625)
        self.assertAlmostEqual(diff_of_means(arm_values(ARM_A, "overtrust"), arm_values(ARM_B, "overtrust")), 0.1875)

    def test_ci_reproducible_and_ordered(self):
        a, b = arm_values(ARM_A, "overtrust"), arm_values(ARM_B, "overtrust")
        c1, c2 = bootstrap_ci(a, b), bootstrap_ci(a, b)
        self.assertEqual(c1, c2)
        self.assertLess(c1[0], c1[1])
        self.assertGreater(c1[0], 0.0)

    def test_appropriate_ci_includes_zero(self):
        lo, hi = bootstrap_ci(arm_values(ARM_A, "appropriate"), arm_values(ARM_B, "appropriate"))
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_bad_counts(self):
        with self.assertRaises(ValueError):
            person_measures(9, 0)

    def test_guide_rows(self):
        self.assertEqual(len(GUIDE_ROWS), 7)
        self.assertTrue(all(valid_guide_row(r) for r in GUIDE_ROWS))
        bad = list(GUIDE_ROWS[0])
        bad[1] = ""
        self.assertFalse(valid_guide_row(bad))
        bad = list(GUIDE_ROWS[0])
        bad[4] = "확실함"
        self.assertFalse(valid_guide_row(bad))
        self.assertEqual(len(COLUMNS), 5)
        self.assertEqual(len(STRENGTHS), 4)


if __name__ == "__main__":
    unittest.main()
