import unittest

from evidence_ledger import (MONTH_1, MONTH_3_A, MONTH_3_B, SOURCES, accepted, accepted_wrong, acceptance_rate,
                             ai_correct, measured_over_months, share_of, total)


class Ledger(unittest.TestCase):
    def test_no_source_measures_months(self):
        self.assertEqual(len(SOURCES), 4)
        self.assertFalse(any(measured_over_months(k) for k in SOURCES))

    def test_zhang_constants(self):
        z = SOURCES["Zhang 외 2025"]
        self.assertEqual((z["papers"], z["pattern_papers"]), (134, 106))
        self.assertAlmostEqual(share_of(106, 134), 0.791, places=3)

    def test_pareek_constants(self):
        self.assertEqual(SOURCES["Pareek 외 2026"]["participants"], 12)

    def test_logs_have_40_suggestions(self):
        for log in (MONTH_1, MONTH_3_A, MONTH_3_B):
            self.assertEqual(total(log), 40)

    def test_hand_calc_month1(self):
        self.assertEqual(accepted(MONTH_1), 24)
        self.assertAlmostEqual(acceptance_rate(MONTH_1), 0.60)

    def test_same_acceptance_different_story(self):
        self.assertEqual(accepted(MONTH_3_A), 32)
        self.assertEqual(accepted(MONTH_3_B), 32)
        self.assertAlmostEqual(acceptance_rate(MONTH_3_A), 0.80)
        self.assertAlmostEqual(acceptance_rate(MONTH_3_B), 0.80)
        self.assertEqual(accepted_wrong(MONTH_1), 4)
        self.assertEqual(accepted_wrong(MONTH_3_A), 2)
        self.assertEqual(accepted_wrong(MONTH_3_B), 8)

    def test_ai_correct_counts(self):
        self.assertEqual(ai_correct(MONTH_1), 28)
        self.assertEqual(ai_correct(MONTH_3_A), 32)
        self.assertEqual(ai_correct(MONTH_3_B), 28)


if __name__ == "__main__":
    unittest.main()
