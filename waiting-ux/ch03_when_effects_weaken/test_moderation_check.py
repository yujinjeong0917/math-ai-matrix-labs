import unittest

from moderation_check import (EXP3, EXP4, REPORTED_DF2, TAN_PERCEIVED, TAN_THOUGHTFUL, TAN_USEFUL, anova_error_df,
                              d_between, pct1, interaction_contrast, static_minus_dynamic)


class Checks(unittest.TestCase):
    def test_hand_differences(self):
        # 손계산: 4.83-3.12=1.71, 3.68-3.25=0.43, 차이의 차이 1.28
        self.assertAlmostEqual(static_minus_dynamic(EXP3["낮은 몰입"]), 1.71)
        self.assertAlmostEqual(static_minus_dynamic(EXP3["높은 몰입"]), 0.43)
        self.assertAlmostEqual(interaction_contrast(EXP3), 1.28)
        self.assertAlmostEqual(interaction_contrast(EXP4), 0.70)

    def test_effect_stays_positive_in_weak_cell(self):
        self.assertGreater(static_minus_dynamic(EXP3["높은 몰입"]), 0)
        self.assertGreater(static_minus_dynamic(EXP4["실용"]), 0)

    def test_df_mismatch(self):
        self.assertEqual(anova_error_df(400, 4), 396)
        self.assertEqual(REPORTED_DF2["실험 4"], 396)
        self.assertNotEqual(REPORTED_DF2["실험 3"], 396)

    def test_index_gap(self):
        from moderation_check import INDIRECT, index_gap
        d3, r3 = index_gap(*INDIRECT["실험 3"])
        d4, r4 = index_gap(*INDIRECT["실험 4"])
        self.assertAlmostEqual(d3, 0.341, places=3)
        self.assertAlmostEqual(d4, 0.409, places=3)
        self.assertNotAlmostEqual(d3, r3, places=2)
        self.assertNotAlmostEqual(d4, r4, places=2)

    def test_tan_combined_shares(self):
        got = {k: pct1(x, n) for k, (x, n) in TAN_PERCEIVED.items()}
        self.assertEqual(got, {"2초": 26.6, "9초": 66.3, "20초": 81.5})

    def test_tan_d(self):
        self.assertAlmostEqual(d_between(TAN_THOUGHTFUL["2초"], TAN_THOUGHTFUL["9초"]), 0.31, places=2)
        self.assertAlmostEqual(d_between(TAN_THOUGHTFUL["2초"], TAN_THOUGHTFUL["20초"]), 0.34, places=2)

    def test_tan_useful_diff(self):
        self.assertAlmostEqual(TAN_USEFUL["9초"] - TAN_USEFUL["2초"], 0.25)


if __name__ == "__main__":
    unittest.main()
