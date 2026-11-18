import unittest

from delay_percentile import (TAN_FELT_DELAY, TAN_THOUGHTFUL, percent_1dp, cohens_d, SAMPLE_A, SAMPLE_B, SAMPLE_C, frames_in, is_good, percentile_linear,
                              percentile_nearest_rank, share_over)


class Percentile(unittest.TestCase):
    def test_hand_example(self):
        # 손계산: 정렬하면 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 3.6 4.8. 위치 0.75*9=6.75 -> 2.3과 2.4 사이 75% 지점
        self.assertAlmostEqual(percentile_linear(SAMPLE_A, 75), 2.375)
        self.assertEqual(percentile_nearest_rank(SAMPLE_A, 75), 2.4)  # 올림(10*0.75)=8번째

    def test_median_and_mean_differ(self):
        self.assertAlmostEqual(percentile_linear(SAMPLE_A, 50), 2.15)
        self.assertAlmostEqual(sum(SAMPLE_A) / 10, 2.48)

    def test_good_threshold(self):
        self.assertTrue(is_good(percentile_linear(SAMPLE_A, 75), 2.5))
        self.assertFalse(is_good(percentile_linear(SAMPLE_B, 75), 2.5))

    def test_mean_passes_but_p75_fails(self):
        # 표본 C: 평균 2.10초는 기준 안, 75번째 백분위 2.85초는 기준 밖, 2.5초를 넘은 방문 40%
        self.assertAlmostEqual(sum(SAMPLE_C) / 10, 2.10)
        self.assertAlmostEqual(percentile_linear(SAMPLE_C, 75), 2.85)
        self.assertTrue(is_good(sum(SAMPLE_C) / 10, 2.5))
        self.assertFalse(is_good(percentile_linear(SAMPLE_C, 75), 2.5))
        self.assertAlmostEqual(share_over(SAMPLE_C, 2.5), 0.4)

    def test_share_over(self):
        self.assertAlmostEqual(share_over(SAMPLE_A, 2.5), 0.2)

    def test_frames(self):
        self.assertAlmostEqual(frames_in(200), 12.0)


class TanValues(unittest.TestCase):
    def test_effect_size_2s_vs_9s(self):
        (m2, s2, n2), (m9, s9, n9) = TAN_THOUGHTFUL[2], TAN_THOUGHTFUL[9]
        self.assertAlmostEqual(m9 - m2, 0.33)
        self.assertAlmostEqual(cohens_d(m2, s2, n2, m9, s9, n9), 0.31, places=2)

    def test_felt_delay_share(self):
        shares = {k: percent_1dp(a, b) for k, (a, b) in TAN_FELT_DELAY.items()}
        self.assertEqual(shares, {2: 26.6, 9: 66.3, 20: 81.5})


if __name__ == "__main__":
    unittest.main()
