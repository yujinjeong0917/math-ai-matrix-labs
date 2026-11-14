import os
import sys
import math
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdd as r


class Hand(unittest.TestCase):
    def test_extrapolated_jump_is_ten_not_fourteen(self):
        h = r.hand_rdd()
        self.assertEqual((h["slope_below"], h["slope_above"]), (4.0, 4.0))
        self.assertEqual((h["below_at_cut"], h["above_at_cut"], h["jump"]), (50.0, 60.0, 10.0))
        self.assertEqual(h["adjacent_bin_gap"], 14.0)


class SingleRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = r.make_data(6000, 21)

    def test_naive_and_global_line_fail(self):
        self.assertEqual(sum(x["d"] for x in self.rows), 1493)
        self.assertEqual(round(r.naive_gap(self.rows), 2), 19.44)
        self.assertEqual(round(r.global_linear_common_slope(self.rows), 2), 16.81)

    def test_local_linear_by_bandwidth(self):
        want = {0.25: (10.75, 152, 163), 0.5: (10.70, 316, 294), 1.0: (10.63, 672, 569),
                2.0: (10.64, 1467, 975), 4.0: (10.54, 2969, 1385), 8.0: (11.49, 4361, 1490)}
        for h, (est, nl, nr) in want.items():
            got = r.local_linear(self.rows, h)
            self.assertEqual((round(got[0], 2), got[1], got[2]), (est, nl, nr), h)


class Replication(unittest.TestCase):
    def test_bandwidth_tradeoff(self):
        res = r.replicate(200, 6000, 1000)
        want = {0.25: (0.09, 1.60), 0.5: (0.16, 1.12), 1.0: (0.04, 0.84),
                2.0: (0.03, 0.55), 4.0: (0.24, 0.40), 8.0: (1.29, 0.34)}
        for h, (bias, spread) in want.items():
            self.assertEqual((round(r.mean(res[h]) - r.TAU, 2), round(r.sd(res[h]), 2)), (bias, spread), h)
        # 본문: 평균 오차와 표준편차를 합친 크기 sqrt(오차^2 + SD^2)는 4.0에서 가장 작다
        size = {h: round(math.hypot(r.mean(res[h]) - r.TAU, r.sd(res[h])), 2) for h in res}
        self.assertEqual(size, {0.25: 1.60, 0.5: 1.13, 1.0: 0.84, 2.0: 0.56, 4.0: 0.47, 8.0: 1.34})


class Manipulation(unittest.TestCase):
    def test_bunching_above_cutoff_and_bias(self):
        orig = r.make_data(6000, 21)
        mani = r.make_data(6000, 21, manipulate=True)
        self.assertEqual(sum(1 for x in mani if x["moved"]), 170)
        o = dict(r.bin_counts(orig))
        m = dict(r.bin_counts(mani))
        self.assertEqual((o[9.0], o[9.5], o[10.0]), (356, 316, 294))
        self.assertEqual((m[9.0], m[9.5], m[10.0]), (262, 240, 464))
        self.assertEqual(o[8.0], m[8.0])  # 올리지 않은 구간은 같은 고객
        got = {h: round(r.local_linear(mani, h)[0], 2) for h in (0.5, 1.0, 2.0)}
        self.assertEqual(got, {0.5: 15.29, 1.0: 14.28, 2.0: 14.09})

    def test_bias_persists_over_200_runs(self):
        res = r.replicate(200, 6000, 1000, manipulate=True)
        got = {h: (round(r.mean(res[h]) - r.TAU, 2), round(r.sd(res[h]), 2)) for h in (0.5, 1.0, 2.0)}
        self.assertEqual(got, {0.5: (4.46, 1.01), 1.0: (3.91, 0.80), 2.0: (3.61, 0.57)})


if __name__ == "__main__":
    unittest.main()
