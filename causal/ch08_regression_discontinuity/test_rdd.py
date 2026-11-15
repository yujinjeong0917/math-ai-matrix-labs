"""8장 테스트: 손계산 점프, 단순 비교·전체 직선의 실패, 대역폭별 국소 선형 추정, 200회 반복의 편향-분산 균형,
기준선 직전 조작(bunching)이 있을 때의 구간 인원 쌓임과 지속되는 편향을 확인한다. 기대값은 고정 시드 결과의 반올림이다."""
import os
import sys
import math
import unittest

# 어느 디렉터리에서 실행해도 같은 폴더의 rdd.py를 찾도록 경로를 앞에 넣는다
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdd as r


class Hand(unittest.TestCase):
    def test_extrapolated_jump_is_ten_not_fourteen(self):
        # 양쪽 기울기 4. 기준선에서 아래 50, 위 60이라 점프 10. 인접 구간 평균 차이 14는 기울기 몫 4가 섞인 값
        h = r.hand_rdd()
        self.assertEqual((h["slope_below"], h["slope_above"]), (4.0, 4.0))
        self.assertEqual((h["below_at_cut"], h["above_at_cut"], h["jump"]), (50.0, 60.0, 10.0))
        self.assertEqual(h["adjacent_bin_gap"], 14.0)


class SingleRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = r.make_data(6000, 21)

    def test_naive_and_global_line_fail(self):
        # 쿠폰 1493명(x >= 10). 단순 비교 19.44와 전체 직선 16.81 모두 진짜 효과 10에서 멀다
        self.assertEqual(sum(x["d"] for x in self.rows), 1493)
        self.assertEqual(round(r.naive_gap(self.rows), 2), 19.44)
        self.assertEqual(round(r.global_linear_common_slope(self.rows), 2), 16.81)

    def test_local_linear_by_bandwidth(self):
        # 값: (추정 점프, 왼쪽 인원, 오른쪽 인원). 대역폭이 8.0이면 곡선 편향이 커져 11.49로 벗어난다
        want = {0.25: (10.75, 152, 163), 0.5: (10.70, 316, 294), 1.0: (10.63, 672, 569),
                2.0: (10.64, 1467, 975), 4.0: (10.54, 2969, 1385), 8.0: (11.49, 4361, 1490)}
        for h, (est, nl, nr) in want.items():
            got = r.local_linear(self.rows, h)
            self.assertEqual((round(got[0], 2), got[1], got[2]), (est, nl, nr), h)


class Replication(unittest.TestCase):
    def test_bandwidth_tradeoff(self):
        res = r.replicate(200, 6000, 1000)
        # 값: (평균 오차, 표준편차). 대역폭이 좁을수록 SD가 크고(1.60), 넓을수록 편향이 커진다(1.29)
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
        # 올려서 넘어간 고객 170명. 9.0~10.0 구간 인원이 줄고(356->262, 316->240) 10.0~10.5 구간이 쌓인다(294->464)
        self.assertEqual(sum(1 for x in mani if x["moved"]), 170)
        o = dict(r.bin_counts(orig))
        m = dict(r.bin_counts(mani))
        self.assertEqual((o[9.0], o[9.5], o[10.0]), (356, 316, 294))
        self.assertEqual((m[9.0], m[9.5], m[10.0]), (262, 240, 464))
        self.assertEqual(o[8.0], m[8.0])  # 올리지 않은 구간은 같은 고객
        # 조작 후 점프가 14~15로 부풀려진다(진짜 10). 위로 옮겨간 고객이 의욕이 높아 위쪽 평균 y를 올리기 때문
        got = {h: round(r.local_linear(mani, h)[0], 2) for h in (0.5, 1.0, 2.0)}
        self.assertEqual(got, {0.5: 15.29, 1.0: 14.28, 2.0: 14.09})

    def test_bias_persists_over_200_runs(self):
        # 200번 반복해도 평균 오차가 3.6~4.5로 남는다: 표본을 늘려도 사라지지 않는 구조적 편향
        res = r.replicate(200, 6000, 1000, manipulate=True)
        got = {h: (round(r.mean(res[h]) - r.TAU, 2), round(r.sd(res[h]), 2)) for h in (0.5, 1.0, 2.0)}
        self.assertEqual(got, {0.5: (4.46, 1.01), 1.0: (3.91, 0.80), 2.0: (3.61, 0.57)})


if __name__ == "__main__":
    unittest.main()
