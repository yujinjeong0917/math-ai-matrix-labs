"""12장 테스트: 정상상태 효과 1.5, 1시간 단위 교대 시 이월을 못 보고 tau만 잡는 편향, 블록·세척기간에 따른 편향 감소,
세척기간이 이월 길이(2슬롯)와 같으면 불편, 세척의 데이터 손실 비율, 기본 곡선이 없을 때의 표준편차, 배정 균형을 확인한다.
허용 오차는 반복 300회의 흔들림을 반영한 값이다."""
import random
import unittest
import switchback as s


class Switchback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = s.report()

    def test_true_steady_state(self):
        self.assertAlmostEqual(s.TRUE_STEADY, 1.5)  # 6.0 - 3.0 - 1.5

    def test_hourly_flip_recovers_only_tau(self):
        # 슬롯마다 켬/끔이 번갈아 무작위라 이전 상태가 켬/끔 양쪽에 같게 섞여 이월이 상쇄된다. 남는 것은 즉시 효과 6(tau)
        x = self.r[(1, 0)]
        self.assertAlmostEqual(x["mean"], 6.0, delta=0.3)  # 이월을 못 보고 tau만 본다
        self.assertGreater(x["bias"], 4.0)

    def test_bias_shrinks_with_block_and_washout(self):
        # 블록 4에서 세척 0 -> 1 -> 2슬롯으로 늘릴수록 편향이 줄고, 2에서 거의 0(절댓값 0.15 미만)
        b0 = self.r[(4, 0)]["bias"]
        b1 = self.r[(4, 1)]["bias"]
        b2 = self.r[(4, 2)]["bias"]
        self.assertGreater(b0, b1)
        self.assertGreater(b1, abs(b2))
        self.assertLess(abs(b2), 0.15)

    def test_washout_equal_to_carryover_length_is_unbiased(self):
        # 세척 2슬롯 = 이월이 2슬롯까지라는 모형의 가정과 같은 길이. 블록이 4 이상이면 남는 슬롯의 이월 영향이 없다
        for key in ((4, 2), (8, 2), (12, 2), (24, 2)):
            self.assertLess(abs(self.r[key]["bias"]), 0.2)

    def test_washout_costs_data(self):
        self.assertAlmostEqual(s.kept_share(4, 2), 0.75, delta=0.02)  # 손계산 1 - 2/4 x 1/2
        self.assertAlmostEqual(s.kept_share(24, 2), 0.958, delta=0.02)

    def test_flat_base_sd_is_small_and_flat(self):
        # 시간대·요일 패턴을 없애면 흩어짐이 잡음 수준(0.5 미만)으로 줄고, 패턴이 있는 설정(4, 2)의 SD보다 작다
        sds = s.flat_sd_report().values()
        self.assertLess(max(sds), 0.5)
        self.assertLess(max(sds), self.r[(4, 2)]['sd'])

    def test_assignment_balanced(self):
        # 켬 슬롯 수가 정확히 N/2: 블록 길이 8 -> 블록 30개 중 15개가 켬 x 8슬롯 = 120
        T = s.assign(8, random.Random(1))
        self.assertEqual(len(T), s.N)
        self.assertEqual(sum(T), s.N // 2)


if __name__ == "__main__":
    unittest.main()
