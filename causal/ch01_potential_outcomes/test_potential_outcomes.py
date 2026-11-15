"""1장 테스트: 손계산 표(8명)의 정확한 값과, 시뮬레이션에서 자기 선택 vs 무작위 배정의 차이를 확인한다.
손계산 값은 정수 연산이라 assertEqual로 정확히 비교하고, 시뮬레이션 값은 고정 시드에서 반올림한 값이다."""
import unittest

import potential_outcomes as po


class HandTable(unittest.TestCase):
    def test_hand_numbers(self):
        h = po.hand_table()
        # ATE (2+2+0+0+4+4+4+4)/8 = 2.5, ATT (2+2+0+0)/4 = 1.0, ATC (4+4+4+4)/4 = 4.0
        self.assertEqual((h["ate"], h["att"], h["atc"]), (2.5, 1.0, 4.0))
        self.assertEqual((h["treated_obs"], h["control_obs"], h["naive"]), (18.0, 4.0, 14.0))
        # 선택 편향: 처치군 Y0 평균 17 - 대조군 Y0 평균 4 = 13
        self.assertEqual(h["selection_bias"], 13.0)

    def test_naive_equals_att_plus_selection_bias(self):
        h = po.hand_table()
        self.assertEqual(h["naive"], h["att"] + h["selection_bias"])

    def test_ate_is_weighted_mean_of_att_and_atc(self):
        h = po.hand_table()
        # 처치군과 대조군이 각각 4명이라 가중치가 1/2씩이다(인원이 다르면 인원 비율 가중)
        self.assertEqual(h["ate"], 0.5 * h["att"] + 0.5 * h["atc"])


class Simulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = po.make_population()

    def test_self_selection_overstates_effect(self):
        e = po.estimates(self.pop, po.self_selected(self.pop))
        self.assertEqual(round(e["ate"], 2), 2.48)
        self.assertEqual(round(e["att"], 2), 2.08)
        self.assertEqual(round(e["naive"], 2), 6.42)
        self.assertEqual(round(e["selection_bias"], 2), 4.34)
        # 분해 항등식은 부동소수점 오차 수준(소수 9자리)까지 맞는다
        self.assertAlmostEqual(e["naive"], e["att"] + e["selection_bias"], places=9)
        self.assertGreater(e["naive"], 2 * e["ate"])

    def test_randomization_removes_selection_bias(self):
        e = po.estimates(self.pop, po.randomized(self.pop))
        self.assertEqual(round(e["naive"], 2), 2.49)
        self.assertEqual(round(e["selection_bias"], 2), 0.01)
        self.assertAlmostEqual(e["naive"], e["ate"], delta=0.15)  # 표본 잡음 허용폭

    def test_itt_is_diluted_by_takeup(self):
        i = po.itt_demo(self.pop)
        # ITT 1.48 = 사용 비율 0.595 x 효과 약 2.49, 즉 배정 기준 효과는 희석된다
        self.assertEqual(round(i["itt"], 2), 1.48)
        self.assertEqual(round(i["takeup"], 3), 0.595)
        self.assertEqual(round(i["itt_over_takeup"], 2), 2.49)


if __name__ == "__main__":
    unittest.main()
