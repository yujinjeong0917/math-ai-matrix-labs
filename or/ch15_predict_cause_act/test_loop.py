import unittest

from loop import (BUDGET, POOLS, REVENUE, experiment, measure_plan, observe, plan_by,
                  predicted_vs_actual, spend, true_incremental)


class LoopTest(unittest.TestCase):
    def test_hand_arithmetic(self):
        # 이 파일의 숫자는 시드 고정 시뮬레이션 결과라 README 기대 출력과 같다. 난수 구현이 바뀌면 달라질 수 있다
        # 참 증분 ROAS = lift * 매출 / 비용. A 0.02*30=0.6, B 0.05*30=1.5.
        self.assertAlmostEqual(POOLS["A"]["lift"] * REVENUE, 0.6)
        self.assertAlmostEqual(POOLS["B"]["lift"] * REVENUE, 1.5)
        # 관찰 ROAS 순서 계획의 참 증분 매출: 30000*0.6 + 20000*1.5 = 48000
        self.assertAlmostEqual(true_incremental({"A": 30000, "B": 20000}), 48000)
        self.assertAlmostEqual(true_incremental({"A": 0, "B": 50000}), 75000)

    def test_observed_roas_ranks_a_first_but_incremental_ranks_b_first(self):
        oa, ob = observe("A", 101), observe("B", 101)
        # 관찰 ROAS는 A(10.18)가 B(2.72)보다 높지만, 무작위 대조로 잰 증분 ROAS는 A 0.77 < B 1.52로 순서가 뒤집힌다
        self.assertAlmostEqual(oa["obs_roas"], 10.18, places=2)
        self.assertAlmostEqual(ob["obs_roas"], 2.72, places=2)
        self.assertGreater(oa["obs_roas"], ob["obs_roas"])
        ea, eb = experiment("A", 303), experiment("B", 303)
        self.assertAlmostEqual(ea["iroas"], 0.77, places=2)
        self.assertAlmostEqual(eb["iroas"], 1.52, places=2)
        self.assertGreater(eb["iroas"], ea["iroas"])

    def test_naive_difference_is_biased_when_targeting_follows_intent(self):
        oa = observe("A", 101)
        # 노출군-미노출군 단순 차이 0.0995는 참 lift 0.02의 약 5배: 선택 편향
        self.assertAlmostEqual(oa["naive_lift"], 0.0995, places=4)
        self.assertGreater(oa["naive_lift"], 3 * POOLS["A"]["lift"])

    def test_prediction_is_accurate_yet_silent_about_effect(self):
        for k in POOLS:
            fit, new = predicted_vs_actual(k)
            # 구매율 예측 오차 1%p 미만: 예측이 정확해도 어느 채널에 쓸지는 알려주지 않는다
            self.assertLess(abs(fit - new), 0.01)

    def test_plans(self):
        # 관찰 ROAS 순서: A에 30000(전체) + 남은 20000을 B에. 증분 ROAS 순서: A(0.77)는 1 미만이라 보류하고 B에 50000
        plan_obs = plan_by(POOLS, {"A": 10.18, "B": 2.72})
        plan_inc = plan_by(POOLS, {"A": 0.77, "B": 1.52}, min_score=1.0)
        self.assertEqual(plan_obs, {"A": 30000, "B": 20000})
        self.assertEqual(plan_inc, {"A": 0, "B": 50000})
        self.assertEqual(spend(plan_obs), BUDGET)
        self.assertEqual(spend(plan_inc), BUDGET)

    def test_interval_for_a_straddles_break_even(self):
        # A의 증분 ROAS 구간 [0.43, 1.11]이 손익분기 1을 가로지른다: 점추정 0.77만 보고 "손해"로 단정할 수 없다
        ea = experiment("A", 303)
        self.assertTrue(ea["iroas_lo"] < 1.0 < ea["iroas_hi"])
        # B는 구간 하한도 1보다 커서 손익분기 이상이 통계적으로 분명하다
        eb = experiment("B", 303)
        self.assertGreater(eb["iroas_lo"], 1.0)

    def test_measurement_covers_truth(self):
        est, lo, hi = measure_plan({"A": 0, "B": 50000}, 404)
        # 다음 달 측정 78089천원은 참값 75000을 구간 [68914, 87264]이 덮는다(delta=1은 반올림 허용)
        self.assertAlmostEqual(est, 78089, delta=1)
        self.assertTrue(lo < true_incremental({"A": 0, "B": 50000}) < hi)

    def test_deterministic(self):
        self.assertEqual(experiment("B", 303), experiment("B", 303))


if __name__ == "__main__":
    unittest.main()
