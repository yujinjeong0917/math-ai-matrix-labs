import unittest

from robust_stock import (FORECAST, draw_scenarios, evaluate, overfit_gap, plan_for_scenarios,
                          profit, regret_plan, robust_plan, shock_set, shock_sweep)


class RobustStockTest(unittest.TestCase):
    def test_forecast_plan_by_hand(self):
        # A 마진이 더 크니 예측 수요 60을 먼저 채우고 나머지 40을 B에 둔다.
        self.assertEqual(plan_for_scenarios([FORECAST]), 60)
        self.assertEqual(profit(60, FORECAST), 30 * 60 + 24 * 40)

    def test_robust_plan_matches_line_intersection(self):
        # A만 낮아질 때 3450-24x, B만 낮아질 때 30x+600. 두 직선이 만나는 x=2850/54=52.78, 정수로는 53.
        self.assertEqual(robust_plan(shock_set()), 53)
        sc = shock_set()
        self.assertEqual(min(profit(53, d) for d in sc), 2178)
        self.assertEqual(min(profit(52, d) for d in sc), 2160)
        self.assertEqual(min(profit(54, d) for d in sc), 2154)

    def test_robust_costs_a_little_at_forecast_and_gains_in_worst_case(self):
        sc = shock_set()
        self.assertEqual(profit(60, FORECAST) - profit(53, FORECAST), 42)
        self.assertEqual(min(profit(53, d) for d in sc) - min(profit(60, d) for d in sc), 168)

    def test_regret_plan(self):
        self.assertEqual(regret_plan(shock_set()), 56)

    def test_deterministic(self):
        self.assertEqual(draw_scenarios(5, 1), draw_scenarios(5, 1))
        t = draw_scenarios(2000, 777)
        self.assertAlmostEqual(evaluate(60, t)["mean"], 2439.7, places=1)
        self.assertAlmostEqual(evaluate(53, t)["mean"], 2419.1, places=1)
        self.assertEqual(evaluate(53, t)["p5"], 1644)
        self.assertEqual(evaluate(60, t)["p5"], 1620)

    def test_overfit_gap_shrinks_with_more_scenarios(self):
        gaps = []
        for n in (3, 5, 20, 200):
            a, b = overfit_gap(n)
            gaps.append(a - b)
        self.assertAlmostEqual(gaps[0], 95.3, places=1)
        self.assertAlmostEqual(gaps[1], 43.4, places=1)
        self.assertGreater(gaps[0], gaps[1])
        self.assertGreater(gaps[1], gaps[2])

    def test_outside_the_set_is_not_protected(self):
        rows = {k: r for k, *r in shock_sweep()}
        self.assertEqual(rows[0][0], 60)
        self.assertEqual(rows[25][0], 53)
        self.assertEqual(rows[25][3], 1578)   # 집합 밖 (15, 50)
        self.assertLess(rows[25][3], rows[25][2])

    def test_width_60_clips_demand_at_zero_and_breaks_tie_by_forecast_profit(self):
        rows = {k: r for k, *r in shock_sweep()}
        self.assertEqual(rows[60], [50, 2700, 1200, 1650])
        self.assertEqual(shock_set(shock=60), [(60, 50), (0, 50), (60, 0)])
        for x in range(40, 51):
            self.assertEqual(min(profit(x, d) for d in shock_set(shock=60)), 1200)

    def test_same_worst_sample_for_three_plans(self):
        t = draw_scenarios(2000, 777)
        self.assertEqual([evaluate(x, t)["min"] for x in (60, 53, 56)], [558, 558, 558])

    def test_infeasible_allocation_fails(self):
        with self.assertRaises(ValueError):
            profit(101, FORECAST)


if __name__ == "__main__":
    unittest.main()
