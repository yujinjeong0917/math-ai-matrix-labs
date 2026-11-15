import math
import unittest
import inventory as inv


class InventoryTest(unittest.TestCase):
    def test_eoq_formula_matches_scan(self):
        # sqrt(2 x 2400 x 30000 / 1200) = sqrt(120000) = 346.41, 최소 비용은 sqrt(2DKh) = 415,692.19원
        self.assertAlmostEqual(inv.eoq(), 346.41, places=2)
        self.assertEqual(inv.eoq_by_scan(), 346)
        self.assertAlmostEqual(inv.eoq_cost(inv.eoq()), math.sqrt(2 * inv.D * inv.K * inv.H), places=3)
        self.assertAlmostEqual(inv.eoq_cost(inv.eoq()), 415_692.19, places=2)

    def test_eoq_cost_is_flat_near_optimum(self):
        # 최적에서 30% 벗어난 Q=450도 비용은 3.4%만 늘어난다(최적 근처가 평평하다)
        self.assertAlmostEqual(inv.cost_ratio(450), 1.034, places=3)
        self.assertAlmostEqual(inv.cost_ratio(200), 1.155, places=3)
        self.assertAlmostEqual(inv.cost_ratio(100), 1.876, places=3)

    def test_newsvendor_ratio_and_quantity(self):
        # Cu = 36,000 - 28,000 = 8,000, Co = 28,000 - 4,000 = 24,000 -> 0.25. 하위 25% 분위수 = 79.77개
        self.assertAlmostEqual(inv.critical_ratio(), 0.25)
        self.assertAlmostEqual(inv.newsvendor_q(), 79.77, places=2)
        self.assertEqual(inv.best_q_by_scan(), 80)

    def test_mean_order_is_worse_when_costs_are_asymmetric(self):
        pmf = inv.demand_pmf()
        self.assertAlmostEqual(sum(pmf), 1.0)
        gap = inv.expected_profit(80, pmf) - inv.expected_profit(100, pmf)
        # 평균(100)대로 발주하면 남는 재고 손해(개당 24,000원)가 커서 80개보다 기대이익이 약 77,906원 낮다(delta=1은 반올림 허용)
        self.assertAlmostEqual(gap, 77_906, delta=1)
        self.assertGreater(gap, 0)

    def test_ratio_flips_when_stockout_is_expensive(self):
        # 판매가 60,000원이면 Cu가 커져 임계비율이 0.5를 넘고, 발주량이 평균 100을 넘는다(비대칭 비용이 방향을 뒤집는다)
        self.assertGreater(inv.newsvendor_q(price=60_000), inv.MEAN)

    def test_safety_stock(self):
        # 1.645 x 30 x sqrt(2) = 69.79, 재주문점 = 200 + 69.79
        self.assertAlmostEqual(inv.safety_stock(0.95), 69.79, places=2)
        self.assertAlmostEqual(inv.reorder_point(0.95), 269.79, places=2)

    def test_service_level_simulation_and_misestimated_sd(self):
        rop = inv.reorder_point(0.95)
        # 10만 주기 시뮬레이션이 목표 95%에 0.5%p 이내. 시드 고정이므로 값은 재현된다
        self.assertAlmostEqual(inv.simulated_cycle_service(rop), 0.95, delta=0.005)
        # 실제 표준편차가 45인데 30으로 계산해 둔 재주문점이면 달성률은 95%가 아니라 86.4%
        self.assertAlmostEqual(inv.analytic_cycle_service(rop, 45), 0.8636, places=3)
        self.assertAlmostEqual(inv.simulated_cycle_service(rop, 45), 0.8636, delta=0.01)


if __name__ == "__main__":
    unittest.main()
