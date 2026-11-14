import math
import unittest
import inventory as inv


class InventoryTest(unittest.TestCase):
    def test_eoq_formula_matches_scan(self):
        self.assertAlmostEqual(inv.eoq(), 346.41, places=2)
        self.assertEqual(inv.eoq_by_scan(), 346)
        self.assertAlmostEqual(inv.eoq_cost(inv.eoq()), math.sqrt(2 * inv.D * inv.K * inv.H), places=3)
        self.assertAlmostEqual(inv.eoq_cost(inv.eoq()), 415_692.19, places=2)

    def test_eoq_cost_is_flat_near_optimum(self):
        self.assertAlmostEqual(inv.cost_ratio(450), 1.034, places=3)
        self.assertAlmostEqual(inv.cost_ratio(200), 1.155, places=3)
        self.assertAlmostEqual(inv.cost_ratio(100), 1.876, places=3)

    def test_newsvendor_ratio_and_quantity(self):
        self.assertAlmostEqual(inv.critical_ratio(), 0.25)
        self.assertAlmostEqual(inv.newsvendor_q(), 79.77, places=2)
        self.assertEqual(inv.best_q_by_scan(), 80)

    def test_mean_order_is_worse_when_costs_are_asymmetric(self):
        pmf = inv.demand_pmf()
        self.assertAlmostEqual(sum(pmf), 1.0)
        gap = inv.expected_profit(80, pmf) - inv.expected_profit(100, pmf)
        self.assertAlmostEqual(gap, 77_906, delta=1)
        self.assertGreater(gap, 0)

    def test_ratio_flips_when_stockout_is_expensive(self):
        self.assertGreater(inv.newsvendor_q(price=60_000), inv.MEAN)

    def test_safety_stock(self):
        self.assertAlmostEqual(inv.safety_stock(0.95), 69.79, places=2)
        self.assertAlmostEqual(inv.reorder_point(0.95), 269.79, places=2)

    def test_service_level_simulation_and_misestimated_sd(self):
        rop = inv.reorder_point(0.95)
        self.assertAlmostEqual(inv.simulated_cycle_service(rop), 0.95, delta=0.005)
        self.assertAlmostEqual(inv.analytic_cycle_service(rop, 45), 0.8636, places=3)
        self.assertAlmostEqual(inv.simulated_cycle_service(rop, 45), 0.8636, delta=0.01)


if __name__ == "__main__":
    unittest.main()
