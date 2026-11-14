import random
import unittest

from transport import (COST, DEMAND, SUPPLY, brute_force, greedy_in_order,
                       nearest_only, plan_cost, plan_ok, solve_transport)


class TransportTest(unittest.TestCase):
    def test_nearest_warehouse_breaks_capacity(self):
        plan, load, cost = nearest_only(SUPPLY, DEMAND, COST)
        self.assertEqual(load["W2"], 9)
        self.assertGreater(load["W2"], SUPPLY["W2"])
        self.assertEqual(cost, 25)
        self.assertFalse(plan_ok(plan, SUPPLY, DEMAND, COST))

    def test_ordered_greedy_is_feasible_but_worse(self):
        plan, cost = greedy_in_order(SUPPLY, DEMAND, COST)
        self.assertTrue(plan_ok(plan, SUPPLY, DEMAND, COST))
        self.assertEqual(cost, 40)

    def test_min_cost_flow_matches_brute_force(self):
        res = solve_transport(SUPPLY, DEMAND, COST)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["cost"], 34)
        self.assertTrue(plan_ok(res["plan"], SUPPLY, DEMAND, COST))
        self.assertEqual(plan_cost(res["plan"], COST), 34)
        self.assertEqual(brute_force(SUPPLY, DEMAND, COST), (34, 1, 63))

    def test_forbidden_pair_never_used(self):
        res = solve_transport(SUPPLY, DEMAND, COST)
        self.assertNotIn(("W3", "C"), res["plan"])

    def test_extra_capacity_saves_two_per_unit(self):
        costs = [solve_transport(dict(SUPPLY, W2=c), DEMAND, COST)["cost"] for c in range(3, 7)]
        self.assertEqual(costs, [34, 32, 30, 28])

    def test_infeasible_case_is_reported(self):
        s = {"W1": 4, "W2": 3, "W3": 1}
        cost2 = {w: dict(COST[w]) for w in COST}
        cost2["W1"]["C"] = None
        cost2["W2"]["C"] = None
        cost2["W3"]["C"] = 6
        res = solve_transport(s, DEMAND, cost2)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["shipped"], 8)
        self.assertEqual(brute_force(s, DEMAND, cost2)[2], 0)

    def test_random_small_instances_agree(self):
        rng = random.Random(20261114)
        for _ in range(60):
            supply = {w: rng.randint(1, 4) for w in ("W1", "W2", "W3")}
            demand = {r: rng.randint(1, 3) for r in ("A", "B", "C")}
            cost = {w: {r: (None if rng.random() < 0.2 else rng.randint(1, 9)) for r in demand} for w in supply}
            res = solve_transport(supply, demand, cost)
            best, _, n_ok = brute_force(supply, demand, cost)
            self.assertEqual(res["feasible"], n_ok > 0)
            if n_ok:
                self.assertEqual(res["cost"], best)


if __name__ == "__main__":
    unittest.main()
