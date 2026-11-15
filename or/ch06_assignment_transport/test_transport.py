import random
import unittest

from transport import (COST, DEMAND, SUPPLY, brute_force, greedy_in_order,
                       nearest_only, plan_cost, plan_ok, solve_transport)


class TransportTest(unittest.TestCase):
    def test_nearest_warehouse_breaks_capacity(self):
        # 4*2 + 3*3 + 2*4 = 25. 하지만 W2에 4+3+2 = 9묶음이 몰려 용량 3을 넘는다
        plan, load, cost = nearest_only(SUPPLY, DEMAND, COST)
        self.assertEqual(load["W2"], 9)
        self.assertGreater(load["W2"], SUPPLY["W2"])
        self.assertEqual(cost, 25)
        self.assertFalse(plan_ok(plan, SUPPLY, DEMAND, COST))

    def test_ordered_greedy_is_feasible_but_worse(self):
        plan, cost = greedy_in_order(SUPPLY, DEMAND, COST)
        self.assertTrue(plan_ok(plan, SUPPLY, DEMAND, COST))
        self.assertEqual(cost, 40)   # 가능하지만 최적 34보다 6 비싸다

    def test_min_cost_flow_matches_brute_force(self):
        res = solve_transport(SUPPLY, DEMAND, COST)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["cost"], 34)
        self.assertTrue(plan_ok(res["plan"], SUPPLY, DEMAND, COST))
        self.assertEqual(plan_cost(res["plan"], COST), 34)
        # (최소비용, 최적 해 수, 가능한 계획 수). 흐름 해법과 전수 탐색이 서로 독립한 길로 같은 34
        self.assertEqual(brute_force(SUPPLY, DEMAND, COST), (34, 1, 63))

    def test_forbidden_pair_never_used(self):
        res = solve_transport(SUPPLY, DEMAND, COST)
        # W3->C는 COST가 None인 금지 조합: 간선 자체가 없다
        self.assertNotIn(("W3", "C"), res["plan"])

    def test_extra_capacity_saves_two_per_unit(self):
        # W2(C로 가는 가장 싼 쪽) 용량이 1 늘 때마다 비용이 2씩 준다: 34, 32, 30, 28
        costs = [solve_transport(dict(SUPPLY, W2=c), DEMAND, COST)["cost"] for c in range(3, 7)]
        self.assertEqual(costs, [34, 32, 30, 28])

    def test_infeasible_case_is_reported(self):
        # C로 갈 수 있는 곳이 W3(공급 1)뿐 -> C 2묶음 중 1묶음을 못 보내 총 8묶음
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
        # 고정 시드로 60개 무작위 소형 문제(금지 조합 약 20% 포함)에서 흐름 해법과 전수 탐색의 가능 여부·비용이 일치하는지 본다
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
