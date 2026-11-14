import unittest

from pareto_front import (all_plans, days, dominates, duplicate_points, eps_constraint, eps_reaches_all, objectives, pareto_front,
                          weighted_pick, weights_that_reach)


class ParetoFrontTest(unittest.TestCase):
    def setUp(self):
        self.plans = all_plans()
        self.front = pareto_front(self.plans)

    def test_counts(self):
        self.assertEqual(len(self.plans), 343)
        self.assertEqual(len(self.front), 15)

    def test_objectives_by_hand(self):
        # C에 2명: 비용 3*2=6, C 배송일 6*0.7^2=2.94, 다른 지역은 3.0(B)이 가장 느리다.
        self.assertEqual(objectives((0, 0, 2)), (6.0, 3.0))
        self.assertAlmostEqual(days("C", 2), 2.94)

    def test_nobody_on_front_is_dominated(self):
        pts = [(c, d) for c, d, _ in self.front]
        for p in pts:
            self.assertFalse(any(dominates(q, p) for q in pts))
        # 전선 밖 점은 전선의 누군가에게 지배당한다.
        for _, obj in self.plans:
            if obj not in pts:
                self.assertTrue(any(dominates(q, obj) for q in pts))

    def test_weighted_sum_misses_seven_front_points(self):
        reach, missed = weights_that_reach(self.front, plans=self.plans)
        self.assertEqual(len(reach), 8)
        self.assertEqual(len(missed), 7)
        self.assertIn((8.0, 2.94), missed)

    def test_break_even_weights_by_hand(self):
        # (0,0,0)->(0,0,1): 비용 3 늘고 1.8일 줄어 0.6. (6,3.0)->(11,2.1): 0.9/5=0.18.
        self.assertAlmostEqual((6.0 - 4.2) / 3, 0.6)
        self.assertAlmostEqual((3.0 - 2.1) / 5, 0.18)
        self.assertEqual(weighted_pick(0.3)[1], (6.0, 3.0))
        self.assertEqual(weighted_pick(0.7)[1], (0.0, 6.0))
        # (8, 2.94)는 (6,3.0)과 (11,2.1)을 잇는 직선 위(2.64)보다 높다.
        self.assertAlmostEqual(3.0 - 0.18 * (8 - 6), 2.64)

    def test_epsilon_constraint_reaches_missed_points(self):
        a, o = eps_constraint(2.0, self.plans)
        self.assertEqual((a, o[0]), ((0, 2, 4), 16.0))
        a, o = eps_constraint(1.5, self.plans)
        self.assertEqual((a, o[0]), ((1, 2, 4), 17.0))
        self.assertIsNone(eps_constraint(0.1, self.plans))

    def test_epsilon_constraint_reaches_all_fifteen(self):
        self.assertEqual(eps_reaches_all(self.front, self.plans), [True] * 15)

    def test_duplicates_exist_so_front_counts_points_not_plans(self):
        self.assertGreater(duplicate_points(self.plans), 0)

    def test_margin_between_1_5_and_1_0_cap(self):
        a, o15 = eps_constraint(1.5, self.plans)
        b, o10 = eps_constraint(1.0, self.plans)
        self.assertAlmostEqual(o15[1] - o10[1], 0.49, places=2)
        self.assertEqual(o10[0] - o15[0], 11.0)


if __name__ == "__main__":
    unittest.main()
