import random
import unittest
from fractions import Fraction as F

from integer_programming import (A, B, C, branch_and_bound, brute_force,
                                 is_feasible, rounding_report, solve_lp,
                                 sweep_budget)


class IntegerProgrammingTest(unittest.TestCase):
    def test_lp_relaxation_is_fractional(self):
        val, x = solve_lp(C, A, B)
        self.assertEqual(x, (F(8, 5), F(11, 5)))
        self.assertEqual(val, F(84, 5))

    def test_hand_check_of_lp_vertex(self):
        # 두 제약이 모두 등호인 점: 4x+3y=13, x+2y=6
        x, y = F(8, 5), F(11, 5)
        self.assertEqual(4 * x + 3 * y, 13)
        self.assertEqual(x + 2 * y, 6)

    def test_nearest_rounding_breaks_constraint(self):
        rep = rounding_report(C, A, B)
        self.assertEqual(rep["nearest"], (2, 2))
        self.assertFalse(rep["nearest_feasible"])
        self.assertEqual(4 * 2 + 3 * 2, 14)  # 한도 13 초과

    def test_best_feasible_rounding_is_worse_than_optimum(self):
        rep = rounding_report(C, A, B)
        self.assertEqual(rep["best_feasible_rounding"], (13, (1, 2)))
        best, _ = brute_force(C, A, B)
        self.assertEqual(best, (15, (3, 0)))

    def test_branch_and_bound_matches_brute_force(self):
        r = branch_and_bound(C, A, B)
        self.assertEqual((r["best"], r["x"]), (15, (3, 0)))
        self.assertEqual(r["nodes"], 13)

    def test_count_of_feasible_integer_points(self):
        _, n = brute_force(C, A, B)
        self.assertEqual(n, 10)

    def test_gap_and_sweep(self):
        rows = {v: (lv, best) for v, lv, best, _, _ in sweep_budget(C, A, B, range(10, 17))}
        self.assertEqual(rows[13], (F(84, 5), 15))
        self.assertEqual(rows[14], (F(18), 18))     # 한도 14는 LP 해가 이미 정수
        self.assertEqual(rows[12][1], rows[13][1])   # 한도 +1에도 정수 최적은 그대로
        for v, (lv, best) in rows.items():
            self.assertGreaterEqual(lv, best)

    def test_random_instances_agree_with_brute_force(self):
        rng = random.Random(20261114)
        for _ in range(150):
            n = rng.choice((2, 3))
            c = tuple(rng.randint(1, 9) for _ in range(n))
            m = rng.choice((1, 2))
            a = tuple(tuple(rng.randint(1, 6) for _ in range(n)) for _ in range(m))
            b = tuple(rng.randint(5, 20) for _ in range(m))
            best, _ = brute_force(c, a, b, limit=20)
            r = branch_and_bound(c, a, b)
            self.assertEqual(r["best"], best[0], (c, a, b))
            self.assertTrue(is_feasible(r["x"], a, b))

    def test_deterministic(self):
        self.assertEqual(branch_and_bound(C, A, B), branch_and_bound(C, A, B))


if __name__ == "__main__":
    unittest.main()
