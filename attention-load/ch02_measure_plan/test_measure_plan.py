import unittest

from measure_plan import cronbach_alpha, load, plan_flags, ranks, spearman


class Plan(unittest.TestCase):
    def setUp(self):
        self.d = load()

    def test_alpha_hand(self):
        # 손계산: 항목 분산 2.5 + 0.7 + 3.3 = 6.5, 총점 분산 17.3, 1.5 * (1 - 6.5/17.3) = 0.936
        sv, tv, a = cronbach_alpha(self.d["responses"])
        self.assertAlmostEqual(sv, 6.5)
        self.assertAlmostEqual(tv, 17.3)
        self.assertAlmostEqual(a, 0.936, places=3)

    def test_spearman_hand(self):
        totals = [sum(r) for r in self.d["responses"]]
        # 자료 A는 순위가 같아 1, 자료 B는 d^2 합 20이라 1 - 6*20/120 = 0
        self.assertAlmostEqual(spearman(totals, self.d["seconds_a"]), 1.0)
        self.assertAlmostEqual(spearman(totals, self.d["seconds_b"]), 0.0)

    def test_ranks_ties(self):
        self.assertEqual(ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])

    def test_plan_flags(self):
        self.assertEqual(len(plan_flags(self.d["plans"]["A"])), 2)
        self.assertEqual(plan_flags(self.d["plans"]["B"]), [])


if __name__ == "__main__":
    unittest.main()
