import unittest

import dag_adjustment as d


class Hand(unittest.TestCase):
    def test_stratified_hand_numbers(self):
        h = d.hand_stratified()
        self.assertAlmostEqual(h["treated_mean"], 29.4286, places=4)
        self.assertEqual(h["control_mean"], 18.0)
        self.assertAlmostEqual(h["naive"], 11.4286, places=4)
        self.assertEqual(h["within"], {"높음": 2.0, "낮음": 2.0})
        self.assertEqual(h["weights"], {"높음": 0.7, "낮음": 0.3})
        self.assertAlmostEqual(h["adjusted"], 2.0, places=9)


class Ols(unittest.TestCase):
    def test_ols_recovers_exact_line(self):
        rows = [{"x": float(i), "y": 3.0 + 2.0 * i} for i in range(10)]
        b = d.ols(rows, "y", ["x"])
        self.assertAlmostEqual(b["1"], 3.0, places=9)
        self.assertAlmostEqual(b["x"], 2.0, places=9)


class Adjustment(unittest.TestCase):
    def test_observational_table(self):
        got = {k: round(v, 2) for k, v in d.adjustment_table()}
        self.assertEqual(got, {
            "아무것도 안 넣음": 5.00, "충성도 Z": 1.99, "Z + 장바구니 M": 0.52,
            "Z + 후기 점수 C": -0.51, "전부 다(Z, M, C)": -0.72})

    def test_only_confounder_adjustment_recovers_total_effect(self):
        got = dict(d.adjustment_table())
        self.assertAlmostEqual(got["충성도 Z"], d.TRUE_TOTAL, delta=0.1)
        self.assertGreater(got["아무것도 안 넣음"] - d.TRUE_TOTAL, 2.5)   # 교란을 안 넣으면 크게 틀린다
        self.assertLess(got["Z + 장바구니 M"], 1.0)                       # 매개를 넣으면 전체 효과가 사라진다
        self.assertLess(got["Z + 후기 점수 C"], 0.0)                      # 충돌변수를 넣으면 부호가 뒤집힌다
        self.assertLess(got["전부 다(Z, M, C)"], 0.0)

    def test_randomized_table_shows_bad_controls_still_hurt(self):
        got = {k: round(v, 2) for k, v in d.adjustment_table(randomize=True)}
        self.assertEqual(got, {
            "아무것도 안 넣음": 1.96, "충성도 Z": 1.99, "Z + 장바구니 M": 0.44,
            "Z + 후기 점수 C": -0.51, "전부 다(Z, M, C)": -0.72})


class Spread(unittest.TestCase):
    def test_seed_spread_explains_small_gaps(self):
        sp = d.seed_spread()
        m, s = sp[(True, "아무것도 안 넣음")]
        self.assertAlmostEqual(m, 2.0, delta=0.1)
        self.assertLess(s, 0.1)
        m, s = sp[(True, "Z + 장바구니 M")]
        self.assertAlmostEqual(m, d.DIRECT, delta=0.1)
        m, s = sp[(False, "Z + 장바구니 M")]
        self.assertAlmostEqual(m, d.DIRECT, delta=0.1)


if __name__ == "__main__":
    unittest.main()
