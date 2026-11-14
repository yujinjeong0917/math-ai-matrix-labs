import unittest

from robustness import (TRUE_EFFECT, breakeven_delta, effect, make_experiment, make_observational,
                        make_panel, naive_diff, omitted_variable_bias, placebo_report,
                        sensitivity_table, stratified_diff, transport, _mean)


class Ch14(unittest.TestCase):
    def setUp(self):
        self.rows = make_observational()

    def test_confounding_inflates_naive(self):
        self.assertGreater(naive_diff(self.rows), 3 * TRUE_EFFECT)

    def test_bias_formula_recovers_truth(self):
        strat = stratified_diff(self.rows)
        d = 0.0
        for v in (0, 1):
            sub = [r for r in self.rows if r["h"] == v]
            d += len(sub) / len(self.rows) * (_mean([r["u"] for r in sub if r["t"]])
                                              - _mean([r["u"] for r in sub if not r["t"]]))
        self.assertLess(abs(strat - omitted_variable_bias(4000, d) - TRUE_EFFECT), 250)

    def test_table_and_breakeven(self):
        tab = sensitivity_table(4588, (4000,), (0.5,))
        self.assertEqual(tab[(4000, 0.5)], 4588 - 2000)
        self.assertAlmostEqual(breakeven_delta(4000, 4000), 1.0)

    def test_time_placebo_catches_differential_trend(self):
        ok = placebo_report(make_panel(10.0, 0.0))
        bad = placebo_report(make_panel(0.0, 3.0))
        self.assertLess(abs(ok["time_placebo"]), 1.5)
        self.assertGreater(ok["time_rank"], 3)
        self.assertGreater(bad["time_placebo"], 3.0)
        self.assertEqual(bad["time_rank"], 1)

    def test_space_placebo_cannot_catch_it(self):
        bad = placebo_report(make_panel(0.0, 3.0))
        self.assertEqual(bad["space_rank"], 1)   # 효과가 0인데도 가장 크게 보인다
        self.assertGreater(bad["did"], 7)

    def test_transport(self):
        a = make_experiment(20000, 0.6, seed=21)
        truth_b = 0.2 * 3000 + 0.8 * 600
        self.assertGreater(effect(a) - truth_b, 800)                    # 그대로 옮기면 과대
        self.assertLess(abs(transport(a, 0.2) - truth_b), 250)          # 다시 가중하면 가깝다
        halved = 0.5 * truth_b
        self.assertGreater(transport(a, 0.2) - halved, 400)             # 숨은 요인은 못 잡는다

    def test_deterministic(self):
        self.assertEqual(naive_diff(make_observational()), naive_diff(make_observational()))


if __name__ == "__main__":
    unittest.main()
