"""14장 테스트: 숨은 교란이 단순 차이를 부풀리는지, 편향 공식 gamma x delta로 진짜 효과를 되찾는지, 민감도 표와 breakeven 공식,
시간 위약은 평행추세 위반을 잡지만 공간 위약은 못 잡는다는 점, 신규 비중으로 다시 가중하면 옮긴 효과가 가까워지지만 숨은 요인은 못 잡는다는 점을 확인한다."""
import unittest

from robustness import (TRUE_EFFECT, breakeven_delta, effect, make_experiment, make_observational,
                        make_panel, naive_diff, omitted_variable_bias, placebo_report,
                        sensitivity_table, stratified_diff, transport, _mean)


class Ch14(unittest.TestCase):
    def setUp(self):
        self.rows = make_observational()

    def test_confounding_inflates_naive(self):
        self.assertGreater(naive_diff(self.rows), 3 * TRUE_EFFECT)  # 진짜 1500의 3배(4500) 넘게 부풀려진다

    def test_bias_formula_recovers_truth(self):
        strat = stratified_diff(self.rows)
        d = 0.0
        for v in (0, 1):
            sub = [r for r in self.rows if r["h"] == v]
            d += len(sub) / len(self.rows) * (_mean([r["u"] for r in sub if r["t"]])
                                              - _mean([r["u"] for r in sub if not r["t"]]))
        # U를 아는 시뮬레이션이라 델타를 직접 구해 공식을 검증한다. 오차 250원은 표본 잡음 허용폭
        self.assertLess(abs(strat - omitted_variable_bias(4000, d) - TRUE_EFFECT), 250)

    def test_table_and_breakeven(self):
        # 4588은 robustness.py를 실행하면 나오는 층화 추정값(원), 편향 4000 x 0.5 = 2000. 추정을 0으로 만드는 delta = 4000 / 4000 = 1.0
        tab = sensitivity_table(4588, (4000,), (0.5,))
        self.assertEqual(tab[(4000, 0.5)], 4588 - 2000)
        self.assertAlmostEqual(breakeven_delta(4000, 4000), 1.0)

    def test_time_placebo_catches_differential_trend(self):
        # ok: 평행추세 + 진짜 효과 10. bad: 효과 0인데 처치 지역만 가파른 추세(기울기 +3)
        ok = placebo_report(make_panel(10.0, 0.0))
        bad = placebo_report(make_panel(0.0, 3.0))
        self.assertLess(abs(ok["time_placebo"]), 1.5)  # 가짜 시점에서는 효과가 0 근처
        self.assertGreater(ok["time_rank"], 3)
        self.assertGreater(bad["time_placebo"], 3.0)  # 처치 전부터 격차가 벌어지고 있다는 경고
        self.assertEqual(bad["time_rank"], 1)

    def test_space_placebo_cannot_catch_it(self):
        bad = placebo_report(make_panel(0.0, 3.0))
        # 공간 위약은 도너들이 평행추세를 따를 때 처치 지역의 가파른 추세를 "큰 효과"로만 보여 줘 위반을 못 가린다
        self.assertEqual(bad["space_rank"], 1)   # 효과가 0인데도 가장 크게 보인다
        self.assertGreater(bad["did"], 7)

    def test_transport(self):
        # A 지역은 신규 60%, B 지역은 신규 20%. B의 진짜 평균 효과 = 0.2 x 3000 + 0.8 x 600 = 1080
        a = make_experiment(20000, 0.6, seed=21)
        truth_b = 0.2 * 3000 + 0.8 * 600
        self.assertGreater(effect(a) - truth_b, 800)                    # 그대로 옮기면 과대
        self.assertLess(abs(transport(a, 0.2) - truth_b), 250)          # 다시 가중하면 가깝다
        halved = 0.5 * truth_b
        # 숨은 요인으로 B의 효과가 절반이 되면 진짜 540인데, 신규 비중 재가중은 이를 모르므로 약 1080을 그대로 낸다
        self.assertGreater(transport(a, 0.2) - halved, 400)             # 숨은 요인은 못 잡는다

    def test_deterministic(self):
        self.assertEqual(naive_diff(make_observational()), naive_diff(make_observational()))


if __name__ == "__main__":
    unittest.main()
