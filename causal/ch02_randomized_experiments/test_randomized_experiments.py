"""2장 테스트: 층화의 SD 감소, 군집 배정의 표준오차 과소평가, SRM 검정식과 버그 탐지, 비순응 추정량의 차이를 확인한다.
반올림한 기대값은 고정 시드에서 나온 값이고, assertAlmostEqual의 delta는 표본 잡음 허용폭이다."""
import math
import unittest

import randomized_experiments as r


class Stratification(unittest.TestCase):
    def test_both_unbiased_but_stratified_is_steadier(self):
        s = r.stratification_demo()
        self.assertEqual((round(s["simple_mean"], 2), round(s["simple_sd"], 2)), (2.03, 1.05))
        self.assertEqual((round(s["strat_mean"], 2), round(s["strat_sd"], 2)), (2.00, 0.50))
        self.assertAlmostEqual(s["simple_mean"], r.EFFECT, delta=0.15)
        self.assertAlmostEqual(s["strat_mean"], r.EFFECT, delta=0.15)
        # 층화 SD 0.50이 단순 배정 SD 1.05의 절반 안팎: 단골 비율이 실험마다 흔들리는 몫이 사라진다
        self.assertLess(s["strat_sd"], 0.6 * s["simple_sd"])


class Cluster(unittest.TestCase):
    def test_independent_se_is_too_small(self):
        c = r.cluster_demo()
        self.assertEqual(round(c["mean_estimate"], 2), 2.02)
        self.assertEqual(round(c["actual_sd"], 2), 0.97)
        self.assertEqual(round(c["naive_se"], 2), 0.26)
        self.assertEqual(round(c["naive_coverage"], 2), 0.39)
        self.assertEqual(round(c["design_effect"], 1), 14.0)
        # 실제 흔들림 / 독립 가정 SE 가 설계 효과의 제곱근과 비슷하다 (0.97 / 0.26 = 3.7, sqrt(14.0) = 3.7)
        self.assertAlmostEqual(c["actual_sd"] / c["naive_se"], math.sqrt(c["design_effect"]), delta=0.4)
        self.assertLess(c["naive_coverage"], 0.6)  # 95%여야 할 포함률이 39%로 떨어진다


class Srm(unittest.TestCase):
    def test_pvalue_formula(self):
        # 손계산: 기대 10000, chi2 = 60^2/10000 + 60^2/10000 = 0.72
        chi2, p = r.srm_pvalue(10060, 9940)
        self.assertAlmostEqual(chi2, 0.72, places=2)
        self.assertAlmostEqual(p, 0.396, places=3)
        # 기대 10000, 편차 200 -> chi2 = 8.0, 자유도 1에서 p는 약 0.0047
        chi2, p = r.srm_pvalue(9800, 10200)
        self.assertAlmostEqual(chi2, 8.0, places=6)
        self.assertAlmostEqual(p, 0.0047, places=4)

    def test_clean_run_passes(self):
        x = r.srm_demo(bug=False)
        # 버그 없는 실행: 인원이 거의 1:1이라 chi2가 작다
        self.assertEqual((x["n_treat"], x["n_control"]), (9983, 10017))
        self.assertEqual(round(x["chi2"], 2), 0.06)
        self.assertGreater(x["p"], 0.05)
        self.assertAlmostEqual(x["estimate"], x["true_effect"], delta=0.2)

    def test_bug_is_flagged_and_biases_estimate(self):
        x = r.srm_demo(bug=True)
        # 처치군 약 12% 감소: 10000 x 0.2 x 0.6 = 1200명이 사라진다. 인원 불일치가 편향의 경고등이다
        self.assertEqual((x["n_treat"], x["n_control"]), (8780, 10008))
        self.assertEqual(round(x["chi2"], 2), 80.26)
        self.assertLess(x["p"], 1e-10)
        self.assertEqual(round(x["estimate"], 2), 1.86)
        self.assertGreater(x["estimate"] - x["true_effect"], 0.5)  # 진짜 1.0보다 0.86 높게 나온다


class Noncompliance(unittest.TestCase):
    def test_itt_wald_and_per_protocol(self):
        n = r.noncompliance_demo()
        # 왈드(2.68)가 사용자의 진짜 효과(2.57)에 가깝고, 사용자-대조 직접 비교(4.32)는 크게 벗어난다
        self.assertEqual(round(n["itt"], 2), 1.48)
        self.assertEqual(round(n["share_used"], 3), 0.551)
        self.assertEqual(round(n["wald"], 2), 2.68)
        self.assertEqual(round(n["per_protocol"], 2), 4.32)
        self.assertEqual(round(n["true_complier_effect"], 2), 2.57)
        self.assertAlmostEqual(n["wald"], n["true_complier_effect"], delta=0.3)
        self.assertGreater(n["per_protocol"] - n["true_complier_effect"], 1.0)


if __name__ == "__main__":
    unittest.main()
