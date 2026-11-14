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
        self.assertLess(s["strat_sd"], 0.6 * s["simple_sd"])


class Cluster(unittest.TestCase):
    def test_independent_se_is_too_small(self):
        c = r.cluster_demo()
        self.assertEqual(round(c["mean_estimate"], 2), 2.02)
        self.assertEqual(round(c["actual_sd"], 2), 0.97)
        self.assertEqual(round(c["naive_se"], 2), 0.26)
        self.assertEqual(round(c["naive_coverage"], 2), 0.39)
        self.assertEqual(round(c["design_effect"], 1), 14.0)
        # 실제 흔들림 / 독립 가정 SE 가 설계 효과의 제곱근과 비슷하다
        self.assertAlmostEqual(c["actual_sd"] / c["naive_se"], math.sqrt(c["design_effect"]), delta=0.4)
        self.assertLess(c["naive_coverage"], 0.6)


class Srm(unittest.TestCase):
    def test_pvalue_formula(self):
        chi2, p = r.srm_pvalue(10060, 9940)
        self.assertAlmostEqual(chi2, 0.72, places=2)
        self.assertAlmostEqual(p, 0.396, places=3)
        chi2, p = r.srm_pvalue(9800, 10200)
        self.assertAlmostEqual(chi2, 8.0, places=6)
        self.assertAlmostEqual(p, 0.0047, places=4)

    def test_clean_run_passes(self):
        x = r.srm_demo(bug=False)
        self.assertEqual((x["n_treat"], x["n_control"]), (9983, 10017))
        self.assertEqual(round(x["chi2"], 2), 0.06)
        self.assertGreater(x["p"], 0.05)
        self.assertAlmostEqual(x["estimate"], x["true_effect"], delta=0.2)

    def test_bug_is_flagged_and_biases_estimate(self):
        x = r.srm_demo(bug=True)
        self.assertEqual((x["n_treat"], x["n_control"]), (8780, 10008))
        self.assertEqual(round(x["chi2"], 2), 80.26)
        self.assertLess(x["p"], 1e-10)
        self.assertEqual(round(x["estimate"], 2), 1.86)
        self.assertGreater(x["estimate"] - x["true_effect"], 0.5)


class Noncompliance(unittest.TestCase):
    def test_itt_wald_and_per_protocol(self):
        n = r.noncompliance_demo()
        self.assertEqual(round(n["itt"], 2), 1.48)
        self.assertEqual(round(n["share_used"], 3), 0.551)
        self.assertEqual(round(n["wald"], 2), 2.68)
        self.assertEqual(round(n["per_protocol"], 2), 4.32)
        self.assertEqual(round(n["true_complier_effect"], 2), 2.57)
        self.assertAlmostEqual(n["wald"], n["true_complier_effect"], delta=0.3)
        self.assertGreater(n["per_protocol"] - n["true_complier_effect"], 1.0)


if __name__ == "__main__":
    unittest.main()
