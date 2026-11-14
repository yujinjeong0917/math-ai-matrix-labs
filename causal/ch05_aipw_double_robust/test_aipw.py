import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aipw as a


class Hand(unittest.TestCase):
    def test_wrong_outcome_model_is_corrected(self):
        psi, avg = a.hand_aipw(60.0, 50.0, 0.5)
        self.assertEqual(psi, [6.0, 10.0, 10.0, 6.0])
        self.assertEqual(avg, 8.0)

    def test_right_outcome_model_ignores_wrong_probability(self):
        for e in (0.5, 0.2):
            psi, avg = a.hand_aipw(59.0, 51.0, e)
            self.assertAlmostEqual(avg, 8.0)
        for got, want in zip(a.hand_aipw(59.0, 51.0, 0.2)[0], [3.0, 13.0, 9.25, 6.75]):
            self.assertAlmostEqual(got, want)


class SingleRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = a.make_data(2000, 3)

    def test_naive_is_far_off(self):
        self.assertEqual(sum(r["t"] for r in self.rows), 738)
        self.assertEqual(round(a.naive_diff(self.rows), 2), 18.94)

    def test_four_combinations(self):
        want = {
            "둘 다 맞음": (8.53, 9.77, 8.47),
            "결과모형만 틀림": (17.90, 9.77, 9.83),
            "성향점수모형만 틀림": (8.53, 17.94, 8.53),
            "둘 다 틀림": (17.90, 17.94, 18.38),
        }
        for name, od, pd in a.COMBOS:
            got = tuple(round(v, 2) for v in a.run_combo(self.rows, od, pd))
            self.assertEqual(got, want[name], name)

    def test_aipw_reduces_to_regression_plus_correction(self):
        # AIPW = 결과모형만 + 보정항. 결과모형이 맞으면 보정항 평균이 0에 가깝다.
        reg, _, aipw = a.run_combo(self.rows, 2, 2)
        self.assertLess(abs(aipw - reg), 0.2)


class Replication(unittest.TestCase):
    def test_double_robustness_over_100_runs(self):
        tab = a.replicate(100, 1000, 500)
        want = {
            "둘 다 맞음": (0.02, 0.23, 0.01, 0.44),
            "결과모형만 틀림": (9.56, 0.23, 0.05, 4.37),
            "성향점수모형만 틀림": (0.02, 9.71, 0.02, 0.34),
            "둘 다 틀림": (9.56, 9.71, 10.59, 1.10),
        }
        for name, (reg, ipw, aipw) in tab.items():
            got = (round(a.mean(reg) - a.TAU, 2), round(a.mean(ipw) - a.TAU, 2),
                   round(a.mean(aipw) - a.TAU, 2), round(a.sd(aipw), 2))
            self.assertEqual(got, want[name], name)


class CrossFit(unittest.TestCase):
    def test_in_sample_one_nearest_residual_is_zero(self):
        rows = a.make_data(400, 900)
        treated = [r for r in rows if r["t"] == 1]
        m1 = a.knn_model(treated, 1)
        self.assertEqual(max(abs(r["y"] - m1(r["x"])) for r in treated), 0.0)

    def test_cross_fit_lowers_bias_but_not_spread(self):
        ins, cf = a.cross_fit_runs()
        self.assertEqual((round(a.mean(ins) - a.TAU, 2), round(a.sd(ins), 2), round(a.rmse(ins), 2)), (0.98, 0.80, 1.26))
        self.assertEqual((round(a.mean(cf) - a.TAU, 2), round(a.sd(cf), 2), round(a.rmse(cf), 2)), (0.36, 1.61, 1.65))


if __name__ == "__main__":
    unittest.main()
