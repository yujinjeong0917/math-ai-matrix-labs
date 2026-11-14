import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propensity as p

KW_POOR = dict(slope=4.0, intercept=-5.0, tau0=8.0, tau1=4.0)


class HandTable(unittest.TestCase):
    def test_naive_gap_is_much_larger_than_effect(self):
        m1, m0, d = p.hand_naive()
        self.assertEqual((round(m1, 1), round(m0, 1), round(d, 1)), (63.6, 36.4, 27.2))

    def test_stratified_and_ipw_recover_eight(self):
        self.assertAlmostEqual(p.hand_stratified(), 8.0)
        a, b, c = p.hand_ipw()
        self.assertEqual((round(a, 1), round(b, 1), round(c, 1)), (54.0, 46.0, 8.0))


class GoodOverlap(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_data(2000, 7)
        self.ps, self.beta = p.propensity_scores(self.rows)

    def test_data_is_deterministic(self):
        self.assertEqual(sum(r["t"] for r in self.rows), 940)

    def test_logit_coefficients(self):
        self.assertEqual((round(self.beta[0], 2), round(self.beta[1], 2)), (-0.17, 0.95))

    def test_naive_is_biased_ipw_and_matching_are_close(self):
        self.assertEqual(round(p.naive_diff(self.rows), 2), 15.67)
        self.assertEqual(round(p.ipw_ate(self.rows, self.ps), 2), 7.97)
        att, n_t, n_c, _ = p.match_att(self.rows, self.ps)
        self.assertEqual((round(att, 2), n_t, n_c), (7.93, 940, 465))

    def test_weighting_balances_x(self):
        w = p.weights(self.rows, self.ps)
        self.assertEqual(round(p.smd(self.rows), 2), 0.87)
        self.assertEqual(round(p.smd(self.rows, w), 2), 0.03)
        self.assertEqual(round(max(w), 1), 12.6)
        self.assertEqual(round(p.effective_sample_size(w)), 1599)


class PoorOverlap(unittest.TestCase):
    def setUp(self):
        self.rows = p.make_data(2000, 7, **KW_POOR)
        self.ps, _ = p.propensity_scores(self.rows)

    def test_ipw_ate_is_far_from_truth(self):
        self.assertEqual(round(p.true_ate(self.rows), 2), 8.11)
        self.assertEqual(round(p.ipw_ate(self.rows, self.ps), 2), 19.52)

    def test_matching_att_and_trimming(self):
        self.assertEqual(round(p.true_att(self.rows), 2), 14.01)
        att, _, n_c, _ = p.match_att(self.rows, self.ps)
        self.assertEqual((round(att, 2), n_c), (15.19, 106))
        est, n, tau = p.trimmed_ipw(self.rows, self.ps)
        self.assertEqual((round(est, 2), n, round(tau, 2)), (11.18, 436, 12.43))

    def test_almost_no_treated_below_zero(self):
        low = [r for r in self.rows if r["x"] < 0]
        self.assertEqual((len(low), sum(r["t"] for r in low)), (985, 1))

    def test_treated_side_overrepresents_high_activity(self):
        w = p.weights(self.rows, self.ps)
        wt = [(r, wi) for r, wi in zip(self.rows, w) if r["t"] == 1]
        wc = [(r, wi) for r, wi in zip(self.rows, w) if r["t"] == 0]
        self.assertEqual((round(p.weighted_mean_x(wt), 2), round(p.weighted_mean_x(wc), 2)), (0.72, 0.02))
        self.assertEqual(round(p.mean([r["x"] for r in self.rows]), 2), 0.03)
        top = max(wt, key=lambda a: a[1])
        self.assertEqual((round(top[0]["x"], 2), round(top[1] / sum(wi for _, wi in wt), 2)), (-0.14, 0.21))

    def test_weights_explode(self):
        w = p.weights(self.rows, self.ps)
        self.assertEqual(round(max(w), 1), 232.8)
        self.assertEqual(round(p.effective_sample_size(w)), 139)


class Replication(unittest.TestCase):
    def test_summary_over_200_runs(self):
        for kw, bias, spread in (({}, 0.04, 0.38), (KW_POOR, 7.18, 4.16)):
            res = p.replicate(200, 2000, 100, **kw)
            err = [a[1] - a[2] for a in res]
            self.assertEqual(round(p.mean(err), 2), bias)
            self.assertEqual(round(p.sd([a[1] for a in res]), 2), spread)


if __name__ == "__main__":
    unittest.main()
