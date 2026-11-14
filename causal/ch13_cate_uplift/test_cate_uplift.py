import unittest

from cate_uplift import (COST, diff_means, make_customers, policy_gain, subgroup_experiment,
                         t_learner_cells, treat_all_gain, true_tau, z_for_two_sided)


class Ch13(unittest.TestCase):
    def setUp(self):
        self.rows = make_customers(20000, seed=2)
        half = len(self.rows) // 2
        self.train, self.hold = self.rows[:half], self.rows[half:]

    def test_deterministic(self):
        self.assertEqual(diff_means(self.rows), diff_means(make_customers(20000, seed=2)))

    def test_ate_within_three_se(self):
        est, se = diff_means(self.rows)
        truth = sum(true_tau(r["new"], r["heavy"]) for r in self.rows) / len(self.rows)
        self.assertLess(abs(est - truth), 3 * se)

    def test_prespecified_segments_differ(self):
        new, _ = diff_means([r for r in self.rows if r["new"] == 1])
        old, _ = diff_means([r for r in self.rows if r["new"] == 0])
        self.assertGreater(new - old, 2000)

    def test_uplift_policy_beats_treat_all(self):
        cells = t_learner_cells(self.train)
        self.assertGreater(policy_gain(cells, self.hold), treat_all_gain(self.hold) + 300)
        self.assertGreater(policy_gain(cells, self.hold), 0)

    def test_cost_value(self):
        self.assertEqual(COST, 1500)

    def test_post_hoc_subgroup_trap(self):
        res = subgroup_experiment()
        self.assertAlmostEqual(res["z05"], 1.96, places=2)
        self.assertGreater(res["share_best_sig"], 0.4)
        self.assertLess(res["share_best_sig_bonf"], 0.10)
        self.assertLess(abs(res["holdout_mean_est"]), 100)
        self.assertLess(res["holdout_share_sig"], 0.12)

    def test_z_inverse(self):
        self.assertAlmostEqual(z_for_two_sided(0.05), 1.96, places=2)


if __name__ == "__main__":
    unittest.main()
