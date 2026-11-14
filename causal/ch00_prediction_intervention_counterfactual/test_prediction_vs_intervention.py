import unittest

import prediction_vs_intervention as p


class SceneA(unittest.TestCase):
    def test_observed_gap_is_large_but_randomized_gap_is_small(self):
        a = p.scene_a()
        self.assertEqual(round(a["observed_gap"], 2), 2.29)
        self.assertEqual(round(a["randomized_gap"], 2), 0.27)
        self.assertGreater(a["observed_gap"], 5 * a["true_effect"])
        self.assertAlmostEqual(a["randomized_gap"], a["true_effect"], delta=0.1)

    def test_deterministic(self):
        self.assertEqual(p.scene_a(), p.scene_a())


class SceneB(unittest.TestCase):
    def test_naive_before_after_swallows_the_season(self):
        b = p.scene_b(0.0)
        self.assertEqual(round(b["naive"], 2), 15.01)
        self.assertEqual(round(b["season_rise"], 2), 10.56)
        self.assertGreater(b["naive"], 2 * b["true_effect"])

    def test_seasonal_counterfactual_recovers_effect_when_only_season_moves(self):
        b = p.scene_b(0.0)
        self.assertEqual(round(b["seasonal_cf"], 2), 4.46)
        m = b["means"]
        self.assertEqual([round(m[k], 2) for k in ("this_pre", "this_post", "last_pre", "last_post")],
                         [112.83, 127.84, 102.43, 112.97])
        self.assertAlmostEqual(b["seasonal_cf"], b["true_effect"], delta=1.5)

    def test_simultaneous_event_breaks_the_seasonal_counterfactual(self):
        b = p.scene_b(12.0)
        self.assertEqual(round(b["naive"], 2), 27.01)
        self.assertEqual(round(b["seasonal_cf"], 2), 16.46)
        self.assertGreater(b["seasonal_cf"] - b["true_effect"], 10)


if __name__ == "__main__":
    unittest.main()
