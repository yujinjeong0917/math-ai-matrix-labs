import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staggered as s


class HandExample(unittest.TestCase):
    def test_already_treated_region_as_control_gives_zero(self):
        self.assertEqual(s.hand_two_by_two(s.HAND_B, s.HAND_A, 1, 2), 0.0)

    def test_clean_comparison_gives_ten(self):
        self.assertEqual(s.hand_two_by_two(s.HAND_A, s.HAND_B, 0, 1), 10.0)


class Panel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = s.make_panel()
        cls.beta, cls.wts = s.twfe(cls.rows)

    def test_truth_and_twfe(self):
        self.assertEqual(sum(r["d"] for r in self.rows), 240)
        self.assertEqual(round(s.true_average_effect(self.rows), 2), 11.50)
        self.assertEqual(round(self.beta, 2), 5.88)
        self.assertEqual(round(s.weighted_average_effect(self.wts), 2), 6.00)

    def test_negative_weights_sit_on_already_treated_region(self):
        by, neg, n_neg, n = s.weight_summary(self.wts)
        self.assertEqual((round(by["A"], 3), round(by["B"], 3), n_neg, n, round(neg, 3)), (0.375, 0.625, 80, 240, -0.125))
        parts = {k: (round(w, 3), round(m, 2)) for k, w, m in s.weight_parts(self.wts)}
        self.assertEqual(parts, {"A 3~6주차": (0.5, 7.5), "A 7~10주차": (-0.125, 19.5), "B 7~10주차": (0.625, 7.5)})
        self.assertAlmostEqual(sum(w * m for _, w, m in s.weight_parts(self.wts)), 6.0, places=9)
        late_a = [w for r, w in self.wts if r["g"] == "A" and r["t"] >= s.GROUPS["B"]]
        self.assertEqual(len(late_a), 80)
        neg = [(r["g"], r["t"]) for r, w in self.wts if w < 0]
        self.assertEqual(set(neg), {("A", t) for t in range(7, 11)})
        self.assertTrue(all(w < 0 for w in late_a))
        self.assertAlmostEqual(abs(sum(w for _, w in self.wts) - 1.0), 0.0, places=9)

    def test_gap_between_6_and_5_88_is_noise(self):
        quiet, _ = s.twfe(s.make_panel(noise=0.0))
        self.assertEqual(round(quiet, 2), 6.00)
        self.assertEqual(round(self.beta, 2), 5.88)

    def test_clean_estimator_recovers_event_study(self):
        att = s.att_gt(self.rows)
        es = s.event_study(att)
        want = {-6: 0.14, -5: 0.35, -4: 1.21, -3: 0.59, -2: -0.88, 0: 2.93, 1: 5.78, 2: 8.88,
                3: 11.93, 4: 15.18, 5: 17.96, 6: 20.92, 7: 23.77}
        self.assertEqual({e: round(v, 2) for e, v in es.items()}, want)
        self.assertEqual(round(s.overall_att(att), 2), 11.41)
        self.assertTrue(all(abs(es[e]) < 1.5 for e in es if e < 0))

    def test_without_never_treated_region_sign_flips(self):
        two = [r for r in self.rows if r["g"] != "C"]
        beta, w = s.twfe(two)
        self.assertEqual(round(s.true_average_effect(two), 2), 11.50)
        self.assertEqual(round(beta, 2), -0.66)
        self.assertEqual(round(s.weighted_average_effect(w), 2), -0.50)
        parts = {k: (round(a, 3), round(m, 2)) for k, a, m in s.weight_parts(w)}
        self.assertEqual(parts, {"A 3~6주차": (1.0, 7.5), "A 7~10주차": (-0.667, 19.5), "B 7~10주차": (0.667, 7.5)})
        self.assertEqual(sum(1 for _, x in w if x < 0), 80)

    def test_constant_effect_is_fine(self):
        flat = s.make_panel()
        for r in flat:
            if r["d"] == 1:
                r["y"] += 8.0 - r["tau"]
                r["tau"] = 8.0
        self.assertEqual(round(s.twfe(flat)[0], 2), 7.88)
        quiet = s.make_panel(noise=0.0)
        for r in quiet:
            if r["d"] == 1:
                r["y"] += 8.0 - r["tau"]
                r["tau"] = 8.0
        self.assertEqual(round(s.twfe(quiet)[0], 2), 8.00)


if __name__ == "__main__":
    unittest.main()
