import unittest
import synth as s


class Fit(unittest.TestCase):
    def test_weights_on_simplex(self):
        tr, dn = s.make_panel(in_hull=True)
        w = s.fit_weights(tr, dn)
        self.assertAlmostEqual(sum(w), 1.0, places=9)
        self.assertTrue(all(x >= 0 for x in w))

    def test_good_fit_recovers_effect(self):
        a = s.report()["in_hull"]
        self.assertLess(a["pre_rmspe"], 1.0)
        self.assertAlmostEqual(a["effect"], 10.0, delta=0.6)
        self.assertEqual(a["placebo_rank"], 1)  # 가짜 처치 8개보다 큰 후/전 비율

    def test_equal_weights_less_accurate(self):
        a = s.report()["in_hull"]
        self.assertGreater(abs(a["equal_weight_did"] - 10), abs(a["effect"] - 10))

    def test_pre_fit_failure_gives_wrong_answer(self):
        a = s.report()["out_hull"]
        self.assertGreater(a["pre_rmspe"], 10)  # 처치 전부터 못 맞춘다
        self.assertGreater(a["treated_pre_mean"], a["max_donor_pre_mean"])  # 도너 어느 것보다 높다
        self.assertGreater(abs(a["effect"] - 10), 50)  # 효과 추정이 크게 틀린다
        self.assertGreater(a["placebo_rank"], 1)  # 가짜 처치 검정으로도 두드러지지 않는다

    def test_deterministic(self):
        self.assertEqual(s.report()["in_hull"]["effect"], s.report()["in_hull"]["effect"])


if __name__ == "__main__":
    unittest.main()
