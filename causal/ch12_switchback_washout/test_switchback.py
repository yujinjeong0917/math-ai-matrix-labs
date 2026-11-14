import random
import unittest
import switchback as s


class Switchback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = s.report()

    def test_true_steady_state(self):
        self.assertAlmostEqual(s.TRUE_STEADY, 1.5)

    def test_hourly_flip_recovers_only_tau(self):
        x = self.r[(1, 0)]
        self.assertAlmostEqual(x["mean"], 6.0, delta=0.3)  # 이월을 못 보고 tau만 본다
        self.assertGreater(x["bias"], 4.0)

    def test_bias_shrinks_with_block_and_washout(self):
        b0 = self.r[(4, 0)]["bias"]
        b1 = self.r[(4, 1)]["bias"]
        b2 = self.r[(4, 2)]["bias"]
        self.assertGreater(b0, b1)
        self.assertGreater(b1, abs(b2))
        self.assertLess(abs(b2), 0.15)

    def test_washout_equal_to_carryover_length_is_unbiased(self):
        for key in ((4, 2), (8, 2), (12, 2), (24, 2)):
            self.assertLess(abs(self.r[key]["bias"]), 0.2)

    def test_washout_costs_data(self):
        self.assertAlmostEqual(s.kept_share(4, 2), 0.75, delta=0.02)  # 손계산 1 - 2/4 x 1/2
        self.assertAlmostEqual(s.kept_share(24, 2), 0.958, delta=0.02)

    def test_flat_base_sd_is_small_and_flat(self):
        sds = s.flat_sd_report().values()
        self.assertLess(max(sds), 0.5)
        self.assertLess(max(sds), self.r[(4, 2)]['sd'])

    def test_assignment_balanced(self):
        T = s.assign(8, random.Random(1))
        self.assertEqual(len(T), s.N)
        self.assertEqual(sum(T), s.N // 2)


if __name__ == "__main__":
    unittest.main()
