import unittest

from reading_months import (FULL, MINIMAL, QUESTIONS, TICKETS, answerable, fixed_mix_rate, mix_share, n_answerable,
                            resolve_rate)


class Months(unittest.TestCase):
    def test_100_tickets_per_month(self):
        for m in (1, 3):
            self.assertEqual(sum(TICKETS[(m, l)][0] for l in ("쉬움", "어려움")), 100)

    def test_hand_calc_overall(self):
        self.assertAlmostEqual(resolve_rate(1), 0.58)
        self.assertAlmostEqual(resolve_rate(3), 0.76)

    def test_every_stratum_falls_but_overall_rises(self):
        self.assertAlmostEqual(resolve_rate(1, "쉬움"), 0.90)
        self.assertAlmostEqual(resolve_rate(3, "쉬움"), 0.85)
        self.assertAlmostEqual(resolve_rate(1, "어려움"), 0.50)
        self.assertAlmostEqual(resolve_rate(3, "어려움"), 0.40)
        self.assertGreater(resolve_rate(3), resolve_rate(1))

    def test_mix_share(self):
        self.assertAlmostEqual(mix_share(1, "쉬움"), 0.20)
        self.assertAlmostEqual(mix_share(3, "쉬움"), 0.80)

    def test_fixed_mix(self):
        self.assertAlmostEqual(fixed_mix_rate(3, 1), 0.85 * 0.2 + 0.40 * 0.8)
        self.assertLess(fixed_mix_rate(3, 1), resolve_rate(1))

    def test_answerable(self):
        self.assertEqual(len(QUESTIONS), 6)
        self.assertEqual(n_answerable(MINIMAL), 2)
        self.assertEqual(n_answerable(FULL), 6)
        self.assertTrue(answerable(MINIMAL)["수락률"])
        self.assertFalse(answerable(MINIMAL)["오용과 불용"])


if __name__ == "__main__":
    unittest.main()
