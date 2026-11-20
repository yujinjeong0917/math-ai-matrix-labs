import unittest

from reliance_log import (AGENTS, acceptance_rate, appropriate_rate, disuse_rate, misuse_rate, n_total, rank_gaps,
                          ranks_desc)


class Reliance(unittest.TestCase):
    def test_each_agent_has_20(self):
        self.assertTrue(all(n_total(a) == 20 for a in AGENTS.values()))

    def test_hand_calc_agent_ga(self):
        a = AGENTS["가"]
        self.assertAlmostEqual(acceptance_rate(a), 13 / 20)
        self.assertAlmostEqual(misuse_rate(a), 1 / 6)
        self.assertAlmostEqual(disuse_rate(a), 2 / 14)
        self.assertAlmostEqual(appropriate_rate(a), 17 / 20)

    def test_agent_na_high_trust_low_use(self):
        a = AGENTS["나"]
        self.assertEqual(a["trust"], 5)
        self.assertAlmostEqual(acceptance_rate(a), 0.45)
        self.assertAlmostEqual(disuse_rate(a), 6 / 14)

    def test_agent_da_low_trust_high_use(self):
        a = AGENTS["다"]
        self.assertEqual(a["trust"], 2)
        self.assertAlmostEqual(acceptance_rate(a), 0.80)
        self.assertAlmostEqual(misuse_rate(a), 5 / 8)

    def test_ranks_and_gaps(self):
        self.assertEqual(ranks_desc({"x": 3, "y": 5, "z": 1}), {"y": 1, "x": 2, "z": 3})
        self.assertEqual(rank_gaps(AGENTS), {"가": 0, "나": 3, "다": 3, "라": 0})

    def test_four_cells_sum_to_80(self):
        s = sum(n_total(a) for a in AGENTS.values())
        self.assertEqual(s, 80)


if __name__ == "__main__":
    unittest.main()
