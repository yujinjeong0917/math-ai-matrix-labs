import unittest

import unmeasured_wait as m


class Wait(unittest.TestCase):
    def test_harrison_percent_by_hand(self):
        # 5.61 / 5 = 1.122, 16.75 / 15 = 1.1167
        self.assertEqual(m.percent_increase(5, 5.61), 12.2)
        self.assertEqual(m.percent_increase(15, 16.75), 11.7)
        for base in (5, 15):
            self.assertEqual(m.percent_increase(base, m.HARRISON_SAME_FEEL[base]), m.HARRISON_REPORTED_PERCENT[base])

    def test_harrison_pair_score(self):
        self.assertEqual(sum(m.HARRISON_PAIR_EXAMPLE.values()), 40)
        self.assertEqual(m.preference_score(16, 6, 18), 0.25)

    def test_myers_uniform_mean(self):
        self.assertEqual(m.uniform_mean(1, 17), 9)
        self.assertAlmostEqual(9 - m.MYERS_EMPIRICAL_MEAN, 0.399)

    def test_myers_percent_is_not_a_whole_number_of_48(self):
        self.assertAlmostEqual(m.MYERS_LIKED_PERCENT / 100 * 48, 41.328)
        self.assertEqual(round(41 / 48 * 100, 1), 85.4)
        self.assertEqual(round(42 / 48 * 100, 1), 87.5)
        self.assertIn((31, 36), m.denominators_for_percent(86.1))
        self.assertNotIn(48, [n for _, n in m.denominators_for_percent(86.1)])

    def test_uncovered_share(self):
        self.assertEqual(m.uncovered(m.FLOW), (6000, 6060, 99.0))

    def test_all_covered(self):
        self.assertEqual(m.uncovered([("a", 10, "INP")]), (0, 10, 0.0))


class Quiz(unittest.TestCase):
    def test_quiz(self):
        self.assertEqual(m.uncovered(m.QUIZ_FLOW), (4000, 4100, 97.6))
        self.assertEqual(round(6 * 11.7 / 100, 1), 0.7)
        self.assertEqual(m.MYERS_PR["진행 표시 유무"], 0.0006)


if __name__ == "__main__":
    unittest.main()
