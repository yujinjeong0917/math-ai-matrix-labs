import unittest

from disclosure_log import LOG, PAREEK, expand_rate, overall_expand_rate, selectivity_points


class Disclosure(unittest.TestCase):
    def test_pareek_constants(self):
        self.assertEqual(PAREEK["participants"], 12)
        self.assertEqual(PAREEK["wanted_toggle"], 8)
        self.assertEqual(PAREEK["said_less_info_needed_over_time"], 2)

    def test_each_month_has_40(self):
        for m in LOG:
            self.assertEqual(sum(v[0] for v in LOG[m].values()), 40)

    def test_hand_calc_month1(self):
        self.assertAlmostEqual(expand_rate(1, "틀림"), 9 / 12)
        self.assertAlmostEqual(expand_rate(1, "맞음"), 15 / 28)
        self.assertAlmostEqual(overall_expand_rate(1), 24 / 40)
        self.assertEqual(selectivity_points(1), 21.4)

    def test_month3_drop(self):
        self.assertAlmostEqual(overall_expand_rate(3), 12 / 40)
        self.assertEqual(selectivity_points(3), 4.8)

    def test_overall_hides_selectivity(self):
        self.assertAlmostEqual(overall_expand_rate(1) - overall_expand_rate(3), 0.30)
        self.assertGreater(selectivity_points(1), selectivity_points(3))


if __name__ == "__main__":
    unittest.main()
