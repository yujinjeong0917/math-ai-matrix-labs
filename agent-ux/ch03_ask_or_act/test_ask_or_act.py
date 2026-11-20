import unittest

import ask_or_act as m


class Checks(unittest.TestCase):
    def test_hand_p_star(self):
        # 손계산: (1.0-0.0) / ((1.0-0.0) + (1.0-0.6)) = 1/1.4
        self.assertAlmostEqual(m.p_star(), 1 / 1.4, places=9)

    def test_equal_expected_utility_at_p_star(self):
        ps = m.p_star()
        self.assertAlmostEqual(m.eu_act(ps), m.eu_none(ps), places=9)

    def test_thresholds(self):
        self.assertAlmostEqual(m.p_none_dialog(), 0.6, places=9)
        self.assertAlmostEqual(m.p_dialog_act(), 0.7 / 0.9, places=9)
        self.assertLess(m.p_none_dialog(), m.p_star())
        self.assertLess(m.p_star(), m.p_dialog_act())

    def test_equal_at_both_thresholds(self):
        p1 = m.p_none_dialog()
        p2 = m.p_dialog_act()
        self.assertAlmostEqual(m.eu_none(p1), m.eu_dialog(p1), places=9)
        self.assertAlmostEqual(m.eu_dialog(p2), m.eu_act(p2), places=9)

    def test_decisions(self):
        self.assertEqual([m.decide(p) for p in (0.5, 0.65, 0.75, 0.8, 0.95)], ["가만히", "묻기", "묻기", "실행", "실행"])

    def test_tie_goes_to_cautious_choice(self):
        self.assertEqual(m.decide(m.p_none_dialog()), "가만히")

    def test_cheaper_mistake_lowers_threshold(self):
        u = dict(m.U)
        u["AnG"] = 0.5
        self.assertLess(m.p_star(u), m.p_star())

    def test_cheaper_delay_raises_threshold(self):
        u = dict(m.U)
        u["nAG"] = 0.9
        self.assertGreater(m.p_star(u), m.p_star())
        self.assertAlmostEqual(m.p_star(u), 1 / 1.1, places=9)

    def test_no_dialog_region_when_dialog_is_costly(self):
        u = dict(m.U)
        u["DG"], u["DnG"] = 0.5, 0.2
        self.assertGreaterEqual(m.p_none_dialog(u), m.p_dialog_act(u))
        for p in (0.1, 0.5, 0.9):
            self.assertNotEqual(m.decide(p, u), "묻기")


if __name__ == "__main__":
    unittest.main()
