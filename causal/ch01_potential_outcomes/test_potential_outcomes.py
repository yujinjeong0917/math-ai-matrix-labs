import unittest

import potential_outcomes as po


class HandTable(unittest.TestCase):
    def test_hand_numbers(self):
        h = po.hand_table()
        self.assertEqual((h["ate"], h["att"], h["atc"]), (2.5, 1.0, 4.0))
        self.assertEqual((h["treated_obs"], h["control_obs"], h["naive"]), (18.0, 4.0, 14.0))
        self.assertEqual(h["selection_bias"], 13.0)

    def test_naive_equals_att_plus_selection_bias(self):
        h = po.hand_table()
        self.assertEqual(h["naive"], h["att"] + h["selection_bias"])

    def test_ate_is_weighted_mean_of_att_and_atc(self):
        h = po.hand_table()
        self.assertEqual(h["ate"], 0.5 * h["att"] + 0.5 * h["atc"])


class Simulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop = po.make_population()

    def test_self_selection_overstates_effect(self):
        e = po.estimates(self.pop, po.self_selected(self.pop))
        self.assertEqual(round(e["ate"], 2), 2.48)
        self.assertEqual(round(e["att"], 2), 2.08)
        self.assertEqual(round(e["naive"], 2), 6.42)
        self.assertEqual(round(e["selection_bias"], 2), 4.34)
        self.assertAlmostEqual(e["naive"], e["att"] + e["selection_bias"], places=9)
        self.assertGreater(e["naive"], 2 * e["ate"])

    def test_randomization_removes_selection_bias(self):
        e = po.estimates(self.pop, po.randomized(self.pop))
        self.assertEqual(round(e["naive"], 2), 2.49)
        self.assertEqual(round(e["selection_bias"], 2), 0.01)
        self.assertAlmostEqual(e["naive"], e["ate"], delta=0.15)

    def test_itt_is_diluted_by_takeup(self):
        i = po.itt_demo(self.pop)
        self.assertEqual(round(i["itt"], 2), 1.48)
        self.assertEqual(round(i["takeup"], 3), 0.595)
        self.assertEqual(round(i["itt_over_takeup"], 2), 2.49)


if __name__ == "__main__":
    unittest.main()
