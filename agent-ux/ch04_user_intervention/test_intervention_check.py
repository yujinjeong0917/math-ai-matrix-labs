import unittest

import intervention_check as m


class Checks(unittest.TestCase):
    def test_cell_sums(self):
        self.assertEqual(sum(m.CELLS.values()), 248)
        self.assertEqual(m.CELLS["UP-AE"] + m.CELLS["UP-UE"], m.PLAN_EDITED_OF_UP[1])
        self.assertEqual(m.CELLS["AP-UE"] + m.CELLS["UP-UE"], m.EXEC_CHANGED_OF_UE[1])

    def test_shares(self):
        self.assertEqual(m.pct(*m.PLAN_EDITED_OF_UP), 86.0)
        self.assertEqual(m.pct(*m.EXEC_CHANGED_OF_UE), 91.9)

    def test_reported_accuracy_does_not_match_counts(self):
        n, wrong = m.TASKS_Q5
        self.assertEqual(m.pct(n - wrong, n), 67.2)
        self.assertNotEqual(m.pct(n - wrong, n), m.REPORTED_Q5_ACC)

    def test_cause_shares_sum(self):
        self.assertAlmostEqual(sum(m.FAIL_CAUSE.values()), 100.0, places=6)

    def test_ct_hand_values(self):
        a, b = m.POOLS["고품질이 많은 묶음"], m.POOLS["결함이 많은 묶음"]
        # 손계산: 고품질 5/6, 1/6
        self.assertAlmostEqual(m.always_trust(a), 5 / 6)
        self.assertAlmostEqual(m.always_trust(b), 1 / 6)
        self.assertAlmostEqual(m.always_distrust(a), 1 / 6)
        self.assertAlmostEqual(m.always_distrust(b), 5 / 6)
        self.assertEqual(m.ideal_reviewer(a), 1.0)
        self.assertEqual(m.ideal_reviewer(b), 1.0)

    def test_always_trust_equals_good_share(self):
        for q in m.POOLS.values():
            self.assertAlmostEqual(m.always_trust(q), sum(1 for x in q if x == 5) / len(q))

    def test_ct_tracks_pool_not_checking(self):
        a, b = m.POOLS["고품질이 많은 묶음"], m.POOLS["결함이 많은 묶음"]
        self.assertGreater(m.always_trust(a), m.always_trust(b))


if __name__ == "__main__":
    unittest.main()
