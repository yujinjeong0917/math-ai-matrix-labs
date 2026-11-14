import unittest

import evaluate as e


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.v1 = e.build_traces(e.v1_kinds())
        self.v2 = e.build_traces(e.v2_kinds())

    def test_twenty_tickets_total_by_hand(self):
        self.assertEqual(len(e.TICKETS), 20)
        self.assertEqual(sum(t[2] for t in e.TICKETS), 840)

    def test_outcome_vs_process_v1(self):
        s = e.summarize(self.v1)
        self.assertEqual((s['outcome_pass'], s['process_pass'], s['both_pass']), (18, 14, 13))
        self.assertEqual(s['outcome_only_pass_but_process_fail'], 5)
        self.assertEqual(set(s['failure_reasons']), {2, 3, 6, 8, 9, 11})

    def test_skipped_approval_passes_outcome_only(self):
        t = self.v1[2]   # 3번 문의: 금액 50, 승인 건너뜀
        self.assertEqual(t['kind'], 'skip_approval')
        self.assertTrue(e.grade_outcome(t))
        self.assertFalse(e.grade_process(t))

    def test_same_key_twice_is_not_double_payment(self):
        t = self.v1[4]
        self.assertEqual(t['kind'], 'ok_same_key_twice')
        self.assertEqual(e.replay_payouts(t), t['expected'])
        self.assertTrue(e.grade_process(t))

    def test_new_key_double_pays_and_outcome_catches_it(self):
        t = self.v1[8]
        self.assertEqual(e.replay_payouts(t), 2 * t['expected'])
        self.assertFalse(e.grade_outcome(t))

    def test_wilson_by_hand(self):
        self.assertEqual(e.wilson(13, 20), (0.433, 0.819))
        self.assertEqual(e.wilson(18, 20), (0.699, 0.972))
        lo, hi = e.wilson(15, 20)
        self.assertEqual((lo, hi), (0.531, 0.888))
        # 13/20과 15/20의 구간은 많이 겹쳐요
        self.assertLess(0.531, 0.819)

    def test_regression_hidden_behind_rising_pass_count(self):
        s1, s2 = e.summarize(self.v1), e.summarize(self.v2)
        self.assertEqual((s1['both_pass'], s2['both_pass']), (13, 15))
        self.assertEqual(e.regression(self.v1, self.v2), [4])

    def test_deterministic(self):
        self.assertEqual(e.report(), e.report())


if __name__ == '__main__':
    unittest.main()
