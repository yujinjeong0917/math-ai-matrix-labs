import unittest

import evaluate as e


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.v1 = e.build_traces(e.v1_kinds())
        self.v2 = e.build_traces(e.v2_kinds())

    def test_twenty_tickets_total_by_hand(self):
        self.assertEqual(len(e.TICKETS), 20)
        # 정답 금액 합 840(손계산)
        self.assertEqual(sum(t[2] for t in e.TICKETS), 840)

    def test_outcome_vs_process_v1(self):
        s = e.summarize(self.v1)
        # README 표: 결과만 18(1·9번 실패), 과정만 14(2·3·6·8·9·11번 실패), 둘 다 13. 5건 = 2·3·6·8·11번.
        self.assertEqual((s['outcome_pass'], s['process_pass'], s['both_pass']), (18, 14, 13))
        self.assertEqual(s['outcome_only_pass_but_process_fail'], 5)
        self.assertEqual(set(s['failure_reasons']), {2, 3, 6, 8, 9, 11})

    def test_skipped_approval_passes_outcome_only(self):
        # 리스트 인덱스는 0부터라 self.v1[2]가 3번 문의다(off-by-one 주의). 아래 [4]는 5번, [8]은 9번 문의.
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
        # README 표의 신뢰구간: 13/20 -> 0.433~0.819, 18/20 -> 0.699~0.972, 15/20 -> 0.531~0.888
        self.assertEqual(e.wilson(13, 20), (0.433, 0.819))
        self.assertEqual(e.wilson(18, 20), (0.699, 0.972))
        lo, hi = e.wilson(15, 20)
        self.assertEqual((lo, hi), (0.531, 0.888))
        # 13/20과 15/20의 구간은 많이 겹쳐요
        # 15/20의 하한이 13/20의 상한보다 작다 = 구간이 겹친다. 20건으로는 13과 15를 구분하기 어렵다.
        self.assertLess(0.531, 0.819)

    def test_regression_hidden_behind_rising_pass_count(self):
        s1, s2 = e.summarize(self.v1), e.summarize(self.v2)
        self.assertEqual((s1['both_pass'], s2['both_pass']), (13, 15))
        # 통과 수 13 -> 15지만 4번은 v1 통과, v2 실패(금액을 5 더 보냄).
        self.assertEqual(e.regression(self.v1, self.v2), [4])

    def test_deterministic(self):
        self.assertEqual(e.report(), e.report())


if __name__ == '__main__':
    unittest.main()
