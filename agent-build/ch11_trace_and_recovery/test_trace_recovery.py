import math
import unittest

import trace_recovery as t


class TraceRecoveryTest(unittest.TestCase):
    def test_find_steps_by_hand(self):
        # 증상 단계 - 원인 단계 + 1: (7-5+1, 6-5+1, 7-3+1, 7-4+1, 6-2+1, 6-6+1)
        self.assertEqual(t.find_summary()['without_trace'], [3, 2, 5, 4, 5, 1])
        self.assertEqual(t.find_summary()['sum_without'], 20)
        self.assertEqual(t.find_summary()['sum_with'], 6)

    def test_recovery_counts(self):
        # README 복구 표. 요청 번호가 없으면 6건 모두 사람 인계.
        self.assertEqual(t.recovery_summary(True), {'자동 재시도': 2, '보상 입금': 1, '사람 인계': 3})
        self.assertEqual(t.recovery_summary(False), {'자동 재시도': 0, '보상 입금': 0, '사람 인계': 6})
        self.assertEqual(sum(t.recovery_summary(True).values()), len(t.FAILURES))

    def test_compensation_candidates(self):
        # 정산표에서 금액 30인 행 4개 = 번호 없을 때의 의심 행 수
        self.assertEqual(t.SETTLEMENT.count(30), 4)
        self.assertEqual(t.compensation_candidates(True), 1)
        self.assertEqual(t.compensation_candidates(False), 4)

    def test_twenty_interleavings(self):
        orders = list(t.interleavings())
        self.assertEqual(len(orders), math.comb(6, 3))
        # 집합으로 바꿔 중복이 없는지 확인(순서 목록을 튜플로 바꿔야 집합에 넣을 수 있다).
        self.assertEqual(len({tuple(o) for o in orders}), 20)
        for o in orders:   # 요청 안쪽 걸음 순서는 지켜져요
            self.assertEqual([s for w, s in o if w == 'a'], [0, 1, 2])
            self.assertEqual([s for w, s in o if w == 'b'], [0, 1, 2])

    def test_sequential_orders_are_safe_but_others_double_pay(self):
        # 두 요청을 완전히 차례로 실행하면 안전(4장 실험과 같은 상황). 확인을 둘 다 먼저 하면 이중 지급.
        a_then_b = [('a', 0), ('a', 1), ('a', 2), ('b', 0), ('b', 1), ('b', 2)]
        both_check_first = [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2), ('b', 2)]
        self.assertEqual(t.run_check_then_store(a_then_b), 1)     # 4장 실험과 같은 결과
        self.assertEqual(t.run_check_then_store(both_check_first), 2)

    def test_race_summary(self):
        r = t.race_summary()
        self.assertEqual(r['orders'], 20)
        # 20가지 중 18가지에서 이중 지급(README). 자리 잡기를 한 번에 하는 방식은 0가지.
        self.assertEqual(r['check_then_store_double_pay_orders'], 18)
        self.assertEqual(r['check_then_store_max_payouts'], 2)
        self.assertEqual(r['atomic_double_pay_orders'], 0)
        self.assertEqual(r['atomic_max_payouts'], 1)
        self.assertEqual(r['atomic_zero_pay_orders'], 0)

    def test_deterministic(self):
        self.assertEqual(t.report(), t.report())


if __name__ == '__main__':
    unittest.main()
