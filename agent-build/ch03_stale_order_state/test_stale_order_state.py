import unittest
from math import comb

from stale_order_state import WORKERS, agent_steps, interleavings, run_all, run_one


class StaleOrderStateTest(unittest.TestCase):
    def test_interleaving_counts_are_binomial(self):
        for name, actions in WORKERS.items():
            for variant in ("V0", "V2"):
                n = len(agent_steps(variant))
                # 순서 개수가 이항계수와 같은지 확인(빠진 순서나 중복이 없다는 뜻). V1은 V0와 단계 수가 같아 따로 확인하지 않는다.
                self.assertEqual(len(list(interleavings(len(actions), n))), comb(len(actions) + n, len(actions)))

    def test_hand_example_customer_cancel(self):
        # 조회-판단-실행 사이 세 칸 중 앞의 두 칸에 취소가 끼면 돈이 두 번 나가고(V0), 맨 앞 칸이면 에이전트가 시도하지 않는다.
        cancel = ["cancel"]
        # 네 칸 중 W의 위치가 앞일수록 에이전트가 늦게 조회한다. 맨 앞이면 이미 취소돼 있어 환불을 계획하지 않는다.
        self.assertEqual(run_one(["W", "A", "A", "A"], "V0", cancel)["attempted"], False)
        self.assertEqual(run_one(["A", "W", "A", "A"], "V0", cancel)["payouts"], 2)
        self.assertEqual(run_one(["A", "A", "W", "A"], "V0", cancel)["payouts"], 2)
        after = run_one(["A", "A", "A", "W"], "V0", cancel)
        self.assertEqual((after["payouts"], after["final_state"]), (1, "환불됨"))  # 취소는 자기 검사에서 거절

    def test_v0_totals(self):
        t = run_all("V0")["total"]
        # README V0 합계 줄: orders 38, attempted 22, stale 18, double 6, unsafe 18
        self.assertEqual((t["orders"], t["attempted"], t["stale_attempts"], t["double_payout"], t["unsafe_refund"]), (38, 22, 18, 6, 18))

    def test_v1_blocks_every_stale_attempt(self):
        t = run_all("V1")["total"]
        # README V1 합계 줄: executed 4, blocked 18. 막힌 18건은 V0의 오래된 시도 18건과 같다.
        self.assertEqual((t["executed"], t["blocked"], t["double_payout"], t["unsafe_refund"]), (4, 18, 0, 0))
        self.assertEqual(t["blocked"], run_all("V0")["total"]["stale_attempts"])

    def test_v2_narrows_but_does_not_close_the_gap(self):
        t = run_all("V2")["total"]
        # README V2 합계 줄. 재조회만으로는 이중 지급 2건, 안전하지 않은 환불 7건이 남는다.
        self.assertEqual((t["orders"], t["attempted"], t["stale_attempts"], t["double_payout"], t["unsafe_refund"]), (60, 11, 7, 2, 7))
        self.assertGreater(t["stale_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
