import unittest
from math import comb

from stale_order_state import WORKERS, agent_steps, interleavings, run_all, run_one


class StaleOrderStateTest(unittest.TestCase):
    def test_interleaving_counts_are_binomial(self):
        for name, actions in WORKERS.items():
            for variant in ("V0", "V2"):
                n = len(agent_steps(variant))
                self.assertEqual(len(list(interleavings(len(actions), n))), comb(len(actions) + n, len(actions)))

    def test_hand_example_customer_cancel(self):
        # 조회-판단-실행 사이 세 칸 중 앞의 두 칸에 취소가 끼면 돈이 두 번 나가고(V0), 맨 앞 칸이면 에이전트가 시도하지 않는다.
        cancel = ["cancel"]
        self.assertEqual(run_one(["W", "A", "A", "A"], "V0", cancel)["attempted"], False)
        self.assertEqual(run_one(["A", "W", "A", "A"], "V0", cancel)["payouts"], 2)
        self.assertEqual(run_one(["A", "A", "W", "A"], "V0", cancel)["payouts"], 2)
        after = run_one(["A", "A", "A", "W"], "V0", cancel)
        self.assertEqual((after["payouts"], after["final_state"]), (1, "환불됨"))  # 취소는 자기 검사에서 거절

    def test_v0_totals(self):
        t = run_all("V0")["total"]
        self.assertEqual((t["orders"], t["attempted"], t["stale_attempts"], t["double_payout"], t["unsafe_refund"]), (38, 22, 18, 6, 18))

    def test_v1_blocks_every_stale_attempt(self):
        t = run_all("V1")["total"]
        self.assertEqual((t["executed"], t["blocked"], t["double_payout"], t["unsafe_refund"]), (4, 18, 0, 0))
        self.assertEqual(t["blocked"], run_all("V0")["total"]["stale_attempts"])

    def test_v2_narrows_but_does_not_close_the_gap(self):
        t = run_all("V2")["total"]
        self.assertEqual((t["orders"], t["attempted"], t["stale_attempts"], t["double_payout"], t["unsafe_refund"]), (60, 11, 7, 2, 7))
        self.assertGreater(t["stale_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
