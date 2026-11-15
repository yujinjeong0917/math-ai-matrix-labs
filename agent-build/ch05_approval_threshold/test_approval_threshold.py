import unittest

from approval_threshold import (AMOUNTS, SPLIT_REQUESTS, THRESHOLDS, needs_approval,
                                run_split, sweep)


class SweepTest(unittest.TestCase):
    def test_input_is_fixed(self):
        self.assertEqual(len(AMOUNTS), 100)
        # 입력 고정 확인: 100건, 합계 4,423,000원, 최대 217,000원(README 마지막 줄 문턱 250,000의 값)
        self.assertEqual(sum(AMOUNTS), 4_423_000)
        self.assertEqual(max(AMOUNTS), 217_000)

    def test_boundary_is_strictly_greater(self):
        self.assertFalse(needs_approval(50_000, 50_000))
        self.assertTrue(needs_approval(50_001, 50_000))

    def test_table_values(self):
        # 키: 문턱, 값: (승인 건수, 자동 합계, 자동 최대). README 표의 앞 세 열과 같다. 자동 최대는 문턱 이하 중 가장 큰 실제 금액이라 문턱보다 작을 수 있다.
        expected = {
            10_000: (90, 79_000, 10_000),
            20_000: (68, 426_000, 20_000),
            30_000: (51, 863_000, 30_000),
            50_000: (31, 1_652_000, 50_000),
            80_000: (14, 2_753_000, 79_000),
            100_000: (9, 3_191_000, 95_000),
            150_000: (2, 4_042_000, 134_000),
            250_000: (0, 4_423_000, 217_000),
        }
        for t, (approvals, auto_sum, auto_max) in expected.items():
            r = sweep(AMOUNTS, t)
            self.assertEqual((r["approvals"], r["auto_sum"], r["auto_max"]), (approvals, auto_sum, auto_max), t)

    def test_monotone(self):
        rows = [sweep(AMOUNTS, t) for t in THRESHOLDS]
        for a, b in zip(rows, rows[1:]):
            self.assertGreaterEqual(a["approvals"], b["approvals"])
            self.assertLessEqual(a["auto_sum"], b["auto_sum"])
            # 문턱이 올라가면 승인은 줄고 자동 합계는 늘며, 두 건수의 합은 늘 100
            self.assertEqual(a["approvals"] + a["auto_count"], 100)

    def test_auto_max_never_exceeds_threshold(self):
        for t in THRESHOLDS:
            self.assertLessEqual(sweep(AMOUNTS, t)["auto_max"], t)

    def test_workload_arithmetic(self):
        r = sweep(AMOUNTS, 50_000)
        # 31건 x 3분 = 93분, 31건 x 30분 = 930분(가정 값의 곱셈)
        self.assertEqual(r["approver_minutes"], 31 * 3)
        self.assertEqual(r["customer_wait_minutes"], 31 * 30)
        self.assertEqual(r["held_sum"] + r["auto_sum"], 4_423_000)


class SplitTest(unittest.TestCase):
    def test_per_request_lets_everything_through(self):
        r = run_split("per_request", 50_000)
        self.assertEqual(r["auto_total"], sum(sum(v) for _, v in SPLIT_REQUESTS))
        # 주문별 요청 합계(천 원) O1 120, O2 90, O3 150, O4 60, O5 49, O6 200 = 669. 하나도 보류되지 않고 나가며, 문턱을 넘긴 주문이 5곳(O5 제외).
        self.assertEqual(r["auto_total"], 669_000)
        self.assertEqual(r["held_requests"], 0)
        self.assertEqual(r["auto_over_threshold_orders"], ["O1", "O2", "O3", "O4", "O6"])
        self.assertEqual(r["max_auto_per_order"], 200_000)

    def test_per_order_cumulative_blocks(self):
        r = run_split("per_order", 50_000)
        # 40+45+50+30+49+49 = 263 (천 원). 첫 요청까지만 나가고 나머지 10건 보류.
        self.assertEqual(r["auto_total"], 263_000)
        self.assertEqual(r["held_requests"], 10)
        self.assertEqual(r["auto_over_threshold_orders"], [])
        self.assertLessEqual(r["max_auto_per_order"], 50_000)

    def test_unsplit_control_is_unaffected(self):
        a = run_split("per_request", 50_000)["auto_paid_by_order"]["O5"]
        b = run_split("per_order", 50_000)["auto_paid_by_order"]["O5"]
        self.assertEqual((a, b), (49_000, 49_000))


if __name__ == "__main__":
    unittest.main()
