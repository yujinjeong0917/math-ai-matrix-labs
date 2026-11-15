import unittest

from scoped_permissions import ORDERS, REQUESTS, VARIANTS, call_tool, run_variant


class ScopedPermissionsTest(unittest.TestCase):
    def test_request_mix(self):
        kinds = [r[1] for r in REQUESTS]
        self.assertEqual((len(kinds), kinds.count("own"), kinds.count("other"), kinds.count("family")), (20, 10, 6, 4))

    def test_other_refund_total_by_hand(self):
        # 타인 환불 3건(O301, O401, O302)의 결제 금액 합 = 33,000+58,000+150,000 = 241,000
        self.assertEqual(ORDERS["O301"][1] + ORDERS["O401"][1] + ORDERS["O302"][1], 241_000)

    def test_expected_counts(self):
        # README 표의 숫자 여섯 열과 같은 값(변형별 기대값)
        expected = {  # (타인 조회 유출, 타인 환불 건수, 타인 환불 합계, 본인 거절, 가족 거절, 사람 대기)
            "V1": (3, 3, 241_000, 0, 0, 0),
            "V2": (3, 3, 241_000, 0, 0, 0),
            "V3": (3, 2, 91_000, 0, 0, 1),
            "V4": (0, 0, 0, 0, 4, 0),
            "V5": (0, 0, 0, 0, 4, 1),
            "V6": (0, 0, 0, 0, 0, 1),
        }
        self.assertEqual(set(expected), set(VARIANTS))
        for v, e in expected.items():
            r = run_variant(v)
            got = (r["leaked_lookups"], r["unauthorized_refunds"], r["unauthorized_sum"],
                   r["legit_denied_own"], r["legit_denied_family"], r["legit_held_for_human"])
            self.assertEqual(got, e, v)

    def test_model_supplied_customer_id_is_not_a_boundary(self):
        # 모델이 주문 주인의 ID를 적어 보내면 V2는 막지 못한다. 세션 범위(V4)는 인자를 보지 않는다.
        # V2는 인자를 믿어 통과하고, V4는 세션 고객 C1이 주인이 아니라 인자와 무관하게 거절한다.
        self.assertEqual(call_tool("V2", "refund", "O301", 33_000, "C3")[0], "ok")
        self.assertEqual(call_tool("V4", "refund", "O301", 33_000, "C3")[0], "denied")
        self.assertEqual(call_tool("V4", "refund", "O301", 33_000, "C1")[0], "denied")

    def test_cap_alone_still_leaks_below_cap(self):
        # 상한만 있으면 상한 이하의 타인 환불(58,000)은 그대로 나간다. 150,000은 보류.
        self.assertEqual(call_tool("V3", "refund", "O401", 58_000, "C4")[0], "ok")
        self.assertEqual(call_tool("V3", "refund", "O302", 150_000, "C3")[0], "held")

    def test_refund_above_paid_is_denied_everywhere(self):
        for v in VARIANTS:
            self.assertEqual(call_tool(v, "refund", "O101", 38_001, "C1")[0], "denied", v)

    def test_proxy_only_in_v6(self):
        # O201은 C2의 주문이고 C1이 대리인으로 등록돼 있다(PROXIES). 대리 등록표는 V6에만 있다.
        self.assertEqual(call_tool("V6", "refund", "O201", 45_000, "C2")[0], "ok")
        self.assertEqual(call_tool("V5", "refund", "O201", 45_000, "C2")[0], "denied")
        # 등록되지 않은 고객의 주문은 V6에서도 막힌다.
        self.assertEqual(call_tool("V6", "lookup", "O301", 0, "C3")[0], "denied")

    def test_unknown_order(self):
        self.assertEqual(call_tool("V6", "lookup", "O999", 0, "C1")[0], "denied")


if __name__ == "__main__":
    unittest.main()
