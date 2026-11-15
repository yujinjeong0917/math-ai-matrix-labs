import unittest

from ticket_routing import COST, ROUTERS, evaluate, load_tickets


class TicketRoutingTest(unittest.TestCase):
    def test_forty_tickets_with_gold_labels(self):
        tickets = load_tickets()
        self.assertEqual(len(tickets), 40)
        counts = {l: sum(t["label"] == l for t in tickets) for l in ("환불", "교환", "배송조회", "기타")}
        # 정답 라벨 분포 13+10+9+8 = 40. 재현율 분모 13이 여기서 나온다.
        self.assertEqual(counts, {"환불": 13, "교환": 10, "배송조회": 9, "기타": 8})

    def test_v1(self):
        r = evaluate(ROUTERS["v1"])
        # README 표 v1 행. 교환을 환불로 보낸 6건이 가장 큰 비용 원인.
        self.assertEqual((r["correct"], r["cost"]), (29, 490000))
        self.assertEqual(len(r["wrong_refund_executions"]), 8)
        self.assertEqual(r["matrix"]["교환"]["환불"], 6)

    def test_v2_fixes_negation_but_breaks_one_real_refund(self):
        r = evaluate(ROUTERS["v2"])
        self.assertEqual((r["correct"], r["cost"]), (31, 313000))
        self.assertEqual(len(r["wrong_refund_executions"]), 5)
        # v2의 부작용: 진짜 환불 요청 한 건을 부정 표현으로 오인해 놓친다.
        self.assertIn("t13", r["missed_refunds"])  # "환불 말고는 방법이 없네요. 환불 진행해 주세요"

    def test_v3_has_lower_accuracy_but_lowest_cost(self):
        r = evaluate(ROUTERS["v3"])
        # v3: 정답 수는 가장 적지만(28) 사람 확인 6건 덕에 비용이 가장 낮다. 정확도와 비용이 다른 기준임을 보인다.
        self.assertEqual((r["correct"], r["handoff"], r["cost"]), (28, 6, 142000))
        self.assertEqual(r["wrong_refund_executions"], ["t30", "t33"])

    def test_refund_precision_and_recall(self):
        # (환불 예측 수, 정밀도, 재현율) = v1 19건 11/19, 11/13 / v2 15건 10/15, 10/13 / v3 11건 9/11, 9/13. 소수 셋째 자리까지 반올림해 비교한다.
        got = [(evaluate(ROUTERS[v])["refund_predicted"], round(evaluate(ROUTERS[v])["refund_precision"], 3),
                round(evaluate(ROUTERS[v])["refund_recall"], 3)) for v in ("v1", "v2", "v3")]
        self.assertEqual(got, [(19, 0.579, 0.846), (15, 0.667, 0.769), (11, 0.818, 0.692)])

    def test_cost_is_asymmetric(self):
        # 환불 잘못 실행(60000)은 환불을 교환으로 보낸 경우(6000)의 10배라는 비대칭
        self.assertEqual(COST["교환"]["환불"], 10 * COST["환불"]["교환"])

    def test_cost_matrix_sum_matches_hand_calculation_for_v1(self):
        # 기타->환불 1건 60,000 + 배송조회->환불 1건 60,000 + 교환->환불 6건 360,000
        # + 환불->기타 2건 6,000 + 기타->교환 1건 4,000
        self.assertEqual(60000 + 60000 + 360000 + 6000 + 4000, evaluate(ROUTERS["v1"])["cost"])


if __name__ == "__main__":
    unittest.main()
