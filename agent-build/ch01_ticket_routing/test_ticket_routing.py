import unittest

from ticket_routing import COST, ROUTERS, evaluate, load_tickets


class TicketRoutingTest(unittest.TestCase):
    def test_forty_tickets_with_gold_labels(self):
        tickets = load_tickets()
        self.assertEqual(len(tickets), 40)
        counts = {l: sum(t["label"] == l for t in tickets) for l in ("환불", "교환", "배송조회", "기타")}
        self.assertEqual(counts, {"환불": 13, "교환": 10, "배송조회": 9, "기타": 8})

    def test_v1(self):
        r = evaluate(ROUTERS["v1"])
        self.assertEqual((r["correct"], r["cost"]), (29, 490000))
        self.assertEqual(len(r["wrong_refund_executions"]), 8)
        self.assertEqual(r["matrix"]["교환"]["환불"], 6)

    def test_v2_fixes_negation_but_breaks_one_real_refund(self):
        r = evaluate(ROUTERS["v2"])
        self.assertEqual((r["correct"], r["cost"]), (31, 313000))
        self.assertEqual(len(r["wrong_refund_executions"]), 5)
        self.assertIn("t13", r["missed_refunds"])  # "환불 말고는 방법이 없네요. 환불 진행해 주세요"

    def test_v3_has_lower_accuracy_but_lowest_cost(self):
        r = evaluate(ROUTERS["v3"])
        self.assertEqual((r["correct"], r["handoff"], r["cost"]), (28, 6, 142000))
        self.assertEqual(r["wrong_refund_executions"], ["t30", "t33"])

    def test_refund_precision_and_recall(self):
        got = [(evaluate(ROUTERS[v])["refund_predicted"], round(evaluate(ROUTERS[v])["refund_precision"], 3),
                round(evaluate(ROUTERS[v])["refund_recall"], 3)) for v in ("v1", "v2", "v3")]
        self.assertEqual(got, [(19, 0.579, 0.846), (15, 0.667, 0.769), (11, 0.818, 0.692)])

    def test_cost_is_asymmetric(self):
        self.assertEqual(COST["교환"]["환불"], 10 * COST["환불"]["교환"])

    def test_cost_matrix_sum_matches_hand_calculation_for_v1(self):
        # 기타->환불 1건 60,000 + 배송조회->환불 1건 60,000 + 교환->환불 6건 360,000
        # + 환불->기타 2건 6,000 + 기타->교환 1건 4,000
        self.assertEqual(60000 + 60000 + 360000 + 6000 + 4000, evaluate(ROUTERS["v1"])["cost"])


if __name__ == "__main__":
    unittest.main()
