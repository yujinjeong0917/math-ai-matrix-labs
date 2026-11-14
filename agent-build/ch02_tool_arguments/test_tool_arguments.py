import unittest

from tool_arguments import ORDERS, load_calls, schema_check, semantic_check, summarize


class ToolArgumentsTest(unittest.TestCase):
    def test_thirty_calls(self):
        self.assertEqual(len(load_calls()), 30)

    def test_summary_counts(self):
        s = summarize()
        self.assertEqual(s["counts"], {"통과(정상)": 15, "형식 오류로 걸림": 6, "의미 오류로 걸림": 5, "조용한 오류": 4})
        self.assertEqual(s["wrong_calls"], 15)
        self.assertEqual(s["wrong_passed_schema_only"], 9)
        self.assertEqual(s["silent_ids"], ["c27", "c28", "c29", "c30"])

    def test_no_correct_call_is_rejected(self):
        self.assertNotIn("정상인데 거절됨", summarize()["counts"])

    def test_schema_check_examples(self):
        self.assertEqual(schema_check({"order_id": "ORD-1001", "reason": "불량"}), ["필수 값 없음: amount"])
        self.assertEqual(schema_check({"order_id": "ORD-1001", "amount": "30000", "reason": "불량"}), ["자료형 틀림: amount"])
        self.assertEqual(schema_check({"order_id": "ORD-1001", "amount": True, "reason": "불량"}), ["자료형 틀림: amount"])

    def test_semantic_check_examples(self):
        self.assertEqual(semantic_check({"order_id": "ORD-2001", "amount": 1000}), ["다른 고객의 주문"])
        self.assertEqual(semantic_check({"order_id": "ORD-1003", "amount": 30001}), ["환불 가능 금액 초과"])
        self.assertEqual(semantic_check({"order_id": "ORD-1003", "amount": 30000}), [])

    def test_unit_error_passes_every_check(self):
        call = {"order_id": "ORD-1002", "amount": 30, "reason": "불량"}  # 30000원 주문에 30
        self.assertEqual(schema_check(call), [])
        self.assertEqual(semantic_check(call), [])
        self.assertLess(call["amount"], ORDERS["ORD-1002"]["paid"])


if __name__ == "__main__":
    unittest.main()
