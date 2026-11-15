import unittest

from tool_arguments import ORDERS, load_calls, schema_check, semantic_check, summarize


class ToolArgumentsTest(unittest.TestCase):
    def test_thirty_calls(self):
        self.assertEqual(len(load_calls()), 30)

    def test_summary_counts(self):
        s = summarize()
        # README 기대 출력과 같은 분류별 건수(15+6+5+4 = 30)
        self.assertEqual(s["counts"], {"통과(정상)": 15, "형식 오류로 걸림": 6, "의미 오류로 걸림": 5, "조용한 오류": 4})
        # 15 = 6+5+4, 9 = 15-6(형식 검사만 있을 때 실행될 잘못된 호출)
        self.assertEqual(s["wrong_calls"], 15)
        self.assertEqual(s["wrong_passed_schema_only"], 9)
        self.assertEqual(s["silent_ids"], ["c27", "c28", "c29", "c30"])

    def test_no_correct_call_is_rejected(self):
        # 이 목록에서는 검사기가 정상 호출을 막은 경우가 없다. 이 데이터에 한한 사실이다.
        self.assertNotIn("정상인데 거절됨", summarize()["counts"])

    def test_schema_check_examples(self):
        # 세 번째는 True가 정수로 통과하지 않는지 확인(bool 함정)
        self.assertEqual(schema_check({"order_id": "ORD-1001", "reason": "불량"}), ["필수 값 없음: amount"])
        self.assertEqual(schema_check({"order_id": "ORD-1001", "amount": "30000", "reason": "불량"}), ["자료형 틀림: amount"])
        self.assertEqual(schema_check({"order_id": "ORD-1001", "amount": True, "reason": "불량"}), ["자료형 틀림: amount"])

    def test_semantic_check_examples(self):
        self.assertEqual(semantic_check({"order_id": "ORD-2001", "amount": 1000}), ["다른 고객의 주문"])
        # 경계: 한도(30000)와 같으면 통과, 1원 넘으면 걸린다(off-by-one 확인)
        self.assertEqual(semantic_check({"order_id": "ORD-1003", "amount": 30001}), ["환불 가능 금액 초과"])
        self.assertEqual(semantic_check({"order_id": "ORD-1003", "amount": 30000}), [])

    def test_unit_error_passes_every_check(self):
        # 단위 실수(원이 아니라 천 원 단위로 쓴 것)는 형식도 의미도 통과한다. 의도를 아는 쪽이 있어야 잡힌다.
        call = {"order_id": "ORD-1002", "amount": 30, "reason": "불량"}  # 30000원 주문에 30
        self.assertEqual(schema_check(call), [])
        self.assertEqual(semantic_check(call), [])
        self.assertLess(call["amount"], ORDERS["ORD-1002"]["paid"])


if __name__ == "__main__":
    unittest.main()
