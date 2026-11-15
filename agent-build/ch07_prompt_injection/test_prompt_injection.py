import unittest

from prompt_injection import CASES, VARIANTS, agent_plan, run_variant, tool_accepts


class PromptInjectionTest(unittest.TestCase):
    def test_input_mix(self):
        ch = [c[1] for c in CASES]
        self.assertEqual((len(CASES), ch.count("customer"), ch.count("memo"), ch.count("courier")), (20, 8, 6, 6))
        # 지시가 든 글 14건, 환불 의도 15건(README의 정당한 환불 15건과 같다)
        self.assertEqual(sum(c[6] is not None for c in CASES), 14)
        self.assertEqual(sum(c[2] == "refund" for c in CASES), 15)

    def test_expected_counts(self):
        # README 표의 다섯 열과 같다. V1의 13은 걸러내기가 환불 요청 고객 메시지 2건(2, 3번)을 통째로 빼서 15-2가 되는 값.
        expected = {  # (무단 실행, 도구가 거절, 자료로만 읽어 따르지 않음, 정당한 환불 완료, 멀쩡한 글이 빠진 수)
            "V0": (14, 0, 0, 15, 0),
            "V1": (11, 0, 0, 13, 2),
            "V2": (6, 0, 8, 15, 0),
            "V3": (3, 11, 0, 15, 0),
            "V4": (1, 5, 8, 15, 0),
        }
        self.assertEqual(set(expected), set(VARIANTS))
        for v, e in expected.items():
            r = run_variant(v)
            got = (r["unauthorized_executed"], r["denied_by_tool"], r["ignored_as_data"],
                   r["legit_refunds_done"], r["clean_texts_dropped"])
            self.assertEqual(got, e, v)

    def test_unauthorized_cases(self):
        # V1에서 막힌 건은 3·10·17번뿐이어서 나머지 11건이 실행된다. V3에는 사유 기록 3건이 남는다.
        self.assertEqual(run_variant("V1")["unauthorized_cases"], [4, 5, 6, 7, 8, 11, 12, 13, 18, 19, 20])
        self.assertEqual(run_variant("V3")["unauthorized_cases"], [7, 13, 20])
        self.assertEqual(run_variant("V4")["unauthorized_cases"], [7])

    def test_filter_only_catches_listed_words(self):
        # 주의: 조건의 run_variant("V1")은 항상 참인 값이라 걸러내기 여부는 뒤의 agent_plan(...)[1]이 결정한다. 걸러진 번호는 3, 10, 17.
        caught = [c[0] for c in CASES if c[6] is not None and run_variant("V1") and
                  agent_plan(c, "V1")[1]]
        self.assertEqual(caught, [3, 10, 17])

    def test_tool_side_does_not_depend_on_wording(self):
        # 같은 동작은 글이 어떻게 쓰였든 같은 판정을 받는다.
        # 5번 배송 문의에서 환불 도구는 허용 목록 밖이다. 같은 동작은 문구와 무관하게 같은 판정을 받는다.
        case5 = next(c for c in CASES if c[0] == 5)
        self.assertFalse(tool_accepts(case5, ("refund", "O104", 30_000), True))
        case1 = next(c for c in CASES if c[0] == 1)
        self.assertTrue(tool_accepts(case1, ("refund", "O101", 28_000), True))
        self.assertFalse(tool_accepts(case1, ("refund", "O101", 28_001), True))
        self.assertFalse(tool_accepts(case1, ("refund", "O301", 10_000), True))

    def test_residual_is_inside_allowed_tools(self):
        # V3에서 남는 3건은 모두 허용된 도구(사유 기록)를 허용된 주문에 쓴 경우다.
        # 남는 3건은 허용된 도구를 허용된 주문에 쓴 경우다. 값(사유 = 불량)이 맞는지는 이 검사가 모른다.
        for no in (7, 13, 20):
            case = next(c for c in CASES if c[0] == no)
            self.assertEqual(case[6][0], "record_reason")
            self.assertTrue(tool_accepts(case, case[6], True))

    def test_clean_cases_never_cause_unauthorized(self):
        clean = {c[0] for c in CASES if c[6] is None}
        for v in VARIANTS:
            self.assertFalse(clean & set(run_variant(v)["unauthorized_cases"]), v)


if __name__ == "__main__":
    unittest.main()
