import unittest

from refund_workflow import TOOLS, load_inquiries, profile, summarize


class RefundWorkflowTest(unittest.TestCase):
    def test_every_call_is_a_known_tool(self):
        for q in load_inquiries():
            for call in q["calls"]:
                self.assertIn(call, TOOLS)

    def test_counts_by_kind(self):
        s = summarize()
        self.assertEqual(s["cases"], 20)
        # 문의 20건 = 환불 12 + 교환 4 + 배송조회 4 (inquiries.json의 고정 구성)
        self.assertEqual(s["by_kind"], {"환불": 12, "교환": 4, "배송조회": 4})

    def test_totals(self):
        s = summarize()
        # README 기대 출력의 total_* 네 줄과 같은 값. 파일이 바뀌면 숫자도 함께 바꿔야 한다.
        self.assertEqual(s["total_calls"], 96)
        self.assertEqual(s["total_writes"], 53)
        self.assertEqual(s["total_irreversible"], 30)
        self.assertEqual(s["total_money_out"], 10)

    def test_refund_only_mean_differs_from_all_20_mean(self):
        s = summarize()
        # 96/20 = 4.8(전체 평균)과 61/12(환불만)가 다르다는 것을 확인한다. 분모 선택이 평균을 바꾼다.
        self.assertAlmostEqual(s["naive_mean_calls_all20"], 4.8)
        self.assertAlmostEqual(s["refund_mean_calls"], 61 / 12)
        self.assertAlmostEqual(s["refund_mean_irreversible"], 22 / 12)
        self.assertAlmostEqual(s["naive_mean_irreversible_all20"], 1.5)

    def test_paid_refunds(self):
        s = summarize()
        self.assertEqual(s["paid_refund_cases"], 9)
        # 순번 합 = 3*6 + 4*3 = 30, 앞선 호출 수 합 = 30 - 9 = 21 이므로 평균 21/9
        self.assertEqual(s["paid_first_irreversible_at"], [3, 3, 4, 3, 4, 3, 4, 3, 3])
        self.assertAlmostEqual(s["paid_mean_reads_before_first_irreversible"], 21 / 9)

    def test_profile_of_a_single_case(self):
        # q01은 issue_refund와 send_message가 되돌릴 수 없는 호출(TOOLS 분류 기준)
        q01 = next(q for q in load_inquiries() if q["id"] == "q01")
        self.assertEqual(profile(q01)["irreversible"], 2)  # issue_refund + send_message
        self.assertEqual(profile(q01)["reads"], 2)


if __name__ == "__main__":
    unittest.main()
