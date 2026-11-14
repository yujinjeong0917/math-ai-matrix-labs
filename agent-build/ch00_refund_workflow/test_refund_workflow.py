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
        self.assertEqual(s["by_kind"], {"환불": 12, "교환": 4, "배송조회": 4})

    def test_totals(self):
        s = summarize()
        self.assertEqual(s["total_calls"], 96)
        self.assertEqual(s["total_writes"], 53)
        self.assertEqual(s["total_irreversible"], 30)
        self.assertEqual(s["total_money_out"], 10)

    def test_refund_only_mean_differs_from_all_20_mean(self):
        s = summarize()
        self.assertAlmostEqual(s["naive_mean_calls_all20"], 4.8)
        self.assertAlmostEqual(s["refund_mean_calls"], 61 / 12)
        self.assertAlmostEqual(s["refund_mean_irreversible"], 22 / 12)
        self.assertAlmostEqual(s["naive_mean_irreversible_all20"], 1.5)

    def test_paid_refunds(self):
        s = summarize()
        self.assertEqual(s["paid_refund_cases"], 9)
        self.assertEqual(s["paid_first_irreversible_at"], [3, 3, 4, 3, 4, 3, 4, 3, 3])
        self.assertAlmostEqual(s["paid_mean_reads_before_first_irreversible"], 21 / 9)

    def test_profile_of_a_single_case(self):
        q01 = next(q for q in load_inquiries() if q["id"] == "q01")
        self.assertEqual(profile(q01)["irreversible"], 2)  # issue_refund + send_message
        self.assertEqual(profile(q01)["reads"], 2)


if __name__ == "__main__":
    unittest.main()
