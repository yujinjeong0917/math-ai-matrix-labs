import unittest

import vitals_verdict as v


class Verdict(unittest.TestCase):
    def test_hand_calc_mobile_lcp(self):
        # 10개이면 올림(0.75 x 10) = 8번째
        self.assertEqual(sorted(v.LCP_MOBILE)[7], 2.9)
        self.assertEqual(v.p75_rank(v.LCP_MOBILE), 2.9)

    def test_hand_calc_desktop_lcp(self):
        self.assertEqual(v.p75_rank(v.LCP_DESKTOP), 1.8)

    def test_pooled_hides_mobile(self):
        pooled = v.LCP_MOBILE + v.LCP_DESKTOP
        self.assertEqual(len(pooled), 20)
        self.assertEqual(v.p75_rank(pooled), 2.4)
        self.assertEqual(v.grade_lcp(2.4), "좋음")
        self.assertEqual(v.grade_lcp(v.p75_rank(v.LCP_MOBILE)), "좋음 아님")

    def test_share_over(self):
        self.assertAlmostEqual(v.share_over(v.LCP_MOBILE, 2.5), 0.4)
        self.assertAlmostEqual(v.share_over(v.LCP_DESKTOP, 2.5), 0.1)

    def test_grades(self):
        self.assertEqual(v.grade_inp(200), "좋음")
        self.assertEqual(v.grade_inp(240), "개선 필요")
        self.assertEqual(v.grade_inp(501), "나쁨")
        self.assertEqual(v.grade_cls(0.1), "좋음")
        self.assertEqual(v.grade_cls(0.18), "개선 필요")
        self.assertEqual(v.grade_cls(0.26), "나쁨")

    def test_all_three_must_pass(self):
        m = v.segment_row(v.LCP_MOBILE, 240, 0.08)
        d = v.segment_row(v.LCP_DESKTOP, 120, 0.04)
        self.assertFalse(m["pass"])
        self.assertTrue(d["pass"])
        # LCP만 좋아도 INP가 좋음이 아니면 통과가 아니다
        self.assertFalse(v.segment_row([1.0], 240, 0.01)["pass"])


if __name__ == "__main__":
    unittest.main()
