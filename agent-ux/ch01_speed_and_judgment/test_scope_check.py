import unittest

import scope_check as s


class Checks(unittest.TestCase):
    def test_refund_matches_only_length_range(self):
        c = s.CASES["가상 환불 진행 화면"]
        same = [k for k, ok in s.compare(c) if ok]
        self.assertEqual(same, ["대기 길이(초)"])
        self.assertEqual(s.n_same(c), 1)

    def test_chat_helper_still_differs_in_how_delay_is_made(self):
        c = s.CASES["가상 채팅 답변 도우미"]
        self.assertEqual(s.n_same(c), 5)
        diff = [k for k, ok in s.compare(c) if not ok]
        self.assertIn("지연을 만든 방식", diff)

    def test_range_edges(self):
        c = dict(s.CASES["가상 환불 진행 화면"])
        c["대기 길이(초)"] = 25
        self.assertFalse(dict(s.compare(c))["대기 길이(초)"])
        c["대기 길이(초)"] = 2
        self.assertTrue(dict(s.compare(c))["대기 길이(초)"])

    def test_mean_differences(self):
        self.assertAlmostEqual(s.mean_diff(6.09, 5.76), 0.33)
        self.assertAlmostEqual(s.mean_diff(6.11, 5.76), 0.35)
        self.assertAlmostEqual(s.mean_diff(6.44, 6.19), 0.25)

    def test_d(self):
        self.assertEqual(s.d_approx(s.THOUGHTFUL["2초"], s.THOUGHTFUL["9초"]), 0.31)
        self.assertEqual(s.d_approx(s.THOUGHTFUL["2초"], s.THOUGHTFUL["20초"]), 0.34)

    def test_headroom_and_n(self):
        self.assertAlmostEqual(s.headroom(5.76), 1.24)
        self.assertAlmostEqual(s.headroom(6.09), 0.91)
        self.assertEqual(s.total_n(), 240)
        self.assertEqual(309 - 6 - 63, 240)
        self.assertEqual(s.excluded_share(), 20.8)


if __name__ == "__main__":
    unittest.main()
