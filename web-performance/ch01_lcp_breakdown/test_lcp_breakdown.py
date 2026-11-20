import unittest

import lcp_breakdown as m


class Breakdown(unittest.TestCase):
    def test_totals_by_hand(self):
        self.assertAlmostEqual(m.total(m.BEFORE), 2.8)          # 0.9+0.5+0.9+0.5
        self.assertAlmostEqual(m.total(m.SHRUNK_IMAGE), 2.8)    # 0.9+0.5+0.5+0.9
        self.assertAlmostEqual(m.total(m.EARLY_DISCOVERY), 2.0) # 0.9+0.1+0.5+0.5

    def test_shares_by_hand(self):
        # 0.9/2.8 = 32.14%, 0.5/2.8 = 17.86%
        self.assertEqual(m.shares(m.BEFORE), [32.1, 17.9, 32.1, 17.9])
        self.assertEqual(m.shares(m.SHRUNK_IMAGE), [32.1, 17.9, 17.9, 32.1])
        self.assertAlmostEqual(sum(m.shares(m.BEFORE)), 100, delta=0.2)

    def test_gap_from_guide(self):
        self.assertEqual(m.gap_from_guide(m.BEFORE), [-7.9, 7.9, -7.9, 7.9])

    def test_time_moves_not_disappears(self):
        # 로드 시간이 0.4초 줄었지만 렌더 지연이 0.4초 늘어 합은 그대로다
        self.assertAlmostEqual(m.SHRUNK_IMAGE[2] - m.BEFORE[2], -0.4)
        self.assertAlmostEqual(m.SHRUNK_IMAGE[3] - m.BEFORE[3], 0.4)
        self.assertAlmostEqual(m.total(m.SHRUNK_IMAGE), m.total(m.BEFORE))

    def test_biggest_part(self):
        self.assertEqual(m.biggest_part(m.BEFORE), "TTFB · 자원 로드 시간")  # 같은 길이면 모두 적는다
        self.assertEqual(m.biggest_part(m.SHRUNK_IMAGE), "TTFB · 요소 렌더 지연")

    def test_reported_lcp(self):
        self.assertEqual(m.reported_lcp(m.CANDIDATES), (2.8, "대표 상품 이미지"))
        self.assertEqual(m.reported_lcp(m.CANDIDATES, 2.0), (1.2, "상품명 텍스트 블록"))
        self.assertEqual(m.reported_lcp(m.CANDIDATES, 2.5), (2.1, "배너 이미지"))
        with self.assertRaises(ValueError):
            m.reported_lcp(m.CANDIDATES, 1.0)


class Quiz(unittest.TestCase):
    def test_quiz(self):
        self.assertAlmostEqual(m.total(m.QUIZ_PARTS), 2.4)
        self.assertEqual(m.shares(m.QUIZ_PARTS)[0], 33.3)
        self.assertEqual(m.reported_lcp(m.CANDIDATES, 2.3), (2.1, "배너 이미지"))


if __name__ == "__main__":
    unittest.main()
