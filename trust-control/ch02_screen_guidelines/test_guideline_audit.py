import unittest

from guideline_audit import GUIDELINES, PAPER_COUNTS, PAPER_TOTAL, SCREENS, paper_shares, rate, tally


class Audit(unittest.TestCase):
    def test_paper_sum(self):
        self.assertEqual(313 + 277 + 89 + 106, 785)
        self.assertEqual(sum(PAPER_COUNTS.values()), PAPER_TOTAL)

    def test_shares(self):
        s = paper_shares()
        self.assertAlmostEqual(sum(s.values()), 1.0)
        self.assertEqual(round(s["적용"] * 100, 1), 39.9)
        self.assertEqual(round(s["위반"] * 100, 1), 35.3)
        self.assertEqual(round(s["중립"] * 100, 1), 11.3)
        self.assertEqual(round(s["해당 없음"] * 100, 1), 13.5)

    def test_screens(self):
        names = list(SCREENS)
        t = [tally(rate(SCREENS[n])) for n in names]
        self.assertEqual((t[0]["적용"], t[0]["위반"]), (1, 5))
        self.assertEqual((t[1]["적용"], t[1]["위반"]), (3, 3))
        self.assertEqual((t[2]["적용"], t[2]["위반"]), (6, 0))

    def test_not_applicable(self):
        r = rate({1: True, 2: None})
        self.assertEqual(r[2], "해당 없음")
        self.assertEqual(r[8], "해당 없음")  # 값이 없으면 해당 없음으로 센다
        self.assertEqual(len(r), len(GUIDELINES))


if __name__ == "__main__":
    unittest.main()
