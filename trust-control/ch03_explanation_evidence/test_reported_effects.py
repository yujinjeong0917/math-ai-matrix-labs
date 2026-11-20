import unittest

from reported_effects import ROWS, TEXT_ONLY, includes_zero, midpoint, mismatches, tally, widths


class Reported(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(ROWS), 17)
        n_ex = sum(1 for r in ROWS if not includes_zero(r[4], r[5]))
        self.assertEqual(n_ex, 11)

    def test_includes_zero(self):
        self.assertTrue(includes_zero(-0.02, 0.42))
        self.assertFalse(includes_zero(0.03, 0.48))
        self.assertFalse(includes_zero(-0.61, -0.01))  # 음수 쪽 구간도 0을 포함하지 않는다

    def test_midpoints_close_to_d(self):
        for r in ROWS:
            self.assertLessEqual(abs(midpoint(r[4], r[5]) - r[3]), 0.025, r)

    def test_tally_by_experiment(self):
        t = tally()
        self.assertEqual(t[(1, "이해")], [3, 0])
        self.assertEqual(t[(1, "신뢰 보정")], [2, 3])
        self.assertEqual(t[(3, "이해")], [1, 2])
        self.assertEqual(t[(3, "신뢰 보정")], [2, 0])
        self.assertEqual(t[(1, "불확실성 인지")], [0, 1])
        self.assertEqual(t[(3, "불확실성 인지")], [1, 0])
        self.assertNotIn((2, "신뢰 보정"), t)  # 실험 2의 신뢰 보정은 값을 옮기지 못했다

    def test_text_only_exp2(self):
        self.assertEqual({(e, p) for e, p, _ in TEXT_ONLY}, {(2, "불확실성 인지"), (2, "신뢰 보정")})

    def test_widths(self):
        lo, hi = widths("이해")
        self.assertAlmostEqual(lo, 0.44)
        self.assertAlmostEqual(hi, 0.56)

    def test_mismatch_single(self):
        m = mismatches()
        self.assertEqual(len(m), 1)
        self.assertEqual((m[0][0], m[0][3]), (1, 0.19))


if __name__ == "__main__":
    unittest.main()
