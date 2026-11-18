import unittest
import gray_box as m


class Test(unittest.TestCase):
    def test_hand_calc(self):
        # 손계산: 320x160=51200, 320x48=15360, 320x120=38400, 320x32=10240, 320x56=17920
        self.assertEqual([w * h for _, w, h, _ in m.BOXES], [51200, 15360, 38400, 10240, 17920])

    def test_check(self):
        ok, rho = m.gray_box_check(m.BOXES)
        self.assertTrue(ok)
        self.assertAlmostEqual(rho, 0.5)

    def test_broken(self):
        boxes = [(n, w, 140 if n == "공지 띠" else h, r) for n, w, h, r in m.BOXES]
        ok, rho = m.gray_box_check(boxes)
        self.assertTrue(ok)
        self.assertLess(rho, 0.5)

    def test_spearman_perfect(self):
        self.assertAlmostEqual(m.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(m.spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_ratio(self):
        self.assertAlmostEqual(m.ratio_chain([56, 24, 17])[0], 56 / 24)
        self.assertEqual(m.shared_values(m.REF_SIZES_1440, m.NEW_SIZES), [])

    def test_rank_gaps_hand_calc(self):
        # 손계산: d = 면적 순위 - 의도 순위 = 0, 2, -1, 1, -2 -> 제곱합 10
        self.assertEqual([d for *_, d in m.rank_gaps(m.BOXES)], [0, 2, -1, 1, -2])
        self.assertEqual(m.squared_gap_sum(m.BOXES), 10)
        self.assertAlmostEqual(1 - 6 * 10 / (5 * 24), 0.5)
        broken = [(n, w, 140 if n == "공지 띠" else h, r) for n, w, h, r in m.BOXES]
        self.assertEqual([d for *_, d in m.rank_gaps(broken)], [0, 3, 0, -2, -1])
        self.assertEqual(m.squared_gap_sum(broken), 14)

    def test_reverse_is_minus_one(self):
        # 정반대 순위(n=5)의 제곱합은 40, rho = -1
        self.assertAlmostEqual(m.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0)

    def test_all_same_size_has_no_rho(self):
        same = [(n, 320, 80, r) for n, w, h, r in m.BOXES]
        self.assertIsNone(m.gray_box_check(same)[1])

    def test_narrow_box_shows_area_vs_height(self):
        self.assertAlmostEqual(m.rho_by_height(m.BOXES), 0.5)  # 너비가 모두 같으면 면적 순위와 높이 순위가 같다
        narrow = [(n, 120 if n == "추천 목록" else w, h, r) for n, w, h, r in m.BOXES]
        self.assertAlmostEqual(m.gray_box_check(narrow)[1], 0.4)
        self.assertAlmostEqual(m.rho_by_height(narrow), 0.5)

    def test_ratio_gap_and_total(self):
        self.assertAlmostEqual(m.ratio_gap(m.REF_SIZES_1440, m.NEW_SIZES), 0.106, places=3)
        self.assertAlmostEqual(m.total_ratio(m.REF_SIZES_1440), 56 / 17)
        self.assertAlmostEqual(m.total_ratio(m.NEW_SIZES), 44 / 15)

    def test_transfer_values(self):
        self.assertEqual((m.HOME_HERO_PADDING_TOP, m.TRACKS_HERO_PADDING_TOP), (96, 56))
        self.assertAlmostEqual(m.H1_1440 / m.BODY, 3.294, places=3)


if __name__ == "__main__":
    unittest.main()
