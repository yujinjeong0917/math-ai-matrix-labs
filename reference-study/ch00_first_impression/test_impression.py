import unittest
import impression as m


class Test(unittest.TestCase):
    def test_hand_calc_600(self):
        # 손계산: 600 x 0.062 = 37.2px, 34와 56 사이
        self.assertAlmostEqual(m.sizes_at(600)["hero"], 37.2)

    def test_min_and_max(self):
        self.assertEqual(m.sizes_at(390)["hero"], 34)   # 24.18px < 34px, 최소로 고정
        self.assertEqual(m.sizes_at(1440)["hero"], 56)  # 89.28px > 56px, 최대로 고정

    def test_ranking_top_is_h1(self):
        for vw in m.VIEWPORTS:
            self.assertEqual(m.ranking(vw)[0][0], "h1 제목")

    def test_ratio(self):
        self.assertAlmostEqual(m.hero_ratio(1440), 56 / 17)

    def test_breakpoints(self):
        self.assertAlmostEqual(m.viewport_where("hero", "max"), 903.2258, places=3)
        self.assertAlmostEqual(m.viewport_where("hero", "min"), 548.3871, places=3)


    def test_lead_breakpoints(self):
        self.assertAlmostEqual(m.viewport_where("lead", "min"), 791.6667, places=3)
        self.assertAlmostEqual(m.viewport_where("lead", "max"), 1000.0, places=3)
        self.assertEqual(m.HERO_PADDING_TOP, 96)


if __name__ == "__main__":
    unittest.main()
