import unittest
import contrast as m


class Test(unittest.TestCase):
    def test_black_white(self):
        self.assertAlmostEqual(m.contrast("#000000", "#ffffff"), 21.0)  # WCAG 범위의 끝
        self.assertAlmostEqual(m.contrast("#ffffff", "#ffffff"), 1.0)

    def test_hand_calc_gray(self):
        # 손계산: #777777 -> 0x77=119, 119/255=0.4667, ((0.4667+0.055)/1.055)^2.4=0.1845
        self.assertAlmostEqual(m.luminance("#777777"), 0.1845, places=3)
        self.assertAlmostEqual(m.contrast("#777777", "#ffffff"), 4.48, places=2)

    def test_site_pairs(self):
        c = lambda f, b: m.contrast(m.COLORS[f], m.COLORS[b])
        self.assertGreater(c("ink", "bg"), 7)
        self.assertGreater(c("btn-ink", "btn-bg"), 4.5)
        self.assertAlmostEqual(c("old-ink-2", "bg"), 4.3, places=1)  # CSS 주석과 같은 값

    def test_large(self):
        self.assertTrue(m.is_large(24))
        self.assertFalse(m.is_large(19))
        self.assertTrue(m.is_large(18.67, True))
        self.assertFalse(m.is_large(18, True))

    def test_verdict(self):
        self.assertEqual(m.verdict(4.5), {"AA": True, "AAA": False})
        self.assertEqual(m.verdict(3.0, large=True), {"AA": True, "AAA": False})

    def test_scale(self):
        q = m.step_ratios(m.SCALE_1440)
        self.assertAlmostEqual(q[0][2], 1.4)
        self.assertEqual(len(q), 6)


if __name__ == "__main__":
    unittest.main()
