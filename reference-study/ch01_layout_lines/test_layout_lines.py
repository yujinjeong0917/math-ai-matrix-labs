import unittest
import layout_lines as m


class Test(unittest.TestCase):
    def test_content_width(self):
        self.assertEqual(m.content_width(1440, 1200), 1156)
        self.assertEqual(m.content_width(390, 1200), 358)  # 419px 이하에서는 좌우 16px
        self.assertEqual(m.content_width(419, 1200), 387)
        self.assertEqual(m.content_width(420, 1200), 376)  # 22px 규칙으로 돌아온다
        self.assertEqual(m.content_width(1024, 1200), 980)

    def test_columns_hand_calc(self):
        # 손계산: (769.66+16)/(340+16)=2.2 -> 2열
        n, cw = m.columns(769.6551724, 340, 16)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(cw, 376.83, places=2)
        self.assertEqual(m.columns(358, 340, 16)[0], 1)
        self.assertAlmostEqual((769.6551724 + 16) / (340 + 16), 2.207, places=3)

    def test_group_layout(self):
        (hx, hw), (mx, mw) = m.group_layout(1440)
        self.assertAlmostEqual(hx, 142)
        self.assertAlmostEqual(hw + 40 + mw, 1156)
        self.assertAlmostEqual(mw / hw, 2 / 0.9)

    def test_lines(self):
        boxes, n, cw = m.start_section_boxes()
        left = dict((v, names) for v, names in m.find_lines(boxes, "left"))
        self.assertEqual(len(left), 1)
        names = list(left.values())[0]
        self.assertEqual(set(names), {"kicker", "h2 제목", "설명"})
        right_edge = m.find_lines(boxes, "right")
        self.assertEqual(right_edge, [])  # 오른쪽 끝이 같은 상자 쌍은 없다
        self.assertEqual(len(m.find_lines(boxes, "top")), 1)

    def test_height_does_not_change_lines(self):
        boxes, _, _ = m.start_section_boxes()
        doubled = [m.Box(b.name, b.x, b.y, b.w, b.h * 2) for b in boxes]
        for k in ("left", "right", "top"):
            self.assertEqual(m.find_lines(doubled, k), m.find_lines(boxes, k))

    def test_center_line(self):
        lines = m.find_lines(m.hero_boxes(), "cx")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0][0], 720)
        self.assertEqual(len(lines[0][1]), 5)


if __name__ == "__main__":
    unittest.main()
