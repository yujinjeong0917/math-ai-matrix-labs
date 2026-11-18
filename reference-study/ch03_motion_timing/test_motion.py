import unittest
import motion as m


class Test(unittest.TestCase):
    def test_frames_hand_calc(self):
        # 손계산: 0.12초 x 60 = 7.2, 0.2초 x 60 = 12
        self.assertAlmostEqual(m.frames(0.12), 7.2)
        self.assertAlmostEqual(m.frames(0.2), 12.0)

    def test_frames_120(self):
        self.assertAlmostEqual(m.frames(0.12, 120), 14.4)
        self.assertAlmostEqual(m.frames(0.2, 120), 24.0)

    def test_bezier_ends_and_monotone(self):
        self.assertAlmostEqual(m.bezier_progress(0), 0, places=6)
        self.assertAlmostEqual(m.bezier_progress(1), 1, places=6)
        v = [m.bezier_progress(i / 20) for i in range(21)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(v, v[1:])))
        self.assertGreater(m.bezier_progress(0.25), 0.25)  # ease는 초반이 빠르다

    def test_linear_is_identity(self):
        self.assertAlmostEqual(m.scale_at(0.5, easing="linear"), 0.975)

    def test_json(self):
        items = m.load()
        self.assertEqual(len(items), 7)
        self.assertEqual(m.summary(items), {0.12: 6, 0.2: 1})
        self.assertEqual(len(m.without_reduced_motion(items)), 4)

    def test_point_on_curve(self):
        x, y = m.bezier_point(0.7)
        self.assertAlmostEqual(x, 0.5005, places=4)
        self.assertAlmostEqual(y, 0.8029, places=4)
        self.assertAlmostEqual(m.bezier_progress(0.5), 0.8024, places=4)

    def test_elsewhere(self):
        ew = m.load_elsewhere()
        self.assertEqual((ew["inline_transition_rules"], ew["keyframes_or_animation"]), (0, 0))

    def test_button(self):
        self.assertAlmostEqual(m.button_height(), 41.8)


if __name__ == "__main__":
    unittest.main()
