import unittest

from feedback_spec import (FANG_MANIPULATED, NEVER_MANIPULATED, WANG_MANIPULATED, f_p_value_df1, degrees_per_frame, frames, linear_fill,
                           total_turns)


class Spec(unittest.TestCase):
    def test_frames(self):
        self.assertEqual(frames(15000), 900)
        self.assertEqual(frames(200), 12)
        self.assertEqual(frames(16.7), 1)

    def test_rotation(self):
        self.assertAlmostEqual(degrees_per_frame(1), 6.0)
        self.assertEqual(total_turns(1, 15), 15)

    def test_linear_fill(self):
        self.assertEqual(linear_fill(7.5, 15), 0.5)
        self.assertEqual(linear_fill(20, 15), 1.0)

    def test_wang_manipulations_by_experiment(self):
        self.assertEqual(WANG_MANIPULATED[1], ["움직임 유무"])
        self.assertEqual(WANG_MANIPULATED[2], ["움직임 유무"])
        self.assertIn("과제 몰입 지시문", WANG_MANIPULATED[3])
        self.assertIn("탐색 목적 지시문", WANG_MANIPULATED[4])
        text = " ".join(x for v in WANG_MANIPULATED.values() for x in v)
        self.assertNotIn("스켈레톤", text)

    def test_p_value_from_reported_F(self):
        # Fang 외 표 4의 장식 F=1.108, 오차 자유도 21: p는 약 0.30(표의 0.304와 같은 값)
        self.assertAlmostEqual(f_p_value_df1(1.108, 21), 0.3045, places=3)
        self.assertAlmostEqual(f_p_value_df1(4.0, 60), 0.0501, places=3)  # t=2, df=60 양측 p

    def test_fang_set(self):
        self.assertEqual(len(FANG_MANIPULATED), 3)
        joined = " ".join(FANG_MANIPULATED)
        self.assertNotIn("스켈레톤", joined)
        self.assertNotIn("스피너", joined)
        self.assertIn("스켈레톤 화면", NEVER_MANIPULATED)


if __name__ == "__main__":
    unittest.main()
