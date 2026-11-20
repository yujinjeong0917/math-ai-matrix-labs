import unittest

import cls_shifts as m


class Cls(unittest.TestCase):
    def test_doc_examples(self):
        self.assertEqual([m.score(a, b) for a, b in m.DOC_EXAMPLES], [0.1875, 0.07, 0.18])

    def test_vertical_shift_by_hand(self):
        # 높이 0.5가 0.10 내려오면 영향 0.6, 점수 0.06
        self.assertEqual(m.vertical_shift_score(0.5, 0.10), 0.06)
        self.assertEqual(m.vertical_shift_score(0.45, 0.15), 0.09)
        self.assertEqual(m.vertical_shift_score(0.40, 0.05), 0.0225)

    def test_checkout_windows(self):
        ws = m.session_windows(m.checkout_shifts())
        self.assertEqual(len(ws), 2)
        self.assertEqual([w["n"] for w in ws], [3, 1])
        self.assertAlmostEqual(ws[0]["sum"], 0.1725)
        self.assertAlmostEqual(ws[1]["sum"], 0.0064)
        self.assertAlmostEqual(m.cls_of(m.checkout_shifts()), 0.1725)
        self.assertEqual(m.grade(0.1725), "개선 필요")

    def test_recent_input_is_excluded(self):
        with_input = sum(s for _, s, _ in m.checkout_shifts())
        self.assertGreater(with_input, 0.1725 + 0.0064)

    def test_window_caps_at_five_seconds(self):
        ws = m.session_windows(m.STEADY)
        self.assertEqual([w["n"] for w in ws], [7, 1])
        self.assertAlmostEqual(m.cls_of(m.STEADY), 0.14)
        self.assertAlmostEqual(sum(s for _, s, _ in m.STEADY), 0.16)

    def test_same_score_either_element(self):
        self.assertEqual(m.vertical_shift_score(0.5, 0.18), 0.1224)
        self.assertEqual(m.grade(0.1224), "개선 필요")

    def test_grade_edges(self):
        self.assertEqual(m.grade(0.1), "좋음")
        self.assertEqual(m.grade(0.25), "개선 필요")
        self.assertEqual(m.grade(0.2501), "나쁨")

    def test_no_shift(self):
        self.assertEqual(m.cls_of([]), 0.0)


class Quiz(unittest.TestCase):
    def test_quiz(self):
        self.assertEqual(m.score(0.5, 0.2), 0.1)
        self.assertEqual(m.grade(0.1), "좋음")
        self.assertAlmostEqual(m.cls_of(m.QUIZ_A), 0.12)
        self.assertAlmostEqual(m.cls_of(m.QUIZ_B), 0.07)
        self.assertEqual(m.grade(0.12), "개선 필요")


if __name__ == "__main__":
    unittest.main()
