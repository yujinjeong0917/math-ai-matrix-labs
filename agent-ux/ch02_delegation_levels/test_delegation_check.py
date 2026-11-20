import unittest

import delegation_check as d

A, B, C = list(d.DESIGNS.values())


class Checks(unittest.TestCase):
    def test_every_step_has_a_type(self):
        self.assertTrue(all(t in d.TYPES for _, t, _ in d.STEPS))
        self.assertEqual(len(d.STEPS), 6)

    def test_design_lengths_match(self):
        for design in d.DESIGNS.values():
            self.assertEqual(len(design), len(d.STEPS))

    def test_hand_counts(self):
        # 손계산: A는 지급과 안내가 W1, 결정 지급 안내 기록이 W2 -> 2 + 4 = 6
        self.assertEqual(d.counts(A), {"W1": 2, "W2": 4, "합": 6})
        self.assertEqual(d.counts(B), {"W1": 0, "W2": 0, "합": 0})
        # C는 고객 안내가 되돌릴 수 없는데 알아서 실행 -> W1 하나
        self.assertEqual(d.counts(C), {"W1": 1, "W2": 0, "합": 1})

    def test_irreversible_auto_always_flagged(self):
        for visible in (True, False):
            design = [(d.LEVELS[0], True)] * 6
            design[3] = (d.AUTO, visible)
            self.assertIn(("W1", "환불 지급"), d.warnings(design))

    def test_visible_auto_reading_step_not_flagged(self):
        design = [(d.AUTO, False)] + [(d.LEVELS[0], True)] * 5
        self.assertEqual(d.warnings(design), [])  # 정보 획득은 W2 대상이 아니다

    def test_changed_steps(self):
        self.assertEqual(len(d.changed_steps(A, B)), 6 - 2)

    def test_cells(self):
        self.assertEqual(len(d.TYPES) * len(d.LEVELS), 12)


if __name__ == "__main__":
    unittest.main()
