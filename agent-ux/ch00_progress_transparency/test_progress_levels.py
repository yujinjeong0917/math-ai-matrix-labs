import unittest

import progress_levels as p


class Checks(unittest.TestCase):
    def test_levels_have_expected_item_counts(self):
        self.assertEqual(len(p.LEVELS["높음"]), 10)
        self.assertEqual(len(p.LEVELS["중간"]), 3)
        self.assertEqual(len(p.LEVELS["낮음"]), 1)

    def test_steps_cover_total(self):
        self.assertEqual(p.STEPS[0][1], 0)
        self.assertEqual(p.STEPS[-1][2], p.TOTAL)
        for a, b in zip(p.STEPS, p.STEPS[1:]):
            self.assertEqual(a[2], b[1])

    def test_low_is_one_long_gap(self):
        r = p.level_summary("낮음")
        self.assertEqual((r["longest"], r["from"], r["to"]), (20, 0, 20))

    def test_mid_hand_values(self):
        # 손계산: 0,2,5,9,15,17,20 사이 간격 2,3,4,6,2,3
        self.assertEqual(p.update_times("중간"), [0, 2, 5, 9, 15, 17, 20])
        self.assertEqual([g[0] for g in p.gaps(p.update_times("중간"))], [2, 3, 4, 6, 2, 3])
        self.assertEqual(p.level_summary("중간")["longest"], 6)

    def test_high_still_has_six_second_gap_in_payment(self):
        r = p.level_summary("높음")
        self.assertEqual(p.update_times("높음"), [0, 1, 2, 4, 5, 8, 9, 15, 17, 20])
        self.assertEqual((r["longest"], r["from"], r["to"], r["step"]), (6, 9, 15, "환불 지급"))

    def test_extra_message_halves_gap(self):
        extra = p.with_extra(p.update_times("높음"), 12)
        self.assertEqual(p.longest_gap(extra)[0], 3)
        mid = p.with_extra(p.update_times("중간"), 12)
        self.assertEqual(p.longest_gap(mid), (4, 5, 9))

    def test_yu_differences(self):
        d = {k: (p.diff(m, 0, 2), p.diff(m, 0, 1)) for k, m in p.YU_MEANS.items()}
        self.assertEqual(d["신뢰"], (1.00, 0.26))
        self.assertEqual(d["만족"], (1.35, 0.38))
        self.assertEqual(d["사용 의향"], (1.20, 0.40))

    def test_preference_counts(self):
        self.assertEqual(p.YU_N - p.YU_PREFER_HIGH - p.YU_PREFER_MID_STUDENTS, 2)
        self.assertEqual(p.share(12, 40), 30.0)
        self.assertEqual(p.share(12, 33), 36.4)


if __name__ == "__main__":
    unittest.main()
