import unittest

from delegation_map import (CHI_PAPERS, FEATURES, PATTERN_COUNTS, TOTAL_CODED, chi_share, n_with_accept_event,
                            pattern_shares, questions)


class Delegation(unittest.TestCase):
    def test_zhang_constants(self):
        self.assertEqual(sum(PATTERN_COUNTS.values()), 106)
        self.assertEqual((TOTAL_CODED, CHI_PAPERS), (134, 108))

    def test_hand_calc_shares(self):
        s = pattern_shares()
        self.assertAlmostEqual(s["반응"], 34 / 106)
        self.assertEqual(round(s["반응"] * 100), 32)
        self.assertEqual(round(s["협력"] * 100), 31)
        self.assertEqual(round(s["수동"] * 100), 17)
        self.assertEqual(round(s["반능동"] * 100), 11)
        self.assertEqual(round(s["선제"] * 100), 8)

    def test_chi_share(self):
        self.assertEqual(round(chi_share() * 100), 81)

    def test_six_features_two_with_accept_events(self):
        self.assertEqual(len(FEATURES), 6)
        self.assertEqual(n_with_accept_event(), 2)

    def test_auto_send_gets_three_questions(self):
        f = next(x for x in FEATURES if x["name"] == "답변 자동 발송")
        self.assertEqual(len(questions(f)), 3)

    def test_dynamic_feature_asks_about_switching(self):
        f = next(x for x in FEATURES if x["dynamic"])
        self.assertTrue(any("바뀌는 기준" in q for q in questions(f)))

    def test_search_feature_has_only_accept_question(self):
        f = FEATURES[0]
        self.assertEqual(len(questions(f)), 1)


if __name__ == "__main__":
    unittest.main()
