import unittest

from attention_audit import both_irrelevant, differing, load, tally


class Audit(unittest.TestCase):
    def setUp(self):
        self.d = load()

    def test_hand_tally_first_buyer(self):
        # 손계산: 결정 6(옵션, 수량, 배송지, 결제 수단, 쿠폰, 주문 버튼), 확인 3, 무관 1
        self.assertEqual(tally(self.d, "첫 구매자"), {"결정": 6, "확인": 3, "무관": 1})

    def test_hand_tally_regular(self):
        # 손계산: 결정 3(옵션, 수량, 주문 버튼), 확인 3(배송지, 결제 수단, 총액), 무관 4
        self.assertEqual(tally(self.d, "단골"), {"결정": 3, "확인": 3, "무관": 4})

    def test_totals_equal_element_count(self):
        for p in self.d["profiles"]:
            self.assertEqual(sum(tally(self.d, p).values()), len(self.d["elements"]))

    def test_same_screen_differs_by_user(self):
        self.assertEqual(len(differing(self.d)), 5)
        self.assertEqual(both_irrelevant(self.d), ["추천 배너"])

    def test_unknown_class_rejected(self):
        bad = {"profiles": ["가"], "elements": [{"name": "x", "need": ["몰라"]}]}
        with self.assertRaises(ValueError):
            tally(bad, "가")


if __name__ == "__main__":
    unittest.main()
