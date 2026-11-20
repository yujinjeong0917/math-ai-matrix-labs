import unittest

from claim_ledger import classify, load, tally


class Ledger(unittest.TestCase):
    def test_hand_tally(self):
        # 손계산: 숫자 규칙 2(1번, 4번), 절대 표현 1(3번), 이론 틀 1(5번), 추측 2(2번, 6번), 출처 없음 1(7번)
        t = tally(load())
        self.assertEqual(
            list(t.values()), [2, 1, 1, 2, 1])
        self.assertEqual(sum(t.values()), 7)

    def test_numeric_beats_source(self):
        self.assertEqual(classify("선택지는 5개 이하", "리뷰가 소개한 틀"), "숫자 규칙: 읽은 근거 없음")

    def test_absolute_beats_source(self):
        self.assertEqual(classify("안내는 반드시 접는다", "설계자 추측"), "절대 표현: 읽은 근거 없음")

    def test_plain_claim_uses_source(self):
        self.assertEqual(classify("단골에게는 군더더기일 수 있다", "설계자 추측"), "설계자 추측")

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            classify("가능하다", "어딘가")


if __name__ == "__main__":
    unittest.main()
