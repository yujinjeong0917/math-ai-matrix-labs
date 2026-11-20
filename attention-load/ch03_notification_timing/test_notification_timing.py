import unittest

from notification_timing import TIMINGS, load, tally, timing


def make(r, a, i):
    return {"text": "x", "relevant": r, "action_now": a, "irreversible": i}


class Timing(unittest.TestCase):
    def test_hand_tally(self):
        # 손계산: 지금 화면 안 2(1번, 3번), 조용히 1(2번), 짧은 알림 1(6번), 끝난 뒤 2(4번, 8번), 보내지 않음 2(5번, 7번)
        t = tally(load())
        self.assertEqual(list(t.values()), [2, 1, 1, 2, 2])
        self.assertEqual(sum(t.values()), 8)

    def test_all_eight_combinations_map_to_a_known_timing(self):
        for r in (True, False):
            for a in (True, False):
                for i in (True, False):
                    self.assertIn(timing(make(r, a, i)), TIMINGS)

    def test_relevant_ignores_irreversible(self):
        self.assertEqual(timing(make(True, True, True)), TIMINGS[0])
        self.assertEqual(timing(make(True, False, True)), TIMINGS[1])

    def test_irrelevant_irreversible_goes_now_short(self):
        self.assertEqual(timing(make(False, False, True)), TIMINGS[2])


if __name__ == "__main__":
    unittest.main()
