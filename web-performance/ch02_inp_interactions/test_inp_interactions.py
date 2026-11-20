import unittest

import inp_interactions as m


class Inp(unittest.TestCase):
    def test_latency_by_hand(self):
        self.assertEqual([m.latency(i) for i in m.VISIT_A], [110, 80, 170, 220, 380])

    def test_visit_inp_is_slowest_when_few(self):
        lat = [m.latency(i) for i in m.VISIT_A]
        self.assertEqual(m.visit_inp(lat), 380)
        self.assertEqual(m.grade(380), "개선 필요")

    def test_ignore_rule(self):
        self.assertEqual(m.ignored_count(49), 0)
        self.assertEqual(m.ignored_count(60), 1)
        lat = m.long_visit_latencies()
        self.assertEqual(len(lat), 60)
        self.assertEqual(m.visit_inp(lat), 220)  # 380이 빠지고 다음으로 느린 220
        self.assertEqual(m.grade(220), "개선 필요")

    def test_payment_wait_is_not_in_the_tap_latency(self):
        self.assertEqual(m.latency(m.PAY_TAP), 60)
        self.assertEqual(m.grade(60), "좋음")
        self.assertEqual(m.latency(m.PAY_TAP) + m.PAY_RESPONSE_MS, 6060)

    def test_p75_over_visits(self):
        self.assertEqual(sorted(m.VISIT_INPS), [90, 110, 140, 150, 160, 180, 220, 380])
        self.assertEqual(m.p75_rank(m.VISIT_INPS), 180)  # 8개이면 6번째
        self.assertEqual(sum(1 for v in m.VISIT_INPS if v > 200), 2)

    def test_grades(self):
        self.assertEqual(m.grade(200), "좋음")
        self.assertEqual(m.grade(500), "개선 필요")
        self.assertEqual(m.grade(501), "나쁨")


class Quiz(unittest.TestCase):
    def test_quiz(self):
        self.assertEqual(m.latency(m.QUIZ_TAP), 165)
        self.assertEqual(m.grade(165), "좋음")
        self.assertEqual(m.ignored_count(110), 2)


if __name__ == "__main__":
    unittest.main()
