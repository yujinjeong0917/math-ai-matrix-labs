import math
import unittest

from service_gap import SERVICE_QUESTIONS, missing_items, traffic, verdict


class TrafficTest(unittest.TestCase):
    # 기대값: 86400/20 = 4320초, 20 x 30 = 600초 = 10분, 20 x 30 / 86400 = 0.0069444
    def test_20_per_day(self):
        t = traffic(20, 30)
        self.assertEqual(t["gap_seconds"], 4320)
        self.assertEqual(t["busy_seconds_per_day"], 600)
        self.assertAlmostEqual(t["busy_hours_per_day"], 10 / 60)  # 부동소수점이라 정확 비교 대신 근사 비교
        self.assertAlmostEqual(t["avg_concurrency"], 0.0069444, places=6)
        self.assertEqual(t["min_workers_for_average"], 1)

    # 기대값: 20000 x 30 = 600000초 = 166.67시간, 평균 동시 6.9444를 올림해 작업자 7
    def test_20000_per_day(self):
        t = traffic(20000, 30)
        self.assertAlmostEqual(t["gap_seconds"], 4.32)
        self.assertEqual(t["busy_seconds_per_day"], 600000)
        self.assertAlmostEqual(t["busy_hours_per_day"], 166.6667, places=4)
        self.assertAlmostEqual(t["avg_concurrency"], 6.9444, places=4)
        self.assertEqual(t["min_workers_for_average"], 7)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            traffic(0, 30)


class ChecklistTest(unittest.TestCase):
    def test_eight_questions(self):
        self.assertEqual(len(SERVICE_QUESTIONS), 8)

    def test_notebook_state(self):
        # README 기대 출력과 같은 상태: 없음 6, 미확인(None) 2, 갖춤 0
        have = {"entry": False, "auth": False, "input_limit": False, "artifact": None,
                "permission": False, "observe": False, "rollout": False, "budget": None}
        r = missing_items(have)
        self.assertEqual(len(r["missing"]), 6)
        self.assertEqual(r["hold"], ["artifact", "budget"])
        self.assertEqual(verdict(have), "NOT_READY")

    def test_hold_when_unknown(self):
        have = {k: True for k, _ in SERVICE_QUESTIONS}
        have["budget"] = None
        self.assertEqual(verdict(have), "HOLD")
        del have["budget"]  # 키가 없는 것도 미확인으로 본다
        self.assertEqual(verdict(have), "HOLD")
        have["budget"] = True
        self.assertEqual(verdict(have), "READY")


if __name__ == "__main__":
    unittest.main()
