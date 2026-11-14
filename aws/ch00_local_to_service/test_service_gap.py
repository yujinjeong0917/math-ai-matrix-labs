import math
import unittest

from service_gap import SERVICE_QUESTIONS, missing_items, traffic, verdict


class TrafficTest(unittest.TestCase):
    def test_20_per_day(self):
        t = traffic(20, 30)
        self.assertEqual(t["gap_seconds"], 4320)
        self.assertEqual(t["busy_seconds_per_day"], 600)
        self.assertAlmostEqual(t["busy_hours_per_day"], 10 / 60)
        self.assertAlmostEqual(t["avg_concurrency"], 0.0069444, places=6)
        self.assertEqual(t["min_workers_for_average"], 1)

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
        del have["budget"]
        self.assertEqual(verdict(have), "HOLD")
        have["budget"] = True
        self.assertEqual(verdict(have), "READY")


if __name__ == "__main__":
    unittest.main()
