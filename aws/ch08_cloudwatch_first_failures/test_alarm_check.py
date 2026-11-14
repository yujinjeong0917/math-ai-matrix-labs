import json
import unittest

import alarm_check as a


class AlarmCheckTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(a.us_to_s(31_500_000), 31.5)
        self.assertEqual(a.ms_to_s(29_000), 29)
        self.assertEqual(a.normalize_gpu(200, 4), 50)

    def test_m_out_of_n(self):
        v = [0, 1, 0, 1]
        self.assertEqual(a.evaluate_alarm(v, 0.5, 3, 2), a.ALARM)      # 최근 3개 [1,0,1] 중 2개가 넘음
        self.assertEqual(a.evaluate_alarm([1, 0, 0], 0.5, 3, 2), a.OK)
        self.assertEqual(a.evaluate_alarm([1, 1, 0], 0.5, 3, 2), a.ALARM)
        self.assertEqual(a.evaluate_alarm([1, 1, 1], 0.5, 3, 3), a.ALARM)
        self.assertEqual(a.evaluate_alarm([1, 1, 0], 0.5, 3, 3), a.OK)

    def test_missing_data_modes_when_all_missing(self):
        v = [None] * 5
        self.assertEqual(a.evaluate_alarm(v, 0, treat_missing="notBreaching"), a.OK)
        self.assertEqual(a.evaluate_alarm(v, 0, treat_missing="breaching"), a.ALARM)
        self.assertEqual(a.evaluate_alarm(v, 0, treat_missing="missing"), a.INSUFFICIENT)
        self.assertEqual(a.evaluate_alarm(v, 0, treat_missing="ignore", previous_state=a.ALARM), a.ALARM)
        with self.assertRaises(ValueError):
            a.evaluate_alarm(v, 0, treat_missing="nope")

    def test_scenario_first_alarm_minutes(self):
        with open("scenario.json", encoding="utf-8") as f:
            r = a.run(json.load(f))["first_alarm_minute"]
        self.assertEqual(r["backlog_without_capacity"], 2)   # 분 2: 용량 0인데 줄이 있음 2분 연속
        self.assertEqual(r["oldest_age"], 3)                 # 분 3: 가장 오래된 요청이 130초
        self.assertEqual(r["model_latency_vs_gateway"], 3)
        self.assertEqual(r["gpu_one_card"], 5)
        self.assertIsNone(r["expired"])
        self.assertIsNone(r["errors_5xx_notBreaching"])
        self.assertIsNone(r["errors_5xx_missing"])
        self.assertEqual(r["wrong_unit_alarm_minutes"], 5)

    def test_leading_indicator_comes_first(self):
        with open("scenario.json", encoding="utf-8") as f:
            r = a.run(json.load(f))["first_alarm_minute"]
        self.assertLess(r["backlog_without_capacity"], r["oldest_age"])

    def test_one_of_one_fires_a_minute_earlier(self):
        with open("scenario.json", encoding="utf-8") as f:
            s = json.load(f)["series"]["HasBacklogWithoutCapacity"]
        self.assertEqual(a.first_index(a.replay(s, 0.5)), 1)
        self.assertEqual(a.first_index(a.replay(s, 0.5, evaluation_periods=2, datapoints_to_alarm=2)), 2)

    def test_quiet_errors_look_different_by_mode(self):
        with open("scenario.json", encoding="utf-8") as f:
            st = a.run(json.load(f))["states"]
        self.assertEqual(set(st["errors_5xx_notBreaching"]), {a.OK})
        self.assertEqual(set(st["errors_5xx_missing"]), {a.INSUFFICIENT})


if __name__ == "__main__":
    unittest.main()
