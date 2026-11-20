import unittest

import supervision_design as s


class Checks(unittest.TestCase):
    def test_he_design_has_two_empty_cells(self):
        ct = s.crosstab(s.HE_TASKS)
        self.assertEqual(ct[("높음", False)], 3)
        self.assertEqual(ct[("낮음", True)], 3)
        self.assertEqual(sorted(s.empty_cells(s.HE_TASKS)), sorted([("높음", True), ("낮음", False)]))
        self.assertFalse(s.is_balanced(s.HE_TASKS))

    def test_balanced_design(self):
        d = s.balanced_design(6)
        self.assertEqual(len(d), 24)
        self.assertTrue(s.is_balanced(d))
        self.assertEqual(s.empty_cells(d), [])

    def test_log_sizes(self):
        for spec in s.LOG_SPEC.values():
            rows = s.expand(spec)
            self.assertEqual(len(rows), 24)
            for cell, counts in spec.items():
                self.assertEqual(sum(counts), 6)

    def test_hand_metrics_P(self):
        # 손계산: 오류 12단계 중 잡음 3+... (낮음 틀림 수정2+중단1=3, 높음 틀림 3+2=5) = 8 -> 66.7%
        m = s.metrics(s.expand(s.LOG_SPEC["화면 P"]))
        self.assertEqual((m["wrong"], m["caught"], m["missed"]), (12, 8, 4))
        self.assertEqual(m["catch_rate"], 66.7)
        # 맞은 12단계 중 괜히 멈춤 낮음 1 + 높음 2 = 3 -> 25.0%
        self.assertEqual((m["right"], m["false_alarm"], m["false_alarm_rate"]), (12, 3, 25.0))
        # 판단이 맞음 = 잡음 8 + 맞고 승인 9 = 17/24
        self.assertEqual(m["decision_accuracy"], 70.8)

    def test_hand_metrics_Q(self):
        m = s.metrics(s.expand(s.LOG_SPEC["화면 Q"]))
        self.assertEqual((m["caught"], m["catch_rate"]), (3, 25.0))
        self.assertEqual((m["false_alarm"], m["false_alarm_rate"]), (0, 0.0))
        self.assertEqual(m["decision_accuracy"], 62.5)

    def test_by_risk(self):
        rows = s.expand(s.LOG_SPEC["화면 P"])
        self.assertEqual(s.metrics(rows, "낮음")["catch_rate"], 50.0)
        self.assertEqual(s.metrics(rows, "높음")["catch_rate"], 83.3)

    def test_tradeoff_in_fake_log(self):
        p = s.metrics(s.expand(s.LOG_SPEC["화면 P"]))
        q = s.metrics(s.expand(s.LOG_SPEC["화면 Q"]))
        self.assertGreater(p["catch_rate"], q["catch_rate"])
        self.assertGreater(p["false_alarm_rate"], q["false_alarm_rate"])

    def test_checklist(self):
        a = s.checklist(s.SCREENS["초안(중간 수준 화면)"])
        b = s.checklist(s.SCREENS["고친 화면"])
        self.assertEqual(sum(a.values()), 1)
        self.assertEqual(sum(b.values()), 4)

    def test_limit_is_a_parameter(self):
        spec = s.SCREENS["초안(중간 수준 화면)"]
        self.assertTrue(s.checklist(spec, limit=6)["가장 긴 조용한 구간이 팀 한계 이하"])
        self.assertFalse(s.checklist(spec, limit=5)["가장 긴 조용한 구간이 팀 한계 이하"])


if __name__ == "__main__":
    unittest.main()
