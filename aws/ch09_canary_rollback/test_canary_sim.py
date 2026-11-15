import json
import unittest
from pathlib import Path

import canary_sim as c

PROFILE = json.loads(Path(__file__).with_name("canary_profile.json").read_text(encoding="utf-8"))


class CanarySimTest(unittest.TestCase):
    def test_exposure_at_both_scales(self):
        self.assertAlmostEqual(c.exposed_requests(20, 5), 1)
        self.assertAlmostEqual(c.exposed_requests(20000, 5), 1000)

    # 149건이 경계: 149건은 95% 이상, 148건은 95% 미만
    def test_needed_samples_for_95_percent(self):
        self.assertEqual(c.needed_exposed(0.02), 149)
        self.assertEqual(c.needed_exposed(0.10), 29)
        self.assertGreaterEqual(c.detect_probability(149, 0.02), 0.95)
        self.assertLess(c.detect_probability(148, 0.02), 0.95)

    # 하루 노출 1건이면 149일, 하루 노출 1000건이면 149/1000 x 24 = 3.576시간(부동소수점이라 근사 비교)
    def test_days_and_hours_to_collect(self):
        self.assertAlmostEqual(c.days_to_collect(20, 5, 149), 149)
        self.assertAlmostEqual(24 * c.days_to_collect(20000, 5, 149), 3.576)

    def test_detect_probability_in_50(self):
        self.assertAlmostEqual(c.detect_probability(50, 0.02), 0.636, places=3)

    # 4건 이상이면 경보: 정상 2%에서 오경보 0.0178, 나쁜 10%에서 감지 0.7497
    def test_alarm_rates_trade_off(self):
        r = c.alarm_rates(PROFILE)
        self.assertAlmostEqual(r["false_alarm"], 0.0178, places=3)
        self.assertAlmostEqual(r["detect_bad_model"], 0.7497, places=3)
        self.assertGreater(r["detect_bad_model"], r["false_alarm"])

    # 29건 / 1건/일 = 29일, 29건 / 1000건/일 x 1440분 = 41.76분. 창 50건은 50일 또는 72분
    def test_new_rate_days_and_window_fill(self):
        r = c.report(PROFILE)
        self.assertAlmostEqual(r["days_small_new"], 29)
        self.assertAlmostEqual(r["minutes_scaled_new"], 41.76)
        self.assertAlmostEqual(r["window_fill_days_small"], 50)
        self.assertAlmostEqual(r["window_fill_minutes_scaled"], 72)

    def test_two_window_rules(self):
        tw = c.two_window_rates(c.alarm_rates(PROFILE))
        self.assertAlmostEqual(tw["both_false_alarm"], 0.0003, places=4)
        self.assertAlmostEqual(tw["both_detect"], 0.562, places=3)
        self.assertAlmostEqual(tw["either_false_alarm"], 0.035, places=3)
        self.assertAlmostEqual(tw["either_detect"], 0.937, places=3)
        self.assertLess(tw["both_detect"], c.alarm_rates(PROFILE)["detect_bad_model"])

    # 기준 3, 4, 5건의 (오경보, 감지) 표: 기준이 엄격할수록 둘 다 줄어드는 교환 관계
    def test_threshold_table(self):
        for k, fa, det in [(3, 0.078, 0.888), (4, 0.018, 0.750), (5, 0.003, 0.569)]:
            self.assertAlmostEqual(c.binomial_tail(50, 0.02, k), fa, places=3)
            self.assertAlmostEqual(c.binomial_tail(50, 0.10, k), det, places=3)

    # 경계: k=0이면 1, k>n이면 0, 동전 2번 모두 앞면 0.25
    def test_binomial_tail_edges(self):
        self.assertEqual(c.binomial_tail(10, 0.5, 0), 1.0)
        self.assertEqual(c.binomial_tail(10, 0.5, 11), 0.0)
        self.assertAlmostEqual(c.binomial_tail(2, 0.5, 2), 0.25)

    def test_alias_weights_follow_lambda_rule(self):
        w = c.alias_weights(0.05)
        self.assertAlmostEqual(w["first_version"], 0.95)
        self.assertEqual(w["additional_version"], 0.05)
        with self.assertRaises(ValueError):
            c.alias_weights(1.0)

    def test_overlap_lower_bound(self):
        self.assertEqual(c.overlap_seconds_lower_bound(600, 600), 1200)

    # 0%는 허용하지 않는다(경계 제외, 100%는 포함)
    def test_invalid_canary_percent(self):
        with self.assertRaises(ValueError):
            c.exposed_requests(100, 0)


if __name__ == "__main__":
    unittest.main()
