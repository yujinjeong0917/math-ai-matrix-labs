import json
import unittest
from pathlib import Path

from architecture_growth import (
    backlog_growth_per_day,
    components_for,
    evaluate,
    last_completion_seconds,
    requests_one_worker_finishes,
    stage_numbers,
    workers_for_burst,
)

SCENARIO = json.loads((Path(__file__).parent / "scenario.json").read_text(encoding="utf-8"))


class ArchitectureGrowthTest(unittest.TestCase):
    def test_monthly_numbers_match_chapter(self):
        first = stage_numbers(SCENARIO, 20)
        grown = stage_numbers(SCENARIO, 20000)
        self.assertEqual(first["requests_per_month"], 600)
        self.assertEqual(first["busy_seconds_per_day"], 600)
        self.assertAlmostEqual(first["busy_hours_per_month"], 5)
        self.assertEqual(grown["requests_per_month"], 600000)
        self.assertEqual(grown["busy_seconds_per_day"], 600000)
        self.assertAlmostEqual(grown["busy_hours_per_month"], 5000)
        self.assertEqual(first["workers_for_average"], 1)
        self.assertEqual(grown["workers_for_average"], 7)

    def test_one_worker_capacity_under_deadline(self):
        # (3600 - 90) // 30 = 117
        self.assertEqual(requests_one_worker_finishes(3600, 30, 90), 117)
        self.assertEqual(requests_one_worker_finishes(100, 30, 90), 0)

    def test_burst_workers(self):
        self.assertEqual(workers_for_burst(2, SCENARIO), 1)
        self.assertEqual(workers_for_burst(2000, SCENARIO), 18)
        self.assertGreaterEqual(17 * 117, 1989)
        self.assertLess(17 * 117, 2000)  # 17개로는 부족하다
        self.assertGreaterEqual(18 * 117, 2000)

    def test_single_worker_cannot_keep_up_after_growth(self):
        self.assertEqual(last_completion_seconds(2, 1, SCENARIO), 150)
        self.assertEqual(last_completion_seconds(2000, 1, SCENARIO), 60090)
        self.assertEqual(backlog_growth_per_day(20, 1, SCENARIO), 0)
        self.assertEqual(backlog_growth_per_day(20000, 1, SCENARIO), 513600)
        self.assertEqual(backlog_growth_per_day(20000, 7, SCENARIO), 0)
        self.assertEqual(backlog_growth_per_day(20000, 6, SCENARIO), 600000 - 6 * 86400)

    def test_last_completion_by_worker_count(self):
        r = evaluate(SCENARIO)
        self.assertEqual(r["grown"]["average_sized_last_completion_seconds"], 8670)
        self.assertEqual(r["grown"]["burst_sized_last_completion_seconds"], 3450)
        self.assertGreater(8670, SCENARIO["acceptable_wait_seconds"])
        self.assertLessEqual(3450, SCENARIO["acceptable_wait_seconds"])
        self.assertAlmostEqual(r["grown"]["single_worker_last_completion_hours"], 16.69, places=2)
        self.assertAlmostEqual(r["grown"]["single_worker_backlog_days_per_day"], 5.944, places=3)
        self.assertEqual(r["grown"]["always_on_hours_burst_sized"], 12960)
        self.assertEqual(r["first"]["always_on_hours_burst_sized"], 720)

    def test_first_stage_needs_only_the_always_components(self):
        r = evaluate(SCENARIO)["first"]
        self.assertEqual(
            set(r["components"]),
            {"artifact_storage", "least_privilege_role", "failure_alarm", "scale_from_zero_rule"},
        )

    def test_grown_stage_adds_queue_scaling_and_split(self):
        r = evaluate(SCENARIO)["grown"]
        for name in ("queue_in_front_of_workers", "backlog_scaling_policy", "duplicate_guard", "traffic_split_for_new_model"):
            self.assertIn(name, r["components"])
        self.assertAlmostEqual(r["utilization_if_burst_sized"], 600000 / (18 * 86400))
        self.assertEqual(round(r["utilization_if_burst_sized"], 3), 0.386)

    def test_components_depend_on_worker_counts(self):
        self.assertNotIn("traffic_split_for_new_model", components_for(1, 1))
        self.assertIn("traffic_split_for_new_model", components_for(1, 2))
        self.assertIn("queue_in_front_of_workers", components_for(2, 1))

    def test_shorter_wait_multiplies_workers(self):
        short = {**SCENARIO, "acceptable_wait_seconds": 300}
        self.assertEqual(requests_one_worker_finishes(300, 30, 90), 7)
        self.assertEqual(workers_for_burst(2000, short), 286)

    def test_sixty_seconds_per_request_is_not_exactly_double(self):
        slow = {**SCENARIO, "seconds_per_request": 60}
        r = evaluate(slow)["grown"]
        self.assertEqual(r["busy_seconds_per_day"], 1_200_000)
        self.assertEqual(r["workers_for_average"], 14)
        self.assertEqual(r["single_worker_last_completion_seconds"], 120090)
        self.assertNotEqual(r["single_worker_last_completion_seconds"], 2 * 60090)
        self.assertEqual(requests_one_worker_finishes(3600, 60, 90), 58)
        self.assertEqual(r["workers_for_burst"], 35)

    def test_shorter_wait_cost_in_hours(self):
        short = {**SCENARIO, "acceptable_wait_seconds": 300}
        r = evaluate(short)["grown"]
        self.assertEqual(r["workers_for_burst"], 286)
        self.assertEqual(r["always_on_hours_burst_sized"], 205920)

    def test_average_sized_always_on_utilization_and_spread_arrival_floor(self):
        grown = evaluate(SCENARIO)["grown"]
        self.assertAlmostEqual(grown["busy_hours_per_month"] / (7 * 24 * 30), 0.992, places=3)
        self.assertAlmostEqual(2000 * 30 / 3600, 16.667, places=3)

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            last_completion_seconds(0, 1, SCENARIO)
        with self.assertRaises(ValueError):
            workers_for_burst(10, {**SCENARIO, "acceptable_wait_seconds": 100})

    def test_no_price_in_scenario(self):
        text = (Path(__file__).parent / "scenario.json").read_text(encoding="utf-8")
        self.assertNotIn("$", text)


if __name__ == "__main__":
    unittest.main()
