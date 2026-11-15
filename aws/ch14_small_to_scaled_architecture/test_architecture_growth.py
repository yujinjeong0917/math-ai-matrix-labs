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
    # README 표의 월 요청 600/600,000건, 월 작업 5/5,000시간, 평균 기준 작업자 1/7개
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

    # 17개는 17 x 117 = 1989건으로 2000건에 모자라고 18개가 되어야 한다(경계 확인)
    def test_burst_workers(self):
        self.assertEqual(workers_for_burst(2, SCENARIO), 1)
        self.assertEqual(workers_for_burst(2000, SCENARIO), 18)
        self.assertGreaterEqual(17 * 117, 1989)
        self.assertLess(17 * 117, 2000)  # 17개로는 부족하다
        self.assertGreaterEqual(18 * 117, 2000)

    # 20건: 90 + 2 x 30 = 150초. 2만 건: 1개로는 하루 513,600초씩 밀리고 7개면 쌓이지 않는다(600,000 < 7 x 86,400 = 604,800)
    def test_single_worker_cannot_keep_up_after_growth(self):
        self.assertEqual(last_completion_seconds(2, 1, SCENARIO), 150)
        self.assertEqual(last_completion_seconds(2000, 1, SCENARIO), 60090)
        self.assertEqual(backlog_growth_per_day(20, 1, SCENARIO), 0)
        self.assertEqual(backlog_growth_per_day(20000, 1, SCENARIO), 513600)
        self.assertEqual(backlog_growth_per_day(20000, 7, SCENARIO), 0)
        self.assertEqual(backlog_growth_per_day(20000, 6, SCENARIO), 600000 - 6 * 86400)

    # 평균 기준 7개는 8,670초로 마감(3,600초) 초과, 몰릴 때 기준 18개는 3,450초로 이내. 60,090초 = 16.69시간
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

    # 20건 단계는 항상 필요한 구성 요소 4개뿐이다(README)
    def test_first_stage_needs_only_the_always_components(self):
        r = evaluate(SCENARIO)["first"]
        self.assertEqual(
            set(r["components"]),
            {"artifact_storage", "least_privilege_role", "failure_alarm", "scale_from_zero_rule"},
        )

    # 2만 건 단계는 대기열, 확장 규칙, 중복 방지, 트래픽 나누기가 더해진다. 가동률 600,000 / (18 x 86,400) = 0.386
    def test_grown_stage_adds_queue_scaling_and_split(self):
        r = evaluate(SCENARIO)["grown"]
        for name in ("queue_in_front_of_workers", "backlog_scaling_policy", "duplicate_guard", "traffic_split_for_new_model"):
            self.assertIn(name, r["components"])
        self.assertAlmostEqual(r["utilization_if_burst_sized"], 600000 / (18 * 86400))
        self.assertEqual(round(r["utilization_if_burst_sized"], 3), 0.386)

    # 인자 순서는 (몰릴 때 작업자, 평균 작업자). 트래픽 나누기는 평균 작업자 2개 이상, 대기열은 몰릴 때 작업자 2개 이상이 기준
    def test_components_depend_on_worker_counts(self):
        self.assertNotIn("traffic_split_for_new_model", components_for(1, 1))
        self.assertIn("traffic_split_for_new_model", components_for(1, 2))
        self.assertIn("queue_in_front_of_workers", components_for(2, 1))

    # 마감 300초면 (300 - 90) // 30 = 7건씩만 끝내 2000/7 = 285.7 -> 286개
    def test_shorter_wait_multiplies_workers(self):
        short = {**SCENARIO, "acceptable_wait_seconds": 300}
        self.assertEqual(requests_one_worker_finishes(300, 30, 90), 7)
        self.assertEqual(workers_for_burst(2000, short), 286)

    # 건당 60초: 일은 2배(1,200,000초)지만 시작 90초는 그대로라 120,090초(2 x 60,090 = 120,180이 아님). 작업자는 35개(36개 아님)
    def test_sixty_seconds_per_request_is_not_exactly_double(self):
        slow = {**SCENARIO, "seconds_per_request": 60}
        r = evaluate(slow)["grown"]
        self.assertEqual(r["busy_seconds_per_day"], 1_200_000)
        self.assertEqual(r["workers_for_average"], 14)
        self.assertEqual(r["single_worker_last_completion_seconds"], 120090)
        self.assertNotEqual(r["single_worker_last_completion_seconds"], 2 * 60090)
        self.assertEqual(requests_one_worker_finishes(3600, 60, 90), 58)
        self.assertEqual(r["workers_for_burst"], 35)

    # 286개 x 30일 x 24시간 = 205,920시간/월
    def test_shorter_wait_cost_in_hours(self):
        short = {**SCENARIO, "acceptable_wait_seconds": 300}
        r = evaluate(short)["grown"]
        self.assertEqual(r["workers_for_burst"], 286)
        self.assertEqual(r["always_on_hours_burst_sized"], 205920)

    # 평균 7개만 켜 두면 가동률 5,000 / 5,040 = 0.992. 2000건 x 30초 = 16.667시간은 몰림 일의 양(작업자 수와 무관)
    def test_average_sized_always_on_utilization_and_spread_arrival_floor(self):
        grown = evaluate(SCENARIO)["grown"]
        self.assertAlmostEqual(grown["busy_hours_per_month"] / (7 * 24 * 30), 0.992, places=3)
        self.assertAlmostEqual(2000 * 30 / 3600, 16.667, places=3)

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            last_completion_seconds(0, 1, SCENARIO)
        with self.assertRaises(ValueError):
            workers_for_burst(10, {**SCENARIO, "acceptable_wait_seconds": 100})

    # 가격을 시나리오에 넣지 않았는지 달러 기호만 단순 점검(완전한 검사는 아님)
    def test_no_price_in_scenario(self):
        text = (Path(__file__).parent / "scenario.json").read_text(encoding="utf-8")
        self.assertNotIn("$", text)


if __name__ == "__main__":
    unittest.main()
