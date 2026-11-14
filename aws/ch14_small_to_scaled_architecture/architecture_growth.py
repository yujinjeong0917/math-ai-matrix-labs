"""같은 문의 답변 모델을 하루 20건과 하루 2만 건에서 비교한다.

가격은 계산하지 않는다. 요청 수에서 작업 시간, 필요한 작업자 수, 몰릴 때의 대기 시간을 구하고,
그 숫자가 구성 요소(대기열, 자동 확장, 트래픽 나누기)를 요구하는지만 판단한다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
DAY_SECONDS = 24 * 60 * 60

# 규모와 상관없이 처음부터 필요한 구성 요소와 그 이유
ALWAYS = {
    "artifact_storage": "모델 파일과 버전을 보관할 곳(2장)",
    "least_privilege_role": "서비스가 쓸 수 있는 자원을 좁히는 권한(3장)",
    "failure_alarm": "실패와 지연을 알려 주는 경보(8장)",
    "scale_from_zero_rule": "0개까지 줄이는 구성에서는 요청이 하나만 와도 켜지도록 하는 규칙이 따로 필요(공식 문서: 이 규칙이 없으면 밀린 요청이 목표치를 넘을 때까지 기다림)",
}


def stage_numbers(scenario: dict[str, Any], requests_per_day: int) -> dict[str, float]:
    sec = scenario["seconds_per_request"]
    days = scenario["days_per_month"]
    busy_per_day = requests_per_day * sec
    return {
        "requests_per_month": requests_per_day * days,
        "busy_seconds_per_day": busy_per_day,
        "busy_hours_per_month": busy_per_day * days / 3600,
        "workers_for_average": math.ceil(busy_per_day / DAY_SECONDS),
        "average_load": busy_per_day / DAY_SECONDS,
    }


def requests_one_worker_finishes(deadline_seconds: float, seconds_per_request: float, startup_seconds: float) -> int:
    """꺼져 있던 작업자 1개가 마감 안에 끝낼 수 있는 요청 수."""
    usable = deadline_seconds - startup_seconds
    if usable < seconds_per_request:
        return 0
    return int(usable // seconds_per_request)


def workers_for_burst(burst_requests: int, scenario: dict[str, Any]) -> int:
    per_worker = requests_one_worker_finishes(
        scenario["acceptable_wait_seconds"], scenario["seconds_per_request"], scenario["startup_seconds"]
    )
    if per_worker == 0:
        raise ValueError("허용 대기 시간 안에 건 하나도 끝낼 수 없어요")
    return math.ceil(burst_requests / per_worker)


def last_completion_seconds(requests: int, workers: int, scenario: dict[str, Any]) -> float:
    """몰린 요청이 한꺼번에 도착했을 때 마지막 요청이 끝나는 시간. 작업자는 모두 꺼져 있다가 함께 시작한다."""
    if requests < 1 or workers < 1:
        raise ValueError("요청 수와 작업자 수는 1 이상이어야 해요")
    return scenario["startup_seconds"] + math.ceil(requests / workers) * scenario["seconds_per_request"]


def backlog_growth_per_day(requests_per_day: int, workers: int, scenario: dict[str, Any]) -> float:
    """작업자 수로 하루에 처리하지 못해 남는 작업 시간(초). 0이면 쌓이지 않는다."""
    work = requests_per_day * scenario["seconds_per_request"]
    return max(0.0, work - workers * DAY_SECONDS)


def components_for(burst_workers: int, average_workers: int) -> dict[str, str]:
    """숫자가 요구하는 구성 요소와 이유."""
    needed = dict(ALWAYS)
    if burst_workers > 1:
        needed["queue_in_front_of_workers"] = "몰린 요청을 작업자 수보다 많이 받아 두려면 줄이 필요. 비동기 엔드포인트는 줄을 내장하므로 새로 만들 것은 줄을 보는 쪽(11장)"
        needed["backlog_scaling_policy"] = "밀린 건수를 보고 작업자를 늘리고 줄이는 규칙(5장)"
        needed["duplicate_guard"] = "줄을 쓰면 같은 요청이 두 번 처리될 수 있어 중복 방지 키가 필요(11장)"
    if average_workers >= 2:
        needed["traffic_split_for_new_model"] = f"작업자가 {average_workers}개 이상이어야 새 모델에 일부 요청만 보낼 수 있음(9장)"
    return needed


def evaluate(scenario: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage, spec in scenario["stages"].items():
        rpd = spec["requests_per_day"]
        nums = stage_numbers(scenario, rpd)
        burst_requests = math.ceil(rpd * scenario["burst_share"])
        burst_workers = workers_for_burst(burst_requests, scenario)
        out[stage] = {
            **nums,
            "burst_requests": burst_requests,
            "workers_for_burst": burst_workers,
            "single_worker_last_completion_seconds": last_completion_seconds(burst_requests, 1, scenario),
            "single_worker_last_completion_hours": last_completion_seconds(burst_requests, 1, scenario) / 3600,
            "single_worker_backlog_seconds_per_day": backlog_growth_per_day(rpd, 1, scenario),
            "single_worker_backlog_days_per_day": backlog_growth_per_day(rpd, 1, scenario) / DAY_SECONDS,
            "average_sized_last_completion_seconds": last_completion_seconds(burst_requests, nums["workers_for_average"], scenario),
            "burst_sized_last_completion_seconds": last_completion_seconds(burst_requests, burst_workers, scenario),
            "always_on_hours_burst_sized": burst_workers * scenario["days_per_month"] * 24,
            "utilization_if_burst_sized": nums["busy_seconds_per_day"] / (burst_workers * DAY_SECONDS),
            "components": components_for(burst_workers, nums["workers_for_average"]),
        }
    return out


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "scenario.json")
    scenario = json.loads(path.read_text(encoding="utf-8"))
    for stage, r in evaluate(scenario).items():
        print(f"[{stage}] 월 요청 {r['requests_per_month']:,.0f}건, 월 작업 {r['busy_hours_per_month']:,.1f}시간")
        print(f"  평균 작업자 {r['workers_for_average']}개(부하 {r['average_load']:.4f}), 몰릴 때 작업자 {r['workers_for_burst']}개")
        print(f"  작업자 1개일 때 몰린 {r['burst_requests']:,}건의 마지막 완료 {r['single_worker_last_completion_seconds']:,.0f}초, 하루 {r['single_worker_backlog_seconds_per_day']:,.0f}초씩 밀림")
        print(f"  몰린 요청의 마지막 완료: 1개 {r['single_worker_last_completion_seconds']:,.0f}초({r['single_worker_last_completion_hours']:.1f}시간), 평균 기준 {r['workers_for_average']}개 {r['average_sized_last_completion_seconds']:,.0f}초, 몰릴 때 기준 {r['workers_for_burst']}개 {r['burst_sized_last_completion_seconds']:,.0f}초 (허용 {scenario['acceptable_wait_seconds']:,}초)")
        print(f"  하루 밀리는 작업 {r['single_worker_backlog_days_per_day']:.1f}일치, 몰릴 때 기준으로 항상 켜 두면 {r['always_on_hours_burst_sized']:,}시간/월, 가동률 {r['utilization_if_burst_sized']:.1%}")
        print("  구성 요소:", ", ".join(r["components"]))


if __name__ == "__main__":
    main()
