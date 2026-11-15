"""같은 문의 답변 모델을 하루 20건과 하루 2만 건에서 비교한다.

가격은 계산하지 않는다. 요청 수에서 작업 시간, 필요한 작업자 수, 몰릴 때의 대기 시간을 구하고,
그 숫자가 구성 요소(대기열, 자동 확장, 트래픽 나누기)를 요구하는지만 판단한다.

재는 것: 단계(first=하루 20건, grown=하루 2만 건)별 월 작업 시간, 평균 기준 작업자 수, 몰릴 때 마감 안에
끝내는 데 필요한 작업자 수, 몰림의 마지막 완료 시각, 항상 켜 둘 때 가동률과 필요한 구성 요소.
입력: scenario.json(가상 값: 건당 30초, 시작 90초, 허용 대기 3,600초, 몰림 10%). 출력: README 기대 출력 표
(작업자 1개와 7개와 18개, 마지막 완료 150초 대 60,090초 대 8,670초 대 3,450초, 가동률 0.7%와 38.6% 등).
몰림은 요청이 한꺼번에 도착하고 작업자가 모두 꺼져 있다가 함께 시작한다고 가정한다(단순화).
작업자 1개는 한 번에 한 건만 처리한다.
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
    """한 단계의 월 요청 수, 작업 시간, 평균 기준 작업자 수(단위: 건, 초, 시간, 개)."""
    sec = scenario["seconds_per_request"]
    days = scenario["days_per_month"]
    busy_per_day = requests_per_day * sec
    return {
        "requests_per_month": requests_per_day * days,
        "busy_seconds_per_day": busy_per_day,
        "busy_hours_per_month": busy_per_day * days / 3600,
        # 평균 부하(하루 작업 초 / 86,400)를 올림. 2만 건이면 6.94 -> 7개, 20건이면 0.007 -> 1개
        "workers_for_average": math.ceil(busy_per_day / DAY_SECONDS),
        "average_load": busy_per_day / DAY_SECONDS,
    }


def requests_one_worker_finishes(deadline_seconds: float, seconds_per_request: float, startup_seconds: float) -> int:
    """꺼져 있던 작업자 1개가 마감 안에 끝낼 수 있는 요청 수."""
    # 시작 90초는 켜질 때 한 번만 쓰고 남은 시간에 한 건씩 처리한다: (3600 - 90) // 30 = 117건
    usable = deadline_seconds - startup_seconds
    if usable < seconds_per_request:
        return 0
    # 함정: 나눗셈을 올림하지 않고 내림. 마감 안에 "끝난" 건만 센다
    return int(usable // seconds_per_request)


def workers_for_burst(burst_requests: int, scenario: dict[str, Any]) -> int:
    """몰린 요청을 허용 대기 안에 모두 끝내는 데 필요한 작업자 수. 2000건 / 117건 = 17.1 -> 18개(17개는 1989건이라 부족)."""
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
    # 시작 90초 + (한 작업자가 맡는 건수) x 30초. 일이 두 배가 돼도 시작 시간은 그대로라 정확히 두 배가 아님(테스트 참고)
    # 2000건, 1개: 90 + 2000 x 30 = 60,090초. 7개: 90 + 286 x 30 = 8,670초. 18개: 90 + 112 x 30 = 3,450초
    return scenario["startup_seconds"] + math.ceil(requests / workers) * scenario["seconds_per_request"]


def backlog_growth_per_day(requests_per_day: int, workers: int, scenario: dict[str, Any]) -> float:
    """작업자 수로 하루에 처리하지 못해 남는 작업 시간(초). 0이면 쌓이지 않는다."""
    # 하루에 들어오는 작업 초에서 작업자들이 하루에 처리할 수 있는 초(작업자 수 x 86,400)를 뺀다. 음수는 0(쌓이지 않음)
    # 2만 건, 1개: 600,000 - 86,400 = 513,600초(약 5.9일치)
    work = requests_per_day * scenario["seconds_per_request"]
    return max(0.0, work - workers * DAY_SECONDS)


def components_for(burst_workers: int, average_workers: int) -> dict[str, str]:
    """숫자가 요구하는 구성 요소와 이유."""
    needed = dict(ALWAYS)
    # 몰릴 때 작업자가 둘 이상이어야 하면 요청을 쌓아 둘 줄이 필요하다
    if burst_workers > 1:
        needed["queue_in_front_of_workers"] = "몰린 요청을 작업자 수보다 많이 받아 두려면 줄이 필요. 비동기 엔드포인트는 줄을 내장하므로 새로 만들 것은 줄을 보는 쪽(11장)"
        needed["backlog_scaling_policy"] = "밀린 건수를 보고 작업자를 늘리고 줄이는 규칙(5장)"
        needed["duplicate_guard"] = "줄을 쓰면 같은 요청이 두 번 처리될 수 있어 중복 방지 키가 필요(11장)"
    # 평균 작업자가 1개면 나눌 묶음이 없어 트래픽 나누기가 의미가 없다. 기준은 평균 작업자(몰릴 때 기준 아님)
    if average_workers >= 2:
        needed["traffic_split_for_new_model"] = f"작업자가 {average_workers}개 이상이어야 새 모델에 일부 요청만 보낼 수 있음(9장)"
    return needed


def evaluate(scenario: dict[str, Any]) -> dict[str, Any]:
    """시나리오의 모든 단계를 계산해 {단계: 지표 사전}으로 돌려준다."""
    out: dict[str, Any] = {}
    for stage, spec in scenario["stages"].items():
        rpd = spec["requests_per_day"]
        nums = stage_numbers(scenario, rpd)
        # 몰린 요청 수: 하루 요청 x 몰림 비율(0.1). 20건이면 2건, 2만 건이면 2000건
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
            # 몰릴 때 기준 18개를 항상 켜 두면 18 x 30 x 24 = 12,960시간/월이고, 가동률은 600,000 / (18 x 86,400) = 38.6%
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
