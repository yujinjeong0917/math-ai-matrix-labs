"""로컬 모델이 서비스가 되기 전에 비어 있는 항목을 찾는다. AWS 호출 없음, 표준 라이브러리만 쓴다.

숫자는 모두 가상(예시용) 입력이다: 하루 20건 또는 2만 건, 건당 30초.

재는 것: (1) 도착 간격과 평균 동시 작업 수, (2) 서비스 전 점검 여덟 물음 중 빈 것.
입력: notebook_state.json (scenarios: 하루 요청 수와 건당 초, notebook_state: 여덟 물음의 True/False/null).
출력: 시나리오별 한 줄(README 기대 출력의 앞 두 줄)과 노트북 상태 판정 한 줄(NOT_READY, 갖춤 0, 없음 6, 미확인 2).
평균만 다루므로 요청이 몰리는 순간의 최대 동시 작업 수는 구하지 못한다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SECONDS_PER_DAY = 24 * 60 * 60

# 서비스가 되려면 답해야 하는 여덟 가지 물음. (키, 물음)
SERVICE_QUESTIONS = [
    ("entry", "요청을 받는 주소(API)가 있는가"),
    ("auth", "누가 호출하는지 확인하는가"),
    ("input_limit", "입력 크기·형식 제한을 정했는가"),
    ("artifact", "모델 파일과 실행 환경의 버전이 기록돼 있는가"),
    ("permission", "실행 역할의 권한이 필요한 만큼만 있는가"),
    ("observe", "실패와 지연을 볼 로그가 있는가"),
    ("rollout", "새 버전을 올리고 되돌리는 순서가 있는가"),
    ("budget", "하루·월 사용량 상한이 있는가"),
]


def traffic(requests_per_day: int, seconds_per_request: float) -> dict[str, float]:
    """도착 간격과 평균 동시 작업 수(도착률 x 건당 시간)를 구한다."""
    # 0 이하 입력은 나눗셈과 도착률이 의미를 잃으므로 먼저 막는다
    if requests_per_day < 1 or seconds_per_request <= 0:
        raise ValueError("요청 수와 건당 시간은 0보다 커야 해요")
    rate_per_sec = requests_per_day / SECONDS_PER_DAY  # 도착률 (건/초)
    busy_seconds = requests_per_day * seconds_per_request
    return {
        # 하루 20건이면 86400/20 = 4320초, 2만 건이면 4.32초 (README 첫 값)
        "gap_seconds": SECONDS_PER_DAY / requests_per_day,
        "busy_seconds_per_day": busy_seconds,
        "busy_hours_per_day": busy_seconds / 3600,
        # 리틀의 법칙: 평균 동시 작업 수 = 도착률 x 건당 시간. 2만 건이면 약 6.94444
        "avg_concurrency": rate_per_sec * seconds_per_request,
        # 올림: 평균 6.94라도 작업자 6명이면 모자라 7명. 평균 기준이라 몰리는 순간은 반영 안 됨
        "min_workers_for_average": math.ceil(rate_per_sec * seconds_per_request),
    }


def missing_items(have: dict[str, bool | None]) -> dict[str, list[str]]:
    """True면 갖춤, False면 없음, None이나 키 없음이면 아직 확인 안 함(HOLD)."""
    # 키가 없는 경우도 get()이 None을 돌려줘 HOLD로 분류된다
    result = {"missing": [], "hold": [], "ok": []}
    for key, question in SERVICE_QUESTIONS:
        value = have.get(key)
        if value is True:
            result["ok"].append(key)
        elif value is False:
            result["missing"].append(key)
        else:
            result["hold"].append(key)
    return result


def verdict(have: dict[str, bool | None]) -> str:
    """없음이 하나라도 있으면 NOT_READY, 없음은 없고 미확인이 있으면 HOLD, 모두 갖추면 READY."""
    r = missing_items(have)
    if r["missing"]:
        return "NOT_READY"
    if r["hold"]:
        return "HOLD"
    return "READY"


def main(path: str) -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, scenario in data["scenarios"].items():
        t = traffic(scenario["requests_per_day"], scenario["seconds_per_request"])
        print(f"[{name}] 도착 간격 {t['gap_seconds']:.2f}초, 하루 처리 {t['busy_hours_per_day']:.4f}시간, "
              f"평균 동시 작업 {t['avg_concurrency']:.5f}, 최소 작업자 {t['min_workers_for_average']}")
    r = missing_items(data["notebook_state"])
    print("노트북 상태:", verdict(data["notebook_state"]), f"(갖춤 {len(r['ok'])}, 없음 {len(r['missing'])}, 미확인 {len(r['hold'])})")
    for key in r["missing"]:
        print("  없음:", dict(SERVICE_QUESTIONS)[key])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("notebook_state.json")))
