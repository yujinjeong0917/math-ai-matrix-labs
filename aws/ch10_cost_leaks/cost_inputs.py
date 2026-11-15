"""비용이 새는 곳을 단가 없이 수량으로 먼저 센다.

수량(시간·GB)은 이 코드가 계산한다. 단가는 독자가 AWS Pricing Calculator에서
확인한 값을 unit_prices에 넣어야만 금액이 나온다. 비어 있으면 그 줄은 HOLD다.
1 GB는 10^9 바이트로 센다.

재는 것: 유휴 시간 비율, 로그 저장량(보관 기간 유무), NAT 게이트웨이(사설망의 외부 접속 통로) 내려받기량,
이전 버전 저장량, 인터넷으로 나가는 응답량. 모두 수량(시간·GB)이며 금액은 단가가 있을 때만.
입력: cost_profile.json(가상 값, scenarios에 small=하루 20건, scaled=하루 2만 건). 출력: README 기대 출력
(작업자 1개/14개, 유휴 0.993/0.504, 로그 12GB 대 144GB, NAT 280.98GB, 모델 46.83GB -> 14.049GB 등).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

GB = 1_000_000_000
PRICE_KEYS = (
    "compute_hours",
    "nat_gb",
    "log_ingest_gb",
    "log_stored_gb",
    "model_storage_gb",
    "image_storage_gb",
    "data_out_gb",
)


def busy_seconds_per_day(requests_per_day: float, seconds_per_request: float) -> float:
    return requests_per_day * seconds_per_request


def workers_needed(requests_per_day: float, seconds_per_request: float, target_utilization: float) -> int:
    """하루 계산 시간을 목표 사용률로 나눠 필요한 작업자 수를 올림한다(최소 1)."""
    if not 0 < target_utilization <= 1:
        raise ValueError("목표 사용률은 0 초과 1 이하여야 해요")
    # 하루 계산 시간 / (하루 초 x 목표 사용률). 2만 건 x 30초 / (86400 x 0.5) = 13.9 -> 올림 14
    need = busy_seconds_per_day(requests_per_day, seconds_per_request) / (86400 * target_utilization)
    return max(1, math.ceil(need))


def compute_hours(workers: int, days: int) -> dict[str, float]:
    # 상시 가동이라 요청이 없어도 작업자 수 x 일수 x 24시간을 켜 둔다
    return {"always_on_hours": workers * days * 24}


def idle_share(requests_per_day: float, seconds_per_request: float, workers: int, days: int) -> dict[str, float]:
    """켜 둔 시간 중 실제 계산한 시간과 놀고 있는 비율. 하루 20건이면 5/720만 일해 idle 0.993."""
    on = workers * days * 24
    busy = requests_per_day * days * seconds_per_request / 3600
    return {"always_on_hours": on, "busy_hours": busy, "idle_hours": on - busy, "idle_share": (on - busy) / on}


def log_gb_per_day(requests_per_day: float, bytes_per_request: float) -> float:
    # 바이트 -> GB는 10^9로 나눈다(1024^3 아님)
    return requests_per_day * bytes_per_request / GB


def stored_log_gb(gb_per_day: float, retention_days: float) -> float:
    """보관 기간이 정해져 있으면 일정 기간 뒤 저장량은 하루 양 × 보관 일수에서 멈춘다."""
    # 하루 0.4GB x 30일 = 12GB에서 멈춘다. 만료가 없으면 계속 쌓임(아래 함수)
    return gb_per_day * retention_days


def stored_without_expiry_gb(gb_per_day: float, days_per_month: int, months: int) -> float:
    """만료 없이 months개월 쌓은 저장량. 0.4 x 30 x 12 = 144GB (보관 30일의 12배)."""
    return gb_per_day * days_per_month * months


def nat_download_gb(model_artifact_gb: float, cold_starts_per_month: int) -> float:
    """시작할 때마다 모델 파일을 NAT 게이트웨이로 내려받는다고 가정한 처리량."""
    # 시작마다 전체를 내려받는다는 가정(현실과 다를 수 있음, README의 확인 못 하는 것 참고). 4.683GB x 60번 = 280.98GB
    return model_artifact_gb * cold_starts_per_month


def versions_storage_gb(size_gb: float, versions: int) -> float:
    """버전마다 파일 전체가 따로 저장된다고 보고 크기 x 버전 수(차이분만 저장하지 않음)."""
    return size_gb * versions


def quantities(profile: dict[str, Any], scenario: str) -> dict[str, float]:
    """시나리오 하나의 모든 수량(시간, GB)을 계산한다."""
    p = profile
    rpd = p["scenarios"][scenario]["requests_per_day"]
    days = p["days_per_month"]
    workers = workers_needed(rpd, p["seconds_per_request"], p["target_utilization"])
    prompt_gb_day = log_gb_per_day(rpd, p["prompt_log_bytes_per_request"])
    meta_gb_day = log_gb_per_day(rpd, p["metadata_log_bytes_per_request"])
    return {
        "workers": workers,
        "compute_hours": compute_hours(workers, days)["always_on_hours"],
        "idle_share": idle_share(rpd, p["seconds_per_request"], workers, days)["idle_share"],
        "prompt_log_gb_per_day": prompt_gb_day,
        "metadata_log_gb_per_day": meta_gb_day,
        # 수집량은 한 달 합, 저장량은 보관 기간 합이라 서로 다른 값이다
        "log_ingest_gb": prompt_gb_day * days,
        "log_stored_gb": stored_log_gb(prompt_gb_day, p["log_retention_days"]),
        "log_stored_gb_no_expiry": stored_without_expiry_gb(prompt_gb_day, days, p["months_without_expiry"]),
        "nat_gb": nat_download_gb(p["model_artifact_gb"], p["cold_starts_per_month"]),
        "model_storage_gb": versions_storage_gb(p["model_artifact_gb"], p["model_versions_kept"]),
        # 수명주기(오래된 버전 삭제) 규칙을 적용해 남기는 버전 수만 다르다
        "model_storage_gb_lifecycle": versions_storage_gb(p["model_artifact_gb"], p["model_versions_with_lifecycle"]),
        "image_storage_gb": versions_storage_gb(p["image_gb"], p["image_versions_kept"]),
        "image_storage_gb_lifecycle": versions_storage_gb(p["image_gb"], p["image_versions_with_lifecycle"]),
        "data_out_gb": rpd * days * p["response_bytes"] / GB,
    }


def cost_lines(q: dict[str, float], unit_prices: dict[str, float | None]) -> dict[str, float | str]:
    """수량 × 단가. 단가가 null이면 HOLD."""
    out: dict[str, float | str] = {}
    for key in PRICE_KEYS:
        price = unit_prices.get(key)
        # 단가가 없으면 0원이 아니라 HOLD. 0으로 두면 "공짜"로 오해하기 때문
        out[key] = "HOLD" if price is None else q[key] * price
    return out


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("cost_profile.json")
    profile = json.loads(path.read_text(encoding="utf-8"))
    for name in profile["scenarios"]:
        q = quantities(profile, name)
        print(f"[{name}] 하루 {profile['scenarios'][name]['requests_per_day']}건")
        print(f"  작업자 {q['workers']}개, 켜 둔 시간 {q['compute_hours']:.0f}시간/월, 놀고 있는 비율 {q['idle_share']:.3f}")
        print(f"  프롬프트 전체 로그 {q['prompt_log_gb_per_day']:.4f}GB/일, 메타데이터만 {q['metadata_log_gb_per_day']:.5f}GB/일")
        print(f"  로그 저장량: 보관 {profile['log_retention_days']}일이면 {q['log_stored_gb']:.3f}GB, 만료 없이 {profile['months_without_expiry']}개월이면 {q['log_stored_gb_no_expiry']:.3f}GB")
        print(f"  NAT로 내려받는 모델 {q['nat_gb']:.2f}GB/월, 인터넷으로 나가는 응답 {q['data_out_gb']:.3f}GB/월")
        print(f"  모델 버전 저장 {q['model_storage_gb']:.3f}GB -> 수명주기 후 {q['model_storage_gb_lifecycle']:.3f}GB")
        print(f"  컨테이너 이미지 저장 {q['image_storage_gb']:.0f}GB -> 수명주기 후 {q['image_storage_gb_lifecycle']:.0f}GB")
        lines = cost_lines(q, profile["unit_prices"])
        print("  금액 줄:", ", ".join(f"{k}={v}" for k, v in lines.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
