"""새 모델을 천천히 바꿀 때 필요한 표본 수·경보 기준·겹쳐 켜 두는 시간을 계산한다.

AWS 계정이 필요 없다. 가격은 다루지 않고 시간·건수만 계산한다.

재는 것: (1) 카나리(새 모델에 일부 요청만 보내는 방식)에 노출되는 건수, (2) 나쁜 답을 95% 확률로
보려면 필요한 노출 건수, (3) 경보 기준(창 안의 나쁜 답 개수)의 오경보 확률과 감지 확률,
(4) 두 묶음이 함께 켜지는 시간.
입력: canary_profile.json(가상 비율). 출력: README 기대 출력 15줄(노출 1건/일과 1000건/일, 149건과 29건,
오경보 0.018 대 감지 0.750 등).
가정: 나쁜 답은 요청마다 독립이고 같은 비율로 난다(모형의 단순화). 질문 유형별로 다르면 필요 건수가 늘어난다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def exposed_requests(total_requests: float, canary_percent: float) -> float:
    """전체 요청 중 새 모델이 받는 요청 수."""
    if not 0 < canary_percent <= 100:
        raise ValueError("canary_percent는 0 초과 100 이하여야 해요")
    # 20건/일 x 5% = 1건, 20,000건/일 x 5% = 1,000건
    return total_requests * canary_percent / 100


def detect_probability(exposed: int, bad_rate: float) -> float:
    """새 모델이 나쁜 답을 낸다고 할 때, 노출된 요청 중 하나 이상에서 보일 확률."""
    if exposed < 0 or not 0 < bad_rate < 1:
        raise ValueError("노출 건수와 나쁜 답 비율을 확인하세요")
    # 여집합: 노출 n건이 전부 정상일 확률 (1-p)^n을 1에서 뺀다. 50건, 2%면 0.636
    return 1 - (1 - bad_rate) ** exposed


def needed_exposed(bad_rate: float, confidence: float = 0.95) -> int:
    """나쁜 답을 하나 이상 볼 확률이 confidence 이상이 되는 최소 노출 건수."""
    if not 0 < bad_rate < 1 or not 0 < confidence < 1:
        raise ValueError("비율은 0과 1 사이여야 해요")
    # 1 - (1-p)^n >= c 를 n에 대해 풀면 n >= ln(1-c) / ln(1-p). 올림하므로 2%는 149건(148건은 95% 미만), 10%는 29건
    return math.ceil(math.log(1 - confidence) / math.log(1 - bad_rate))


def days_to_collect(requests_per_day: float, canary_percent: float, needed: int) -> float:
    """필요한 노출 건수를 모으는 데 걸리는 일수. 하루 노출이 1건이면 149건에 149일."""
    return needed / exposed_requests(requests_per_day, canary_percent)


def binomial_tail(n: int, p: float, k: int) -> float:
    """n건 중 나쁜 답이 k건 이상일 확률(정확한 이항분포 계산)."""
    if k <= 0:
        return 1.0  # "0건 이상"은 항상 참
    # 함정: k가 n을 넘으면 range가 비어 합이 0.0이 된다(테스트 참고). 각 항은 이항분포의 확률질량
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def alarm_rates(profile: dict[str, Any]) -> dict[str, float]:
    """창 하나(window_requests건)에서 나쁜 답이 alarm_error_count건 이상이면 경보가 울린다고 할 때의 확률."""
    # 같은 창(50건)에 같은 기준(4건 이상)을 적용하되 나쁜 답 비율만 바꿔, 정상 모델(오경보)과 새 모델(감지)을 비교
    n = profile["window_requests"]
    k = profile["alarm_error_count"]
    return {
        "false_alarm": binomial_tail(n, profile["baseline_bad_rate"], k),
        "detect_bad_model": binomial_tail(n, profile["new_model_bad_rate"], k),
    }


def two_window_rates(single: dict[str, float]) -> dict[str, float]:
    """연속 두 창을 판정하는 두 방식. both: 두 창 모두 울려야 경보, either: 둘 중 하나만 울려도 경보."""
    fa, det = single["false_alarm"], single["detect_bad_model"]
    # 두 창이 독립이라고 가정. both는 곱(둘 다 울림), either는 1 - 둘 다 안 울림
    # both: 오경보 0.0003으로 크게 줄지만 감지도 0.750 -> 0.562로 떨어진다. either는 반대로 둘 다 올라간다
    return {
        "both_false_alarm": fa * fa,
        "both_detect": det * det,
        "either_false_alarm": 1 - (1 - fa) ** 2,
        "either_detect": 1 - (1 - det) ** 2,
    }


def alias_weights(extra_weight: float) -> dict[str, float]:
    """Lambda 별칭 가중치 예시: 추가 버전이 extra_weight를 받고 나머지는 첫 버전이 받는다."""
    if not 0 < extra_weight < 1:
        raise ValueError("추가 버전 가중치는 0과 1 사이여야 해요")
    # 부동소수점 잔차(0.9500000000000001)를 막으려 10자리 반올림
    return {"first_version": round(1 - extra_weight, 10), "additional_version": extra_weight}


def overlap_seconds_lower_bound(wait_seconds: int, termination_wait_seconds: int) -> int:
    """카나리 대기와 종료 대기 동안 두 묶음이 함께 켜져 있는 시간의 하한(새 묶음을 준비하는 시간은 뺀 값)."""
    # 하한인 이유: 새 묶음을 띄우는 시간은 포함하지 않았다. 600 + 600 = 1200초
    return wait_seconds + termination_wait_seconds


def report(profile: dict[str, Any]) -> dict[str, Any]:
    """프로필로 모든 계산을 모아 사전으로 돌려준다. 키 이름의 small은 하루 20건, scaled는 하루 20000건 규모."""
    p = profile
    need_base = needed_exposed(p["baseline_bad_rate"], p["confidence"])
    need_bad = needed_exposed(p["new_model_bad_rate"], p["confidence"])
    return {
        "exposed_per_day_small": exposed_requests(p["requests_per_day"], p["canary_percent"]),
        "exposed_per_day_scaled": exposed_requests(p["scaled_requests_per_day"], p["canary_percent"]),
        "needed_exposed_at_baseline_rate": need_base,
        "needed_exposed_at_new_rate": need_bad,
        "days_small": days_to_collect(p["requests_per_day"], p["canary_percent"], need_base),
        # 하루 단위 일수를 시간(x24), 분(x1440)으로 환산: 149건 / 1000건/일 x 24 = 3.576시간
        "hours_scaled": 24 * days_to_collect(p["scaled_requests_per_day"], p["canary_percent"], need_base),
        "days_small_new": days_to_collect(p["requests_per_day"], p["canary_percent"], need_bad),
        "minutes_scaled_new": 24 * 60 * days_to_collect(p["scaled_requests_per_day"], p["canary_percent"], need_bad),
        "window_fill_days_small": p["window_requests"] / exposed_requests(p["requests_per_day"], p["canary_percent"]),
        "window_fill_minutes_scaled": 24 * 60 * p["window_requests"] / exposed_requests(p["scaled_requests_per_day"], p["canary_percent"]),
        "two_windows": two_window_rates(alarm_rates(p)),
        "detect_in_50": detect_probability(50, p["baseline_bad_rate"]),
        "alarm": alarm_rates(p),
        "alias_weights": alias_weights(p["alias_extra_weight"]),
        "overlap_seconds": overlap_seconds_lower_bound(p["bake_wait_seconds"], p["termination_wait_seconds"]),
    }


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("canary_profile.json")
    r = report(json.loads(path.read_text(encoding="utf-8")))
    print(f"하루 요청 20건, 카나리 5%: 새 모델 노출 {r['exposed_per_day_small']:.1f}건/일")
    print(f"하루 요청 20000건, 카나리 5%: 새 모델 노출 {r['exposed_per_day_scaled']:.0f}건/일")
    print(f"나쁜 답 2%를 95% 확률로 보려면 노출 {r['needed_exposed_at_baseline_rate']}건 필요")
    print(f"나쁜 답 10%를 95% 확률로 보려면 노출 {r['needed_exposed_at_new_rate']}건 필요")
    print(f"하루 20건에서 {r['needed_exposed_at_baseline_rate']}건 모으는 데 {r['days_small']:.1f}일")
    print(f"하루 20000건에서 {r['needed_exposed_at_baseline_rate']}건 모으는 데 {r['hours_scaled']:.1f}시간")
    print(f"50건 노출 때 나쁜 답 2%를 하나 이상 볼 확률 {r['detect_in_50']:.3f}")
    print(f"창 50건, 4건 이상이면 경보: 정상 모델 오경보 {r['alarm']['false_alarm']:.3f}, 나쁜 모델 감지 {r['alarm']['detect_bad_model']:.3f}")
    print(f"하루 20건에서 {r['needed_exposed_at_new_rate']}건 모으는 데 {r['days_small_new']:.1f}일")
    print(f"하루 20000건에서 {r['needed_exposed_at_new_rate']}건 모으는 데 {r['minutes_scaled_new']:.1f}분")
    print(f"경보 창 50건이 차는 데: 하루 20건 {r['window_fill_days_small']:.1f}일, 하루 20000건 {r['window_fill_minutes_scaled']:.1f}분")
    tw = r["two_windows"]
    print(f"연속 두 창 모두 울려야 경보: 오경보 {tw['both_false_alarm']:.4f}, 감지 {tw['both_detect']:.3f}")
    print(f"연속 두 창 중 하나만 울려도 경보: 오경보 {tw['either_false_alarm']:.3f}, 감지 {tw['either_detect']:.3f}")
    print(f"별칭 가중치: {r['alias_weights']}")
    print(f"두 묶음이 함께 켜진 시간 하한: {r['overlap_seconds']}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
