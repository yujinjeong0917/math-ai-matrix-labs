"""0장 실습: 지연 구간의 백분위를 직접 계산하고 web.dev 기준값과 대조한다.

모든 숫자는 가상, 예시용이다. 체감(느낌)을 계산하지 않는다. 실제로 잰 시간만 다룬다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math

# web.dev "Web Vitals" 문서(https://web.dev/articles/vitals, 2026-11-18 확인, 문서 갱신일 2024-10-31)가 적은 "좋음" 기준.
# 같은 문서는 페이지 방문의 75번째 백분위로 판단하라고 쓴다.
LCP_GOOD_SECONDS = 2.5
INP_GOOD_MS = 200
CLS_GOOD = 0.1


def percentile_linear(values: list[float], p: float) -> float:
    """선형 보간 백분위(p는 0~100). 정렬한 값 사이를 직선으로 잇는다."""
    if not values:
        raise ValueError("값이 없다")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def percentile_nearest_rank(values: list[float], p: float) -> float:
    """가장 가까운 순위 백분위: 올림(n*p/100)번째 값."""
    xs = sorted(values)
    rank = max(1, math.ceil(len(xs) * p / 100))
    return xs[rank - 1]


def is_good(value: float, limit: float) -> bool:
    return value <= limit


def share_over(values: list[float], limit: float) -> float:
    return sum(1 for v in values if v > limit) / len(values)


def pooled_sd(sd1: float, n1: int, sd2: float, n2: int) -> float:
    """두 집단의 합동 표준편차."""
    return math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def cohens_d(mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int) -> float:
    """Cohen의 d: 평균 차이를 합동 표준편차로 나눈 값(2번째 집단 평균 - 1번째 집단 평균)."""
    return (mean2 - mean1) / pooled_sd(sd1, n1, sd2, n2)


def percent_1dp(count: int, total: int) -> float:
    """소수 첫째 자리까지 반올림한 백분율(0.5는 올림). 66.25%를 66.3%로 쓰는 논문 표기와 맞춘다."""
    return math.floor(count * 1000 / total + 0.5) / 10


def frames_in(duration_ms: float, fps: int = 60) -> float:
    return duration_ms / (1000 / fps)


# 가상 데이터: 같은 페이지의 방문 10번에서 잰 LCP(초). 예시용.
SAMPLE_A = [1.8, 2.1, 1.9, 2.4, 3.6, 2.2, 1.7, 2.0, 4.8, 2.3]
SAMPLE_B = [2.0, 2.2, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 2.4, 2.3]
# 표본 C: 평균은 기준 안인데 75번째 백분위는 기준 밖이 되는, 느린 쪽이 4번 몰린 가상 표본.
SAMPLE_C = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.7, 2.9, 3.1, 3.5]


# Tan 외(2026) 카드가 적은 값(사려 깊음 평균과 SD)과 칸별 인원을 과제 유형끼리 합친 집단 크기.
# Creation 41/42/41 + Advice 38/38/40 = 79/80/81명. 카드가 맞춰 본 합친 인원이다.
TAN_THOUGHTFUL = {2: (5.76, 1.18, 79), 9: (6.09, 0.90, 80), 20: (6.11, 0.82, 81)}
TAN_FELT_DELAY = {2: (21, 79), 9: (53, 80), 20: (66, 81)}  # (지연을 느꼈다고 답한 사람, 집단 인원)


def main() -> None:
    for name, xs in (("A", SAMPLE_A), ("B", SAMPLE_B), ("C", SAMPLE_C)):
        mean = sum(xs) / len(xs)
        print(f"[{name}] 평균 {mean:.2f}초({'기준 안' if is_good(mean, LCP_GOOD_SECONDS) else '기준 밖'}), 가운데값 {percentile_linear(xs, 50):.3f}초, "
              f"p75(선형) {percentile_linear(xs, 75):.3f}초, p75(순위) {percentile_nearest_rank(xs, 75):.1f}초, "
              f"2.5초 초과 {share_over(xs, LCP_GOOD_SECONDS):.0%}, p75 판정 {'좋음' if is_good(percentile_linear(xs, 75), LCP_GOOD_SECONDS) else '기준 밖'}")
    (m2, s2, n2), (m9, s9, n9) = TAN_THOUGHTFUL[2], TAN_THOUGHTFUL[9]
    print(f"[Tan 사려 깊음] 2초 {m2}(SD {s2}, n={n2}), 9초 {m9}(SD {s9:.2f}, n={n9}): 차이 {m9 - m2:.2f}점, "
          f"합동 SD {pooled_sd(s2, n2, s9, n9):.2f}, d {cohens_d(m2, s2, n2, m9, s9, n9):.2f}")
    print("[Tan 지연을 느꼈다고 답한 비율] " + ", ".join(f"{k}초 {a}/{b}={percent_1dp(a, b)}%" for k, (a, b) in TAN_FELT_DELAY.items()))
    print(f"60Hz 한 프레임 {1000 / 60:.1f}ms")
    print(f"200ms는 60Hz 화면에서 {frames_in(200):.0f}프레임")


if __name__ == "__main__":
    main()
