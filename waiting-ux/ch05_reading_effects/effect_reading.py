"""5장 실습: 효과 크기와 신뢰구간을 읽고 가이드 표의 빈칸을 점검한다 (가상 데이터).

모든 숫자는 가상의 예시이고, 실제 논문의 결과를 재현한 것이 아니다.
표준 라이브러리만 쓴다(random, math, statistics).

네 부분:
  1) cohen_d, d_confidence_interval: 효과 크기와 95% 신뢰구간(정규 근사)을 계산한다
  2) simulate, coverage: 진짜 d를 아는 가상 데이터로 구간을 반복해 본다
  3) share_at_least: 한 번의 표본 d가 얼마나 흔들리는지 반복 횟수로 센다
  4) valid_guide_row, GUIDE_ROWS: 가이드 표 한 줄에 다섯 칸이 모두 있는지 검사한다
(균형 라틴 방격과 표본 수 함수는 4장 실습에 있고 여기서는 쓰지 않는다.)
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

Z = NormalDist()  # 표준정규분포


# ---------------------------------------------------------------- 1) 역균형
def balanced_latin_square(n: int) -> list[list[int]]:
    """짝수 n개 조건의 균형 라틴 방격을 만든다.

    각 줄은 한 참가자 유형의 조건 제시 순서이다(0부터 n-1까지의 조건 번호).
    첫 줄은 0, 1, n-1, 2, n-2, ... 이고, 다음 줄은 모든 값에 1을 더하고 n으로 나눈 나머지를 쓴다.
    이렇게 하면 모든 조건이 모든 위치에 한 번씩 나오고,
    어떤 조건 바로 뒤에 어떤 조건이 오는지도 모든 짝이 같은 횟수로 나온다.
    홀수 n에서는 이 성질이 없어서(줄을 거꾸로 한 줄 더 써야 함) 지원하지 않는다.
    """
    if n % 2 != 0 or n < 2:
        raise ValueError("짝수 n만 지원한다")
    first = [0]
    lo, hi = 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    return [[(c + shift) % n for c in first] for shift in range(n)]


def assign_participants(n_participants: int, conditions: list[str], seed: int = 1) -> list[list[str]]:
    """참가자를 순서 줄에 번갈아 배정한다. 줄 순서는 seed로 섞은 뒤 돌려 쓴다."""
    square = balanced_latin_square(len(conditions))
    rng = random.Random(seed)
    rows = list(range(len(square)))
    rng.shuffle(rows)
    return [[conditions[c] for c in square[rows[i % len(rows)]]] for i in range(n_participants)]


# ---------------------------------------------------------------- 2) 표본 수
def n_per_group(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """두 집단 평균 비교(양측)에서 한 집단당 필요한 인원의 근사값.

    n = 2 * ((z_{1-alpha/2} + z_{power}) / d)^2 + z_{1-alpha/2}^2 / 4
    뒤의 항은 t 분포가 정규분포보다 꼬리가 두꺼운 것을 대략 보정한다.
    올림해서 정수로 돌려준다. 정확한 값은 t 분포의 비중심 분포로 계산하는 도구(예: G*Power)를 쓴다.
    """
    if d <= 0:
        raise ValueError("d는 0보다 커야 한다")
    za = Z.inv_cdf(1 - alpha / 2)
    zb = Z.inv_cdf(power)
    n = 2 * ((za + zb) / d) ** 2 + za**2 / 4
    return math.ceil(n)


def power_for_n(d: float, n: int, alpha: float = 0.05) -> float:
    """한 집단 n명일 때 검정력의 정규 근사."""
    za = Z.inv_cdf(1 - alpha / 2)
    return Z.cdf(d * math.sqrt(n / 2) - za)


# ---------------------------------------------------------------- 3) 효과 크기와 신뢰구간
def cohen_d(a: list[float], b: list[float]) -> float:
    """합동 표준편차 기준 Cohen의 d = (평균a - 평균b) / 합동 SD."""
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (statistics.fmean(a) - statistics.fmean(b)) / sp


def d_confidence_interval(d: float, na: int, nb: int, level: float = 0.95) -> tuple[float, float]:
    """d의 신뢰구간(정규 근사). SE = sqrt((na+nb)/(na*nb) + d^2/(2(na+nb)))."""
    se = math.sqrt((na + nb) / (na * nb) + d**2 / (2 * (na + nb)))
    z = Z.inv_cdf(0.5 + level / 2)
    return d - z * se, d + z * se


def simulate(n: int, true_d: float, seed: int) -> tuple[list[float], list[float]]:
    """가상 데이터: 두 집단 모두 평균 4, 표준편차 1인 정규분포에서 뽑고 한 집단의 평균만 true_d만큼 낮춘다.

    seed는 무작위 난수의 시작값이다. 같은 seed면 같은 가상 데이터가 나온다.

    7점 척도 같은 상한은 두지 않았다(단순화).
    """
    rng = random.Random(seed)
    static = [rng.gauss(4.0, 1.0) for _ in range(n)]
    dynamic = [rng.gauss(4.0 - true_d, 1.0) for _ in range(n)]
    return static, dynamic


def coverage(n: int, true_d: float, reps: int = 200) -> tuple[float, float]:
    """시드 1..reps로 같은 실험을 반복해, 95% 구간이 진짜 d를 포함한 비율과 표본 d의 평균을 낸다."""
    hit = 0
    ds = []
    for seed in range(1, reps + 1):
        a, b = simulate(n, true_d, seed)
        d = cohen_d(a, b)
        lo, hi = d_confidence_interval(d, n, n)
        hit += lo <= true_d <= hi
        ds.append(d)
    return hit / reps, statistics.fmean(ds)


def share_at_least(n: int, true_d: float, threshold: float, reps: int = 200) -> int:
    """시드 1..reps 중 표본 d가 threshold 이상인 시드의 수."""
    return sum(1 for seed in range(1, reps + 1) if cohen_d(*simulate(n, true_d, seed)) >= threshold)


# 근거 강도: "얼마나 믿을 수 있는가". 이유 칸("왜 그런가")과 따로 적는다.
STRENGTH = (
    "직접 측정",  # 논문이 그 결과(대기 느낌, 평가)를 참가자에게 물어 잰 것
    "한 실험에서만",
    "한 실험에서만(논문 안 숫자 불일치 있음, 신뢰 낮음)",
    "논문이 측정하지 않음",
)
REQUIRED = ("조건", "대상", "이유", "근거", "강도")


def valid_guide_row(row: dict) -> bool:
    """가이드 표 한 줄에 다섯 칸이 모두 있고 강도가 정해진 값 중 하나인지 본다. 내용이 맞는지는 보지 않는다."""
    return all(row.get(k) for k in REQUIRED) and row["강도"] in STRENGTH


# 본문 가이드 표의 일곱 줄(요약). 이유 칸의 "저자 설명"은 논문이 낸 설명이라는 표시이다.
GUIDE_ROWS = [
    {"조건": "움직이는 로딩 화면이 멈춘 화면보다 지각된 대기 시간이 낮았음(15초, 보고된 평균)", "대상": "미국 성인 온라인 패널, 데스크톱",
     "이유": "시간에 쓰는 주의가 줄어서(저자 설명, 자기보고 매개)", "근거": "Wang 외 실험 1·2", "강도": STRENGTH[0]},
    {"조건": "과제 몰입이 높을 때 그 차이가 줄었다고 보고됨", "대상": "위 줄과 같은 표본",
     "이유": "시간 주의 차이가 줄어서(저자 설명)", "근거": "Wang 외 실험 3", "강도": STRENGTH[2]},
    {"조건": "빠르게 끝내려는 실용 목적일 때 그 차이가 줄었다고 보고됨", "대상": "위 줄과 같은 표본",
     "이유": "저자 설명", "근거": "Wang 외 실험 4", "강도": STRENGTH[2]},
    {"조건": "진행 정보가 있는 막대가 없는 쪽(반복 점)보다 빠르다고 평가되지 않았음(방향만)", "대상": "대학 중심 참여자 약 23명",
     "이유": "알 수 없음", "근거": "Fang 외 실험 1", "강도": STRENGTH[2]},
    {"조건": "서버가 첫 글자를 붙잡아 만든 인위적 지연에서 2초 응답이 9초 응답보다 덜 사려 깊다고 평가됨(작은 효과, 지연을 설계 도구로 권하지 않음)",
     "대상": "미국 AI 사용 경험자", "이유": "오래 걸리면 공들였다고 읽는다는 해석(저자 설명)", "근거": "Tan 외", "강도": STRENGTH[1]},
    {"조건": "모바일에서도 같은 효과가 난다", "대상": "모바일 사용자",
     "이유": "알 수 없음", "근거": "읽은 논문이 모바일을 허용하지 않음", "강도": STRENGTH[3]},
    {"조건": "움직이는 화면이 이탈을 줄인다", "대상": "실제 앱 사용자",
     "이유": "알 수 없음", "근거": "읽은 논문이 이탈 행동을 재지 않음", "강도": STRENGTH[3]},
]


def main() -> None:
    group_a, group_b = [5, 6, 7], [3, 4, 5]
    d = cohen_d(group_a, group_b)
    print(f"[1] 손계산 예: d={d:.1f}, 구간 {tuple(round(x, 2) for x in d_confidence_interval(d, 3, 3))}")
    print("[2] 가상 데이터, 진짜 d=0.4, 시드 11")
    for n in (20, 100, 400):
        x, y = simulate(n, 0.4, seed=11)
        dd = cohen_d(x, y)
        lo, hi = d_confidence_interval(dd, n, n)
        print(f"  n={n}: d={dd:.2f}, 구간 [{lo:.2f}, {hi:.2f}], 폭 {hi - lo:.2f}, 0 포함 {lo <= 0 <= hi}")
    print("[3] 시드 1..200 반복(진짜 d=0.4)")
    for n in (20, 100, 400):
        rate, mean_d = coverage(n, 0.4)
        print(f"  n={n}: 구간이 진짜 값을 포함한 비율 {rate:.1%}, 표본 d 평균 {mean_d:.2f}")
    k = share_at_least(100, 0.4, 0.58)
    print(f"[4] n=100에서 표본 d가 0.58 이상인 시드: 200개 중 {k}개({k / 200:.0%})")
    bad = {**GUIDE_ROWS[0], "대상": ""}
    ok_count = sum(valid_guide_row(r) for r in GUIDE_ROWS)
    print(f"[5] 가이드 표 {len(GUIDE_ROWS)}줄 검사: 통과 {ok_count}줄, 대상 칸을 비운 줄은 {'통과' if valid_guide_row(bad) else '걸러짐'}")


if __name__ == "__main__":
    main()
