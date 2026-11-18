"""4장 실습: 대기 UI 실험의 순서 배정과 표본 수 (가상 데이터).

모든 숫자는 가상의 예시이고, 실제 논문의 결과를 재현한 것이 아니다.
표준 라이브러리만 쓴다(random, math, statistics).

세 부분:
  1) balanced_latin_square: 4개 조건의 제시 순서를 균형 있게 배정한다 (5주차 역균형)
  2) n_per_group: 기대 효과 크기에서 한 집단당 필요한 인원을 계산한다 (5주차 표본 수)
  3) cohen_d, d_confidence_interval: 효과 크기와 95% 신뢰구간을 계산한다 (6주차)
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

Z = NormalDist()  # 표준정규분포


# ---------------------------------------------------------------- 1) 역균형
def balanced_latin_square(k: int) -> list[list[int]]:
    """짝수 k개 조건의 균형 라틴 방격을 만든다.

    각 줄은 한 참가자 유형의 조건 제시 순서이다(0부터 k-1까지의 조건 번호).
    첫 줄은 0, 1, k-1, 2, k-2, ... 이고, 다음 줄은 모든 값에 1을 더하고 k로 나눈 나머지를 쓴다.
    이렇게 하면 모든 조건이 모든 위치에 한 번씩 나오고,
    어떤 조건 바로 뒤에 어떤 조건이 오는지도 모든 짝이 같은 횟수로 나온다.
    홀수 k에서는 이 성질이 없어서(줄을 거꾸로 한 줄 더 써야 함) 지원하지 않는다.
    """
    if k % 2 != 0 or k < 2:
        raise ValueError("짝수 k만 지원한다")
    first = [0]
    lo, hi = 1, k - 1
    take_lo = True
    while len(first) < k:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    return [[(c + shift) % k for c in first] for shift in range(k)]


def assign_participants(n_participants: int, conditions: list[str], seed: int = 1) -> list[list[str]]:
    """참가자를 순서 줄에 번갈아 배정한다. 줄 순서는 seed로 섞은 뒤 돌려 쓴다."""
    square = balanced_latin_square(len(conditions))
    rng = random.Random(seed)
    rows = list(range(len(square)))
    rng.shuffle(rows)
    return [[conditions[c] for c in square[rows[i % len(rows)]]] for i in range(n_participants)]


# ---------------------------------------------------------------- 2) 표본 수
def n_per_group(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """서로 다른 사람으로 이뤄진 독립인 두 집단(집단 간)의 평균 비교(양측 검정)에서 한 집단당 필요한 인원의 근사값.

    같은 사람이 여러 조건을 보는 집단 내 설계에는 쓸 수 없다(이 파일은 그 계산을 하지 않는다).


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
    """가상 데이터: 두 집단 모두 평균 4, 표준편차 1인 정규분포에서 뽑고 한 집단만 true_d만큼 낮춘다.

    7점 척도 같은 상한은 두지 않았다(단순화).
    """
    rng = random.Random(seed)
    static = [rng.gauss(4.0, 1.0) for _ in range(n)]
    dynamic = [rng.gauss(4.0 - true_d, 1.0) for _ in range(n)]
    return static, dynamic


def ordered_pair_counts(square: list[list[int]]) -> dict[tuple[int, int], int]:
    """줄을 가로로 읽어 '앞 조건 바로 뒤 조건' 쌍이 몇 번 나오는지 센다."""
    counts: dict[tuple[int, int], int] = {}
    for row in square:
        for x, y in zip(row, row[1:]):
            counts[(x, y)] = counts.get((x, y), 0) + 1
    return counts


def total_people(n_per_cell: int, n_cells: int) -> int:
    """집단 간 설계에서 한 사람이 한 칸에만 들어가면 전체 인원은 칸 수 곱하기 칸당 인원이다."""
    return n_per_cell * n_cells


def excluded_share(excluded: int, remaining_before: int) -> float:
    """배정 뒤 규칙으로 뺀 사람의 비율(%)."""
    return 100 * excluded / remaining_before


def position_counts(orders: list[list[str]], condition: str) -> list[int]:
    """배정된 순서들에서 한 조건이 1번째, 2번째, ... 위치에 몇 번 나오는지 센다."""
    return [sum(1 for o in orders if o[pos] == condition) for pos in range(len(orders[0]))]


def main() -> None:
    conditions = ["빈 화면", "스피너", "진행 막대", "스켈레톤"]
    print("[1] 균형 라틴 방격(4조건)")
    for row in balanced_latin_square(4):
        print("  " + " ".join(map(str, row)))
    counts = ordered_pair_counts(balanced_latin_square(4))
    print(f"  앞뒤 쌍 {len(counts)}가지, 각각 {sorted(set(counts.values()))}번")
    print("\n[2] 참가자 8명 배정(시드 7)")
    for i, order in enumerate(assign_participants(8, conditions, seed=7), 1):
        print(f"  P{i}: " + " > ".join(order))
    print("\n[3] 한 집단당 필요한 인원(알파 0.05, 검정력 0.80)")
    for d in (0.2, 0.4, 0.5, 0.8, 0.85):
        print(f"  d={d}: {n_per_group(d)}명")
    print(f"\n[4] 집단 간 설계 전체 인원: 2x2 칸당 100명 -> {total_people(100, 4)}명")
    print(f"[5] 비교 6쌍을 모두 하면 알파를 0.05/6으로 낮춘다(본페로니 보정): d=0.4에서 {n_per_group(0.4, alpha=0.05 / 6)}명")
    print(f"[6] 칸당 100명, d=0.4의 검정력(정규 근사) {power_for_n(0.4, 100):.2f}, 칸당 50명이면 {power_for_n(0.4, 50):.2f}")
    print(f"[7] 가상이 아닌 논문 값 검산: 303명 중 63명 제외 = {excluded_share(63, 303):.1f}%")
    print(f"[8] z 합 {1.959964 + 0.841621:.3f}")
    for n in (8, 10):
        orders = assign_participants(n, conditions, seed=7)
        print(f"[9] {n}명 배정에서 '빈 화면'이 1~4번째 위치에 놓인 횟수 {position_counts(orders, '빈 화면')}")


if __name__ == "__main__":
    main()
