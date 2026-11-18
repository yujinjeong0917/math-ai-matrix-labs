"""3장 실습: 조건에 따라 효과가 달라진다는 말을 칸별 평균에서 직접 계산한다.

입력은 Wang, Si & Ali (2025) 실험 3·4에 적힌 칸 평균과 Tan 외(2026)에 적힌 인원 수이다(2026-11-18 확인).
논문의 조절된 매개 지수는 쓰지 않는다(논문 안에서 두 조건의 간접효과 차이와 맞지 않는다).
계산 결과는 논문에 없는 직접 계산이다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math

# Wang 실험 3: 과제 몰입 낮음/높음 별 (동적, 정적) 지각된 대기 시간 평균(7점 척도 합성 지표)
EXP3 = {"낮은 몰입": (3.12, 4.83), "높은 몰입": (3.25, 3.68)}
# Wang 실험 4: 탐색 목적 쾌락/실용 별 (동적, 정적)
EXP4 = {"쾌락": (2.30, 3.20), "실용": (3.05, 3.25)}


def static_minus_dynamic(cell: tuple[float, float]) -> float:
    dynamic, static = cell
    return static - dynamic


def interaction_contrast(cells: dict[str, tuple[float, float]]) -> float:
    a, b = list(cells.values())
    return static_minus_dynamic(a) - static_minus_dynamic(b)


def anova_error_df(n_total: int, n_cells: int) -> int:
    """칸 하나에 한 사람씩 들어가는 요인 설계의 오차 자유도 = 전체 인원 - 칸 수."""
    return n_total - n_cells


# Wang 실험 3·4 본문의 분모 자유도(398, 396)와 대조한다.
REPORTED_DF2 = {"실험 3": 398, "실험 4": 396}


# 논문이 보고한 조건부 간접효과(몰입 낮음/높음, 쾌락/실용)와 조절된 매개 지수. 지수는 인용하지 않고 두 조건 차이와 맞는지만 본다.
INDIRECT = {"실험 3": (-0.432, -0.091, 0.432), "실험 4": (-0.524, -0.115, 0.527)}


def index_gap(low_or_hedonic: float, high_or_utilitarian: float, reported_index: float) -> tuple[float, float]:
    """두 조건부 간접효과의 차이와, 논문이 적은 지수를 돌려준다."""
    return high_or_utilitarian - low_or_hedonic, reported_index


# Tan 외(2026): 지연을 느꼈다고 답한 인원 / 전체 인원(과제 유형을 합친 값). 칸별 비율은 쓰지 않는다.
TAN_PERCEIVED = {"2초": (21, 79), "9초": (53, 80), "20초": (66, 81)}
# 사려 깊음 평균(SD)
TAN_THOUGHTFUL = {"2초": (5.76, 1.18), "9초": (6.09, 0.90), "20초": (6.11, 0.82)}


# 유용성 평균(2초, 9초). 20초 평균은 본문에 없다. SD가 없어 평균 차이만 낸다.
TAN_USEFUL = {"2초": 6.19, "9초": 6.44}


def pct1(x: int, n: int) -> float:
    """백분율을 소수 한 자리로 반올림(5는 올림)."""
    return math.floor(1000 * x / n + 0.5) / 10


def d_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    (ma, sa), (mb, sb) = a, b
    return (mb - ma) / math.sqrt((sa**2 + sb**2) / 2)


def main() -> None:
    for name, cells in (("실험 3 몰입", EXP3), ("실험 4 목적", EXP4)):
        parts = ", ".join(f"{k} {static_minus_dynamic(v):.2f}" for k, v in cells.items())
        print(f"{name}: 정적-동적 차이 {parts}, 차이의 차이 {interaction_contrast(cells):.2f}")
    print(f"2×2, N=400이면 오차 자유도 {anova_error_df(400, 4)} (논문: 실험 3 {REPORTED_DF2['실험 3']}, 실험 4 {REPORTED_DF2['실험 4']})")
    for k, (lo, hi, rep) in INDIRECT.items():
        diff, rep = index_gap(lo, hi, rep)
        print(f"{k} 조건부 간접효과 차이 {diff:.3f}, 논문 보고 지수 {rep}")
    for k, (x, n) in TAN_PERCEIVED.items():
        print(f"Tan 지연을 느꼈다 {k}: {x}/{n} = {pct1(x, n)}%")
    print(f"Tan 사려 깊음 2초 대 9초 차이 {TAN_THOUGHTFUL['9초'][0] - TAN_THOUGHTFUL['2초'][0]:.2f}점, d={d_between(TAN_THOUGHTFUL['2초'], TAN_THOUGHTFUL['9초']):.2f}")
    print(f"Tan 사려 깊음 2초 대 20초 차이 {TAN_THOUGHTFUL['20초'][0] - TAN_THOUGHTFUL['2초'][0]:.2f}점, d={d_between(TAN_THOUGHTFUL['2초'], TAN_THOUGHTFUL['20초']):.2f}")
    print(f"Tan 유용성 2초 대 9초 차이 {TAN_USEFUL['9초'] - TAN_USEFUL['2초']:.2f}점")


if __name__ == "__main__":
    main()
