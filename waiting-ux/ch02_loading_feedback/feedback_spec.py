"""2장 실습: 로딩 화면 명세의 프레임 계산과, 논문들이 조작한 것이 무엇인지의 기록.

속도·프레임 숫자는 가상, 예시용이다. 체감 시간을 계산하지 않는다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math


def frames(duration_ms: float, fps: int = 60) -> int:
    return round(duration_ms / (1000 / fps))


def degrees_per_frame(turns_per_second: float, fps: int = 60) -> float:
    return 360 * turns_per_second / fps


def total_turns(turns_per_second: float, duration_s: float) -> float:
    return turns_per_second * duration_s


def linear_fill(t_s: float, total_s: float) -> float:
    """진행 막대를 시간에 비례해 채울 때의 비율(0~1). 실제 진행률이 아니라 연출 규칙이다."""
    return min(1.0, max(0.0, t_s / total_s))


# Wang, Si & Ali (2025)가 실험마다 바꾼 것(본문에 적힌 설계, 2026-11-18 확인한 MDPI 공개본).
# 실험 1, 2는 움직임 유무만 바꿨다. 실험 3, 4는 움직임 유무에 더해 지시문(몰입, 탐색 목적)도 바꿨고,
# 정지 조건의 서술이 논문 안에서 어긋나 자극 모양(막대 여부 등)은 적지 않는다.
WANG_MANIPULATED = {
    1: ["움직임 유무"],
    2: ["움직임 유무"],
    3: ["움직임 유무", "과제 몰입 지시문"],
    4: ["움직임 유무", "탐색 목적 지시문"],
}
WANG_N = {1: 198, 2: 411, 3: 400, 4: 400}

# Fang, Hu, Shi & Liu (2022)가 바꾼 것(2026-11-18 확인한 열린 PDF). 참여자 수와 통계값은 논문 안에서 서로 맞지 않는 곳이 있어 효과 크기는 적지 않는다.
FANG_MANIPULATED = {"진행 정보 유무(막대 대 반복해서 움직이는 점)", "만화 장식", "음악"}
# 이 장에서 읽은 두 논문이 어느 쪽도 바꾸지 않은 것
NEVER_MANIPULATED = {"스켈레톤 화면", "스피너와 진행 막대의 직접 비교"}


def f_p_value_df1(F: float, df2: int, steps: int = 20000) -> float:
    """자유도 (1, df2)인 F의 p값. t=√F인 양측 t 검정과 같다. t 분포 밀도를 심프슨 공식으로 적분한다."""
    t = math.sqrt(F)
    c = math.exp(math.lgamma((df2 + 1) / 2) - math.lgamma(df2 / 2)) / math.sqrt(df2 * math.pi)

    def pdf(x: float) -> float:
        return c * (1 + x * x / df2) ** (-(df2 + 1) / 2)

    h = t / steps
    total = pdf(0) + pdf(t) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, steps))
    return 1 - 2 * (total * h / 3)


def main() -> None:
    print(f"15초 = {frames(15000)}프레임(60Hz)")
    print(f"1초에 한 바퀴 도는 고리: 프레임당 {degrees_per_frame(1):.1f}도, 15초 동안 {total_turns(1, 15):.0f}바퀴")
    print(f"15초에 걸쳐 채우는 막대: 7.5초에 {linear_fill(7.5, 15):.0%}")
    print(f"200ms = {frames(200)}프레임, 16.7ms = {frames(16.7)}프레임")
    print(f"Fang 외가 바꾼 것 {len(FANG_MANIPULATED)}가지, 두 논문이 바꾸지 않은 것 {sorted(NEVER_MANIPULATED)}")
    for k, v in WANG_MANIPULATED.items():
        print(f"Wang 외 실험 {k}(N={WANG_N[k]}): 바꾼 것 {', '.join(v)}")
    print(f"Fang 외 장식 F=1.108, 자유도 (1,21)의 p={f_p_value_df1(1.108, 21):.3f}(본문은 0.034, 표는 0.304)")
    print("Fang 외 쌍 비교 예: (-1×1 + 1×16 + 0×6) / 23 = %.2f" % ((-1 * 1 + 16 + 0) / 23))


if __name__ == "__main__":
    main()
