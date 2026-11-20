"""2장 실습: 탭 한 번의 지연을 세 부분으로 나누고, 방문 하나의 INP와 방문들의 75번째 백분위를 구한다.

모든 시간은 가상, 예시용이다. 사람이 느낀 시간은 계산하지 않는다.
INP 기준(200ms 이하 좋음, 500ms 초과 나쁨)과 세 부분(입력 지연, 처리 시간, 표시 지연)은
web.dev 문서(inp, optimize-inp, 2026-11-20 확인)에서 정리했다.
"50개당 가장 높은 것 하나씩 무시"는 문서 요약 기준이며 경계 규칙(50번째에서 몇 개를 빼는지)은 원문으로 확인해야 한다.
이 실습은 floor(n / 50)개를 뺀다고 가정한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math

INP_GOOD_MS = 200
INP_POOR_MS = 500

# (이름, 입력 지연, 처리 시간, 표시 지연) 단위 ms. 가상: 상품 목록과 결제 화면에서 한 방문이 한 탭들
VISIT_A = [
    ("정렬 탭", 20, 60, 30),
    ("필터 열기", 15, 40, 25),
    ("상품 담기", 30, 90, 50),
    ("쿠폰 입력", 40, 120, 60),
    ("결제하기", 110, 180, 90),
]

# 가상: 결제하기를 누르자 로딩 표시를 곧바로 그리고, 결제 응답은 6000ms 뒤에 오는 방문
PAY_TAP = ("결제하기(로딩 표시를 먼저 그림)", 10, 30, 20)
PAY_RESPONSE_MS = 6000

# 가상: 방문 8번이 각각 남긴 INP(ms)
VISIT_INPS = [90, 110, 140, 160, 180, 220, 380, 150]


def latency(inter: tuple) -> int:
    _, a, b, c = inter
    return a + b + c


def ignored_count(n_interactions: int) -> int:
    return n_interactions // 50


def visit_inp(latencies: list[int]) -> int:
    xs = sorted(latencies, reverse=True)
    k = ignored_count(len(xs))
    return xs[k]


def grade(ms: float) -> str:
    if ms <= INP_GOOD_MS:
        return "좋음"
    return "개선 필요" if ms <= INP_POOR_MS else "나쁨"


def p75_rank(values: list[float]) -> float:
    xs = sorted(values)
    return xs[max(1, math.ceil(len(xs) * 0.75)) - 1]


def long_visit_latencies() -> list[int]:
    """가상: 60번 상호작용. 90ms짜리 56번과 정렬 110, 상품 담기 170, 쿠폰 220, 결제하기 380."""
    return [90] * 56 + [110, 170, 220, 380]


# 스스로 점검용 가상 값
QUIZ_TAP = ("점검 탭", 25, 100, 40)
QUIZ_PAY_WAIT_MS = 3000


def main() -> None:
    lat = [latency(i) for i in VISIT_A]
    for i, v in zip(VISIT_A, lat):
        print(f"[탭] {i[0]}: {i[1]}+{i[2]}+{i[3]} = {v}ms")
    print(f"[방문 A] 상호작용 {len(lat)}번, 빼는 개수 {ignored_count(len(lat))}, INP {visit_inp(lat)}ms({grade(visit_inp(lat))})")
    long_lat = long_visit_latencies()
    print(f"[긴 방문] 상호작용 {len(long_lat)}번, 빼는 개수 {ignored_count(len(long_lat))}, 가장 느린 탭 {max(long_lat)}ms, INP {visit_inp(long_lat)}ms({grade(visit_inp(long_lat))})")
    pay = latency(PAY_TAP)
    print(f"[결제 탭] 눌러서 다음 화면까지 {PAY_TAP[1]}+{PAY_TAP[2]}+{PAY_TAP[3]} = {pay}ms({grade(pay)}), 결제 응답까지 {PAY_RESPONSE_MS}ms 더 기다림, 합 {pay + PAY_RESPONSE_MS}ms")
    print(f"[점검 1] {QUIZ_TAP[1]}+{QUIZ_TAP[2]}+{QUIZ_TAP[3]} = {latency(QUIZ_TAP)}ms({grade(latency(QUIZ_TAP))})")
    print(f"[점검 2] 상호작용 110번이면 빼는 개수 {ignored_count(110)}")
    print(f"[점검 3] 탭 {pay}ms, 응답 대기 {QUIZ_PAY_WAIT_MS}ms, INP에 들어가는 값 {pay}ms")
    print(f"[방문 8번] 정렬 {sorted(VISIT_INPS)}")
    print(f"[방문 8번] ceil(0.75 x 8) = {math.ceil(0.75 * 8)}번째 값 = INP p75 {p75_rank(VISIT_INPS)}ms({grade(p75_rank(VISIT_INPS))}), 200ms 초과 방문 {sum(1 for v in VISIT_INPS if v > 200)}번({sum(1 for v in VISIT_INPS if v > 200) / len(VISIT_INPS):.0%})")


if __name__ == "__main__":
    main()
