"""3장 실습: 레이아웃 이동 점수(영향 비율 x 거리 비율)와 세션 창으로 CLS를 구한다.

모든 이동 기록은 가상, 예시용이다. 사람이 느낀 불편은 계산하지 않는다.
점수 식, 세션 창(이동 사이 1초 미만, 한 창 최대 5초, 가장 큰 창이 CLS), 최근 입력 500ms 제외,
기준(0.1 이하 좋음, 0.25 초과 나쁨)은 web.dev 문서(cls, 2026-11-20 확인)를 정리한 것이다.
단순화: 이동하는 요소는 화면 폭 전체를 쓰고 세로로만 움직인다고 보고,
영향 비율 = min(1, 요소 높이 비율 + 이동 거리 비율)로 계산한다. 화면은 세로가 더 긴 휴대폰이라고 가정한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

CLS_GOOD = 0.1
CLS_POOR = 0.25
GAP_S = 1.0       # 이동 사이 간격이 이보다 짧아야 같은 창
MAX_WINDOW_S = 5.0
RECENT_INPUT_S = 0.5

# web.dev 문서가 든 계산 예(영향 비율, 거리 비율). 실제 사이트의 측정값이 아니다.
DOC_EXAMPLES = [(0.75, 0.25), (0.5, 0.14), (0.60, 0.30)]


def score(impact: float, distance: float) -> float:
    return round(impact * distance, 6)


def vertical_shift_score(height_frac: float, distance_frac: float) -> float:
    impact = min(1.0, height_frac + distance_frac)
    return score(impact, distance_frac)


def grade(cls: float) -> str:
    if cls <= CLS_GOOD:
        return "좋음"
    return "개선 필요" if cls <= CLS_POOR else "나쁨"


def session_windows(shifts: list[tuple[float, float, bool]]) -> list[dict]:
    """shifts: (시각 초, 점수, 최근 입력 뒤 500ms 이내 여부). 입력 직후 이동은 뺀다."""
    kept = [(t, s) for t, s, recent in shifts if not recent]
    windows: list[dict] = []
    for t, s in kept:
        w = windows[-1] if windows else None
        if w and (t - w["last"]) < GAP_S and (t - w["start"]) <= MAX_WINDOW_S:
            w["last"] = t
            w["sum"] = round(w["sum"] + s, 6)
            w["n"] += 1
        else:
            windows.append({"start": t, "last": t, "sum": s, "n": 1})
    return windows


def cls_of(shifts: list[tuple[float, float, bool]]) -> float:
    ws = session_windows(shifts)
    return max((w["sum"] for w in ws), default=0.0)


# 가상: 결제 화면을 여는 방문의 이동 기록(시각 초, 요소 높이 비율, 이동 거리 비율, 입력 직후 여부)
CHECKOUT_RAW = [
    (1.2, 0.50, 0.10, False),   # 쿠폰 배너가 끼어듦
    (1.7, 0.45, 0.15, False),   # 상품 이미지가 늦게 뜸
    (2.4, 0.40, 0.05, False),   # 추천 위젯
    (6.0, 0.30, 0.02, False),   # 글꼴 교체
    (8.0, 0.50, 0.20, True),    # 사용자가 장바구니를 펼침(탭 직후)
]

# 가상: 0.8초 간격으로 작은 이동이 계속되는 방문. 점수는 각각 0.02
STEADY = [(round(0.5 + 0.8 * i, 1), 0.02, False) for i in range(8)]


def checkout_shifts() -> list[tuple[float, float, bool]]:
    return [(t, vertical_shift_score(h, d), r) for t, h, d, r in CHECKOUT_RAW]


# 스스로 점검용 가상 값
QUIZ_A = [(1.0, 0.03, False), (1.5, 0.04, False), (2.0, 0.05, False)]
QUIZ_B = [(1.0, 0.03, False), (1.5, 0.04, False), (3.5, 0.05, False)]


def main() -> None:
    for impact, dist in DOC_EXAMPLES:
        print(f"[문서의 예] {impact} x {dist} = {score(impact, dist)}")
    shifts = checkout_shifts()
    for (t, h, d, r), (_, s, _) in zip(CHECKOUT_RAW, shifts):
        print(f"[이동] {t}초: 높이 {h}, 거리 {d}, 영향 {min(1.0, h + d)} -> 점수 {s}{' (입력 직후라 뺌)' if r else ''}")
    ws = session_windows(shifts)
    for i, w in enumerate(ws, 1):
        print(f"[창 {i}] {w['start']}초부터 {w['last']}초, 이동 {w['n']}번, 합 {w['sum']}")
    print(f"[결제 방문 CLS] {cls_of(shifts)}({grade(cls_of(shifts))}), 모든 이동의 합(입력 직후 포함) {round(sum(s for _, s, _ in shifts), 6)}")
    print(f"[간격] 1.2초와 1.7초 {round(1.7 - 1.2, 1)}초, 1.7초와 2.4초 {round(2.4 - 1.7, 1)}초, 2.4초와 6.0초 {round(6.0 - 2.4, 1)}초")
    ws2 = session_windows(STEADY)
    print(f"[꾸준한 이동] 이동 {len(STEADY)}번, 합 {round(sum(s for _, s, _ in STEADY), 6)}, 창 {len(ws2)}개({[w['n'] for w in ws2]}번씩), CLS {cls_of(STEADY)}")
    print(f"[점검 1] 0.5 x 0.2 = {score(0.5, 0.2)}({grade(score(0.5, 0.2))})")
    print(f"[점검 2] 이동 1.0, 1.5, 2.0초: CLS {cls_of(QUIZ_A)}({grade(cls_of(QUIZ_A))}), 세 번째가 3.5초이면 CLS {cls_of(QUIZ_B)}({grade(cls_of(QUIZ_B))}), 창 {[w['n'] for w in session_windows(QUIZ_B)]}번씩")
    footer = vertical_shift_score(0.5, 0.18)
    pay = vertical_shift_score(0.5, 0.18)
    print(f"[같은 점수] 아래쪽 안내 문구가 밀림 {footer}, 결제 버튼이 밀림 {pay}")


if __name__ == "__main__":
    main()
