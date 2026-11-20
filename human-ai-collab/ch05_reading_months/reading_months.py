"""5장 실습: 몇 달 치 팀 기록을 읽을 때 (1) 문의 구성이 바뀌면 전체 해결률이 오를 수 있음을 세고, (2) 기록 열로 어떤 질문에 답할 수 있는지 점검한다.

모든 숫자는 가상, 예시용이다. 팀 성과나 신뢰를 모형으로 흉내 내지 않고, 손으로 적은 고정 표를 센다.
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# ------------------------------------------------------------ 1) 문의 구성이 바뀐 두 달
# (월, 난이도) -> (문의 수, 한 번에 해결한 수). 가상 시나리오: 3개월 차에 쉬운 문의가 늘었다(예: 안내 문구가 바뀜).
TICKETS = {
    (1, "쉬움"): (20, 18), (1, "어려움"): (80, 40),
    (3, "쉬움"): (80, 68), (3, "어려움"): (20, 8),
}
LEVELS = ["쉬움", "어려움"]


def resolve_rate(month: int, level: str | None = None) -> float:
    levels = LEVELS if level is None else [level]
    n = sum(TICKETS[(month, l)][0] for l in levels)
    k = sum(TICKETS[(month, l)][1] for l in levels)
    return k / n


def mix_share(month: int, level: str) -> float:
    total = sum(TICKETS[(month, l)][0] for l in LEVELS)
    return TICKETS[(month, level)][0] / total


def fixed_mix_rate(month: int, weights_month: int) -> float:
    """다른 달의 문의 구성 비율(가중치)을 그대로 두고 이 달의 난이도별 해결률을 평균한 값."""
    return sum(resolve_rate(month, l) * mix_share(weights_month, l) for l in LEVELS)


# ------------------------------------------------------------ 2) 기록 열로 답할 수 있는 질문
QUESTIONS = {
    "수락률": {"accepted"},
    "오용과 불용": {"accepted", "ai_correct"},
    "문의 유형별 격차": {"accepted", "ai_correct", "category"},
    "달별 수락률 변화": {"accepted", "month"},
    "펼침 선택도": {"expanded", "ai_correct"},
    "문의 구성을 맞춘 해결률 비교": {"category", "outcome", "month"},
}


def answerable(columns: set) -> dict:
    return {q: need <= columns for q, need in QUESTIONS.items()}


def n_answerable(columns: set) -> int:
    return sum(answerable(columns).values())


MINIMAL = {"suggestion_id", "accepted", "month"}
FULL = {"suggestion_id", "accepted", "month", "ai_correct", "category", "expanded", "outcome"}


def main() -> None:
    for m in (1, 3):
        print(f"[{m}개월 차] 전체 해결률 {resolve_rate(m):.0%}, 쉬움 {resolve_rate(m, '쉬움'):.1%}(비중 {mix_share(m, '쉬움'):.0%}), "
              f"어려움 {resolve_rate(m, '어려움'):.1%}(비중 {mix_share(m, '어려움'):.0%})")
    print(f"[2] 1개월 차 구성으로 맞춘 3개월 차 해결률 {fixed_mix_rate(3, 1):.1%} (1개월 차 전체 {resolve_rate(1):.0%})")
    print(f"[3] 최소 열로 답할 수 있는 질문 {n_answerable(MINIMAL)}/{len(QUESTIONS)}, 전체 열로 {n_answerable(FULL)}/{len(QUESTIONS)}")


if __name__ == "__main__":
    main()
