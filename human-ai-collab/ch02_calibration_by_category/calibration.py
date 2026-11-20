"""2장 실습: 문의 유형별, 달별로 AI 정답률과 수락률(의존의 거친 대용값)을 나란히 놓고 격차를 센다.

모든 숫자는 가상, 예시용이다. 신뢰를 모형으로 만들지 않고, 손으로 적은 고정 표를 센다.
"보정, 해상도, 특정성"은 Lee & See(2004)의 개념이고, 아래 대용값(격차, 범위 비율)은 이 장이 정한 측정 방식이다.
수락률은 의존(행동)이지 신뢰(태도)가 아니다. 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# (문의 유형, 월) -> (본 제안 수, AI가 맞은 수, 상담원이 받아들인 수). 가상 시나리오: 2개월 차에 배송 안내 정책이 바뀌었다.
LOG = {
    ("환불", 1): (20, 17, 15), ("배송", 1): (20, 16, 14), ("계정", 1): (20, 10, 14), ("결제", 1): (20, 8, 13),
    ("환불", 3): (20, 18, 16), ("배송", 3): (20, 10, 15), ("계정", 3): (20, 12, 13), ("결제", 3): (20, 9, 12),
}
CATEGORIES = ["환불", "배송", "계정", "결제"]


def accuracy(cat: str, month: int) -> float:
    n, correct, _ = LOG[(cat, month)]
    return correct / n


def acceptance(cat: str, month: int) -> float:
    n, _, acc = LOG[(cat, month)]
    return acc / n


def gap_points(cat: str, month: int) -> float:
    """수락률 - 정답률(퍼센트포인트). 양수는 능력보다 더 받아들인 쪽, 음수는 덜 받아들인 쪽."""
    return round((acceptance(cat, month) - accuracy(cat, month)) * 100, 1)


def overall(month: int) -> tuple[float, float]:
    n = sum(LOG[(c, month)][0] for c in CATEGORIES)
    correct = sum(LOG[(c, month)][1] for c in CATEGORIES)
    acc = sum(LOG[(c, month)][2] for c in CATEGORIES)
    return correct / n, acc / n


def range_ratio(month: int) -> float:
    """해상도의 거친 대용값: (유형별 수락률의 최대-최소) / (유형별 정답률의 최대-최소). 1에 가까울수록 능력 차이를 따라간다."""
    a = [accuracy(c, month) for c in CATEGORIES]
    r = [acceptance(c, month) for c in CATEGORIES]
    return (max(r) - min(r)) / (max(a) - min(a))


def main() -> None:
    for m in (1, 3):
        print(f"[{m}개월 차]")
        for c in CATEGORIES:
            n, k, a = LOG[(c, m)]
            print(f"  {c}: 정답 {k}/{n} = {accuracy(c, m):.0%}, 수락 {a}/{n} = {acceptance(c, m):.0%}, 격차 {gap_points(c, m):+.1f}p")
        ac, ar = overall(m)
        print(f"  전체: 정답률 {ac:.2%}, 수락률 {ar:.2%}, 격차 {(ar - ac) * 100:+.2f}p, 범위 비율 {range_ratio(m):.2f}")


if __name__ == "__main__":
    main()
