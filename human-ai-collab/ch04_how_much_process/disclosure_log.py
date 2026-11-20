"""4장 실습: "펼쳐 보기" 기록에서 틀린 제안과 맞은 제안에 대한 펼침 비율과 그 차이(선택도)를 센다.

모든 숫자는 가상, 예시용이다. 사람의 점검 습관을 모형으로 만들지 않고, 손으로 적은 고정 표를 센다.
선택도(틀린 제안의 펼침 비율 - 맞은 제안의 펼침 비율)는 이 장이 정한 값이고, Pareek 외(2026)나 다른 논문의 지표가 아니다.
Pareek 외의 값(12명, 8명, 2명, 변형 5종 등)은 논문에 적힌 값을 상수로 옮겼다. 그 연구는 질적 관찰이라 이 코드는 그 값에 통계 계산을 하지 않는다.
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# ------------------------------------------------------------ 1) Pareek 외(2026)가 적은 값(옮겨 적기만 한다)
PAREEK = {"participants": 12, "wanted_toggle": 8, "said_less_info_needed_over_time": 2, "variants": 5, "tasks": 2}

# ------------------------------------------------------------ 2) 가상 펼침 기록
# 달별로 제안 40건. (AI가 틀렸나 맞았나) -> (제안 수, 화면의 "근거 펼치기"를 연 수)
LOG = {
    1: {"틀림": (12, 9), "맞음": (28, 15)},
    3: {"틀림": (12, 4), "맞음": (28, 8)},
}


def expand_rate(month: int, kind: str) -> float:
    n, opened = LOG[month][kind]
    return opened / n


def overall_expand_rate(month: int) -> float:
    n = sum(v[0] for v in LOG[month].values())
    opened = sum(v[1] for v in LOG[month].values())
    return opened / n


def selectivity_points(month: int) -> float:
    """틀린 제안의 펼침 비율 - 맞은 제안의 펼침 비율(퍼센트포인트)."""
    return round((expand_rate(month, "틀림") - expand_rate(month, "맞음")) * 100, 1)


def main() -> None:
    print("[1] Pareek 외가 적은 값:", PAREEK, "(질적 관찰이라 통계 계산을 하지 않는다)")
    for m in (1, 3):
        w, r = expand_rate(m, "틀림"), expand_rate(m, "맞음")
        print(f"[{m}개월 차] 전체 펼침 {overall_expand_rate(m):.0%}, 틀린 제안 {LOG[m]['틀림'][1]}/{LOG[m]['틀림'][0]} = {w:.1%}, "
              f"맞은 제안 {LOG[m]['맞음'][1]}/{LOG[m]['맞음'][0]} = {r:.1%}, 선택도 {selectivity_points(m):+.1f}p")


if __name__ == "__main__":
    main()
