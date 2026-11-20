"""0장 실습: 읽은 연구가 "몇 달 쓰는 동안의 변화"를 쟀는지 장부로 정리하고, 같은 수락률이 다른 이야기를 숨기는 가상 로그를 센다.

모든 숫자는 가상, 예시용이다(논문 값이 아니다). 신뢰나 습관을 시뮬레이션하지 않고, 손으로 적은 고정 표를 센다.
표준 라이브러리만 쓴다.

세 부분:
  1) SOURCES: 이 트랙이 읽은 네 편의 종류, 범위, 시간 길이, 읽은 판본(논문 값은 상수로 옮겨 적은 것)
  2) measured_over_months: 읽은 연구가 개월 단위 변화를 쟀는지 묻는 함수(모두 False여야 한다)
  3) month_counts, acceptance_rate, accepted_wrong: 가상 로그(정답 여부 x 수락 여부)에서 수락률과 틀린 제안을 받아들인 수를 센다
"""

from __future__ import annotations

# ------------------------------------------------------------ 1) 읽은 연구 장부
# 논문에 적힌 값만 옮겼다(Zhang은 arXiv v2 기준, 2026-06 정오표 반영 여부는 확인하지 못했다).
SOURCES = {
    "Zhang 외 2025": {
        "kind": "범위 검토(문헌 지도)",
        "scope": "공동 창작에서 AI 주도권을 다룬 HCI 논문 134편",
        "papers": 134,
        "pattern_papers": 106,
        "duration": "해당 없음(문헌 분류)",
        "edition": "arXiv v2",
        "measures_change_over_time": False,
    },
    "Pareek 외 2026": {
        "kind": "질적 관찰(설계 중심 비교 구조 관찰)",
        "scope": "다중 에이전트 화면 변형 5종, 과제 2종, 참가자 12명",
        "participants": 12,
        "duration": "참가자당 한 번의 대면 세션(약 65분)",
        "session_minutes": 65,
        "edition": "출판본",
        "measures_change_over_time": False,
    },
    "Lee & See 2004": {
        "kind": "통합 리뷰(개념 논문)",
        "scope": "자동화에 대한 신뢰와 의존 문헌",
        "duration": "해당 없음(리뷰)",
        "edition": "저자 출판 직전 원고",
        "measures_change_over_time": False,
    },
    "Parasuraman 외 2000": {
        "kind": "개념 논문",
        "scope": "자동화의 유형과 수준 틀",
        "duration": "해당 없음(개념 틀)",
        "edition": "DOI 기준 인용(공식 판본은 열람하지 못함)",
        "measures_change_over_time": False,
    },
}


def measured_over_months(name: str) -> bool:
    """이 장이 읽은 연구가 개월 단위 변화를 직접 쟀는가."""
    return SOURCES[name]["measures_change_over_time"]


def share_of(part: int, whole: int) -> float:
    return part / whole


# ------------------------------------------------------------ 2) 가상 로그
# 고객지원 팀이 AI 답변 제안 40건을 본 달의 기록(가상, 예시용).
# 키: (AI 제안이 맞았나, 상담원이 받아들였나)
MONTH_1 = {("맞음", "수락"): 20, ("맞음", "수정"): 8, ("틀림", "수락"): 4, ("틀림", "수정"): 8}
# 3개월 차 로그 두 가지. 둘 다 수락이 32건으로 같다.
MONTH_3_A = {("맞음", "수락"): 30, ("맞음", "수정"): 2, ("틀림", "수락"): 2, ("틀림", "수정"): 6}  # AI 제안이 더 자주 맞은 쪽
MONTH_3_B = {("맞음", "수락"): 24, ("맞음", "수정"): 4, ("틀림", "수락"): 8, ("틀림", "수정"): 4}  # AI 정답 비율은 그대로, 틀린 것도 받아들인 쪽


def total(log: dict) -> int:
    return sum(log.values())


def accepted(log: dict) -> int:
    return sum(v for (c, a), v in log.items() if a == "수락")


def acceptance_rate(log: dict) -> float:
    return accepted(log) / total(log)


def ai_correct(log: dict) -> int:
    return sum(v for (c, a), v in log.items() if c == "맞음")


def accepted_wrong(log: dict) -> int:
    return log[("틀림", "수락")]


def main() -> None:
    print("[1] 읽은 연구 장부")
    for k, v in SOURCES.items():
        print(f"  {k}: {v['kind']} | {v['scope']} | {v['duration']} | 개월 단위 변화를 쟀나: {measured_over_months(k)}")
    z = SOURCES["Zhang 외 2025"]
    print(f"[2] Zhang: 패턴 분류는 {z['pattern_papers']}편/{z['papers']}편 = {share_of(z['pattern_papers'], z['papers']):.0%}")
    for name, log in (("1개월 차", MONTH_1), ("3개월 차 A", MONTH_3_A), ("3개월 차 B", MONTH_3_B)):
        print(f"[3] {name}: 제안 {total(log)}건, 수락 {accepted(log)}건 = {acceptance_rate(log):.0%}, "
              f"AI 정답 {ai_correct(log)}건 = {ai_correct(log) / total(log):.0%}, 틀린 제안을 받아들인 수 {accepted_wrong(log)}건")


if __name__ == "__main__":
    main()
