"""3장 실습: 가상 AI 도우미의 기능을 자동화 유형(Parasuraman 외 2000의 질문 틀)과 결정권(Zhang 외 2025의 세 축)으로 나누고, 기능마다 "물어볼 질문"을 뽑는다.

모든 기능 설명과 값은 가상, 예시용이다. 아래 규칙은 점검용 질문을 꺼내는 것이고, 어느 수준이 옳은지 정하는 기준이 아니다
(Parasuraman 외는 모형이 처방하지 않는다고 밝힌다). 수준 번호나 개수 기준은 쓰지 않는다.
Zhang 외의 값은 논문에 적힌 값을 옮긴 것이다(arXiv v2 기준, 2026-06 정오표 반영 여부는 확인하지 못했다).
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# ------------------------------------------------------------ 1) Zhang 외(2025)의 값
PATTERN_COUNTS = {"수동": 18, "반응": 34, "반능동": 12, "선제": 9, "협력": 33}  # 구체적 설계가 적힌 106편의 분류
TOTAL_CODED = 134    # 최종 말뭉치
CHI_PAPERS = 108     # 말뭉치 중 CHI 논문


def pattern_shares() -> dict:
    n = sum(PATTERN_COUNTS.values())
    return {k: v / n for k, v in PATTERN_COUNTS.items()}


def chi_share() -> float:
    return CHI_PAPERS / TOTAL_CODED


# ------------------------------------------------------------ 2) 가상 도우미 기능
# type: 획득, 분석, 결정, 행동(네 유형). starter: 누가 시작하나. locus: 주된 결정권. dynamic: 상황에 따라 분배가 바뀌나.
# granularity: 큰 방향(전략)인가 세부인가. human_check: 효과가 고객에게 닿기 전에 사람이 확인하는 단계가 있나.
# reversible: 되돌릴 수 있나. accept_event: 로그에 수락/수정 기록이 남나.
FEATURES = [
    {"name": "비슷한 과거 티켓 찾기", "type": "획득", "starter": "상담원", "locus": "상담원", "dynamic": False,
     "granularity": "세부", "human_check": True, "reversible": True, "accept_event": False},
    {"name": "티켓 자동 요약", "type": "분석", "starter": "AI", "locus": "상담원", "dynamic": False,
     "granularity": "세부", "human_check": True, "reversible": True, "accept_event": False},
    {"name": "답변 초안 제안", "type": "결정", "starter": "AI", "locus": "공동", "dynamic": False,
     "granularity": "세부", "human_check": True, "reversible": True, "accept_event": True},
    {"name": "환불 승인 추천", "type": "결정", "starter": "AI", "locus": "공동", "dynamic": True,
     "granularity": "전략", "human_check": True, "reversible": True, "accept_event": True},
    {"name": "답변 자동 발송", "type": "행동", "starter": "AI", "locus": "AI", "dynamic": False,
     "granularity": "세부", "human_check": False, "reversible": False, "accept_event": False},
    {"name": "고객에게 먼저 추가 질문 보내기", "type": "행동", "starter": "AI", "locus": "AI", "dynamic": False,
     "granularity": "세부", "human_check": False, "reversible": False, "accept_event": False},
]


def questions(f: dict) -> list:
    """기능 하나에 대해 팀이 먼저 물어볼 질문을 꺼낸다. 질문일 뿐 판정이 아니다."""
    qs = []
    if not f["human_check"] and not f["reversible"]:
        qs.append("사람이 확인하는 단계가 없고 되돌릴 수 없어요. 틀린 경우를 어떤 기록으로 알아낼까요?")
    if f["type"] in ("결정", "행동") and f["locus"] != "상담원":
        qs.append("결정권이 AI에게 있거나 나뉘어 있어요. 상담원이 끼어들 수 있는 시점은 언제인가요?")
    if not f["accept_event"]:
        qs.append("수락이나 수정 기록이 남지 않아요. 의존을 무엇으로 셀까요?")
    if f["dynamic"]:
        qs.append("분배가 상황에 따라 바뀌어요. 바뀌는 기준과 시점을 로그에 남기나요?")
    return qs


def n_with_accept_event() -> int:
    return sum(1 for f in FEATURES if f["accept_event"])


def main() -> None:
    sh = pattern_shares()
    print("[1] Zhang 외 패턴 분포(106편):", ", ".join(f"{k} {PATTERN_COUNTS[k]}편({sh[k]:.0%})" for k in PATTERN_COUNTS))
    print(f"[2] 말뭉치 중 CHI {CHI_PAPERS}/{TOTAL_CODED} = {chi_share():.0%}")
    print(f"[3] 가상 기능 {len(FEATURES)}개 중 수락 기록이 남는 기능 {n_with_accept_event()}개")
    for f in FEATURES:
        print(f"  {f['name']}: {f['type']}, 결정권 {f['locus']}, 질문 {len(questions(f))}개")


if __name__ == "__main__":
    main()
