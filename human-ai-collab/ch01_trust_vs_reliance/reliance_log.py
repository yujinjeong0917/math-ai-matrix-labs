"""1장 실습: 신뢰(태도)와 의존(행동)을 구분해 가상 로그에서 오용, 불용, 적정 의존을 센다.

모든 숫자는 가상, 예시용이다. 신뢰를 모형으로 만들지 않는다. 설문 점수도 손으로 적은 고정 값이다.
오용, 불용, 적정 의존의 계산식은 이 장이 정한 측정 방식이고, Lee & See(2004)가 정한 측정 절차가 아니다.
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# 키: 칸 이름. CA=AI 맞음+수락, CO=AI 맞음+수정, WA=AI 틀림+수락, WO=AI 틀림+수정
# 상담원 4명이 한 달 동안 각각 AI 제안 20건을 본 기록과, 같은 달 설문의 "AI를 믿는 정도" 1~5점 한 문항(가상)
AGENTS = {
    "가": {"CA": 12, "CO": 2, "WA": 1, "WO": 5, "trust": 4},
    "나": {"CA": 8, "CO": 6, "WA": 1, "WO": 5, "trust": 5},
    "다": {"CA": 11, "CO": 1, "WA": 5, "WO": 3, "trust": 2},
    "라": {"CA": 9, "CO": 3, "WA": 2, "WO": 6, "trust": 3},
}


def n_total(a: dict) -> int:
    return a["CA"] + a["CO"] + a["WA"] + a["WO"]


def acceptance_rate(a: dict) -> float:
    """수락률: 수락한 제안 / 본 제안. 이것은 의존(행동)의 한 가지 숫자이다."""
    return (a["CA"] + a["WA"]) / n_total(a)


def misuse_rate(a: dict) -> float:
    """오용 비율(이 장의 정의): 틀린 제안 가운데 받아들인 비율."""
    return a["WA"] / (a["WA"] + a["WO"])


def disuse_rate(a: dict) -> float:
    """불용 비율(이 장의 정의): 맞은 제안 가운데 고친(받아들이지 않은) 비율."""
    return a["CO"] / (a["CA"] + a["CO"])


def appropriate_rate(a: dict) -> float:
    """적정 의존 비율(이 장의 정의): 맞은 제안은 받아들이고 틀린 제안은 고친 건수 / 본 제안."""
    return (a["CA"] + a["WO"]) / n_total(a)


def ranks_desc(values: dict) -> dict:
    """큰 값이 1등인 순위(동점 없음을 가정한다)."""
    order = sorted(values, key=lambda k: -values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def rank_gaps(agents: dict) -> dict:
    """설문 점수 순위와 수락률 순위의 차이(절댓값)."""
    tr = ranks_desc({k: v["trust"] for k, v in agents.items()})
    ac = ranks_desc({k: acceptance_rate(v) for k, v in agents.items()})
    return {k: abs(tr[k] - ac[k]) for k in agents}


def main() -> None:
    print("[1] 상담원별 가상 로그(제안 20건)")
    for k, a in AGENTS.items():
        print(f"  {k}: 수락 {a['CA'] + a['WA']}건 = {acceptance_rate(a):.0%}, 오용 {misuse_rate(a):.1%}, "
              f"불용 {disuse_rate(a):.1%}, 적정 {appropriate_rate(a):.0%}, 설문 {a['trust']}점")
    gaps = rank_gaps(AGENTS)
    print("[2] 설문 순위와 수락률 순위의 차이:", ", ".join(f"{k} {v}" for k, v in gaps.items()))
    tot = {c: sum(a[c] for a in AGENTS.values()) for c in ("CA", "CO", "WA", "WO")}
    print("[3] 네 명 합계:", tot, f"수락률 {(tot['CA'] + tot['WA']) / sum(tot.values()):.1%}")


if __name__ == "__main__":
    main()
