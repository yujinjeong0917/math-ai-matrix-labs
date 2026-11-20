"""0장 실습: 세 지표의 75번째 백분위를 모바일과 데스크톱으로 나눠 판정한다.

모든 방문 기록과 지표 값은 가상, 예시용이다. 사람이 느낀 시간은 계산하지 않는다.
기준값은 web.dev 문서(vitals, lcp, inp, cls, 2026-11-20 확인)가 권고하는 기준이다.
75번째 백분위는 "올림한 순위" 방식(작은 순서로 세운 값의 ceil(0.75 x n)번째)으로 구한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import math

LCP_GOOD_S = 2.5          # 이하이면 좋음
INP_GOOD_MS = 200         # 이하이면 좋음
INP_POOR_MS = 500         # 초과이면 나쁨, 그 사이는 개선 필요
CLS_GOOD = 0.1            # 이하이면 좋음
CLS_POOR = 0.25           # 초과이면 나쁨, 그 사이는 개선 필요
# LCP의 나쁨 기준(4.0초 초과)은 문서 본문이 아니라 그림 설명에서 확인한 값이라 판정에 쓰지 않는다.

# 가상: 상품 목록 페이지를 연 방문 10번씩의 LCP(초)
LCP_MOBILE = [1.9, 2.2, 2.4, 2.1, 3.3, 2.6, 2.0, 3.8, 2.3, 2.9]
LCP_DESKTOP = [1.2, 1.5, 1.4, 1.8, 1.3, 2.0, 1.6, 2.6, 1.7, 1.5]

# 가상: 대시보드가 이미 p75로 보여 준 INP(밀리초)와 CLS
P75_TABLE = {
    "모바일": {"INP": 240, "CLS": 0.08},
    "데스크톱": {"INP": 120, "CLS": 0.04},
}


def p75_rank(values: list[float]) -> float:
    xs = sorted(values)
    rank = max(1, math.ceil(len(xs) * 0.75))
    return xs[rank - 1]


def grade_lcp(v: float) -> str:
    return "좋음" if v <= LCP_GOOD_S else "좋음 아님"


def grade_inp(v: float) -> str:
    if v <= INP_GOOD_MS:
        return "좋음"
    return "개선 필요" if v <= INP_POOR_MS else "나쁨"


def grade_cls(v: float) -> str:
    if v <= CLS_GOOD:
        return "좋음"
    return "개선 필요" if v <= CLS_POOR else "나쁨"


def share_over(values: list[float], limit: float) -> float:
    return sum(1 for v in values if v > limit) / len(values)


def segment_row(lcp_values: list[float], inp_p75: float, cls_p75: float) -> dict:
    lcp = p75_rank(lcp_values)
    grades = {"LCP": grade_lcp(lcp), "INP": grade_inp(inp_p75), "CLS": grade_cls(cls_p75)}
    return {
        "LCP": lcp, "INP": inp_p75, "CLS": cls_p75, "grades": grades,
        "pass": all(g == "좋음" for g in grades.values()),
    }


def main() -> None:
    mobile = segment_row(LCP_MOBILE, P75_TABLE["모바일"]["INP"], P75_TABLE["모바일"]["CLS"])
    desktop = segment_row(LCP_DESKTOP, P75_TABLE["데스크톱"]["INP"], P75_TABLE["데스크톱"]["CLS"])
    for name, row in (("모바일", mobile), ("데스크톱", desktop)):
        g = row["grades"]
        print(f"[{name}] LCP p75 {row['LCP']}초({g['LCP']}), INP p75 {row['INP']}ms({g['INP']}), "
              f"CLS p75 {row['CLS']}({g['CLS']}) -> {'세 지표 모두 좋음' if row['pass'] else '통과 아님'}")
    print(f"[LCP 모바일 정렬] {sorted(LCP_MOBILE)}")
    print(f"[LCP 데스크톱 정렬] {sorted(LCP_DESKTOP)}")
    print(f"[LCP 2.5초 초과 비율] 모바일 {share_over(LCP_MOBILE, LCP_GOOD_S):.0%}, 데스크톱 {share_over(LCP_DESKTOP, LCP_GOOD_S):.0%}")
    pooled = LCP_MOBILE + LCP_DESKTOP
    print(f"[합친 20번] 정렬 {sorted(pooled)}")
    print(f"[합친 20번] ceil(0.75 x 20) = {math.ceil(len(pooled) * 0.75)}번째 값 = LCP p75 {p75_rank(pooled)}초({grade_lcp(p75_rank(pooled))})")


if __name__ == "__main__":
    main()
