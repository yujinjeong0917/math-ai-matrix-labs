"""2장 실습: 가상 응답으로 문항 일관성(크론바흐 알파), 순위 상관(스피어만), 측정 계획 경고를 계산한다.

입력은 virtual_data.json(가상, 예시용)이다. 계산법을 보이는 용도이며 어떤 실제 효과도 말하지 않는다.
표본이 5명이라 알파는 매우 불안정하다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import json
import os
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "virtual_data.json"), encoding="utf-8") as f:
        return json.load(f)


def sample_var(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def cronbach_alpha(rows: list[list[float]]) -> tuple[float, float, float]:
    """(항목 분산의 합, 총점 분산, 알파). 알파 = k/(k-1) * (1 - 항목 분산 합 / 총점 분산)."""
    k = len(rows[0])
    item_vars = [sample_var([r[j] for r in rows]) for j in range(k)]
    total_var = sample_var([sum(r) for r in rows])
    alpha = k / (k - 1) * (1 - sum(item_vars) / total_var)
    return sum(item_vars), total_var, alpha


def ranks(xs: list[float]) -> list[float]:
    """오름차순 순위(동점은 평균 순위)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            out[order[t]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    """순위의 피어슨 상관. 동점이 없으면 1 - 6*sum(d^2)/(n(n^2-1))과 같다."""
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5


def plan_flags(plan: list[dict]) -> list[str]:
    flags = []
    if all(m["kind"] == "자기보고" for m in plan):
        flags.append("모두 자기보고")
    for a, b in combinations(plan, 2):
        shared = sorted(set(a["terms"]) & set(b["terms"]))
        if shared:
            flags.append(f"문항 겹침 후보({a['name']}, {b['name']}: {', '.join(shared)})")
    return flags


def main() -> None:
    d = load()
    rows = d["responses"]
    sv, tv, alpha = cronbach_alpha(rows)
    totals = [sum(r) for r in rows]
    k = len(rows[0])
    item_vars = [sample_var([r[j] for r in rows]) for j in range(k)]
    mean_t = sum(totals) / len(totals)
    ss_t = sum((x - mean_t) ** 2 for x in totals)
    print(f"[0] 항목별 분산 {[round(v, 1) for v in item_vars]}, 총점 평균 {mean_t:.1f}, 총점 편차 제곱합 {ss_t:.1f}")
    print(f"[1] 총점 {totals}, 항목 분산 합 {sv:.1f}, 총점 분산 {tv:.1f}, 알파 {alpha:.3f} (응답 {len(rows)}명)")
    print(f"[1b] 1 - 항목 분산 합 / 총점 분산 = {1 - sv / tv:.3f}")
    ra = spearman(totals, d["seconds_a"])
    rb = spearman(totals, d["seconds_b"])
    print(f"[2] 총점과 완료 시간의 순위 상관: 시간 자료 A {ra:.3f}, 시간 자료 B {rb:.3f}")
    for name, plan in d["plans"].items():
        flags = plan_flags(plan)
        print(f"[3] 계획 {name} 경고 {len(flags)}개: {'; '.join(flags) if flags else '없음'}")


if __name__ == "__main__":
    main()
