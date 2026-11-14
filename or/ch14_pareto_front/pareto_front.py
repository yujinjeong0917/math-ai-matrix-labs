"""14장: 배송 속도 개선 인력을 세 지역에 나눌 때 비용과 지역 간 격차 중 무엇을 줄일까.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 완전 탐색(모든 정수 배분을 나열)한다.
"""
from itertools import product

REGIONS = ("A", "B", "C")
BASE_DAYS = {"A": 2.0, "B": 3.0, "C": 6.0}   # 인력을 더하기 전 평균 배송일(가상)
COST = {"A": 1.0, "B": 2.0, "C": 3.0}         # 한 명 추가 비용(백만원/월, 가상)
CUT = 0.30                                    # 한 명 추가할 때마다 배송일이 곱해지는 감소율: days = base * (1-CUT)^x
MAX_PER_REGION = 6


def days(region, x):
    return BASE_DAYS[region] * (1 - CUT) ** x


def objectives(alloc):
    """(총비용, 가장 느린 지역의 배송일). 둘 다 작을수록 좋다."""
    cost = sum(COST[r] * x for r, x in zip(REGIONS, alloc))
    worst = max(days(r, x) for r, x in zip(REGIONS, alloc))
    return round(cost, 6), round(worst, 6)


def all_plans():
    return [(a, objectives(a)) for a in product(range(MAX_PER_REGION + 1), repeat=3)]


def dominates(p, q):
    """p가 q를 지배: 두 목표 모두 같거나 좋고 하나는 확실히 좋다."""
    return p[0] <= q[0] and p[1] <= q[1] and (p[0] < q[0] or p[1] < q[1])


def pareto_front(plans=None):
    """같은 (비용, 배송일) 점은 하나만 남기고 지배당하지 않는 점을 비용 순으로 돌려준다."""
    plans = plans or all_plans()
    best = {}
    for alloc, obj in plans:
        best.setdefault(obj, alloc)
        best[obj] = min(best[obj], alloc)
    pts = list(best)
    front = [p for p in pts if not any(dominates(q, p) for q in pts)]
    return sorted((p[0], p[1], best[p]) for p in front)


def weighted_pick(weight, plans=None):
    """비용에 weight(일 단위로 환산한 가격), 느린 지역 배송일에 1을 곱해 더한 값이 가장 작은 계획."""
    plans = plans or all_plans()
    return min(plans, key=lambda t: (weight * t[1][0] + t[1][1], t[1], t[0]))


def weights_that_reach(front, grid=None, plans=None):
    """가중치를 훑어 가중합이 실제로 뽑는 전선 점과, 어떤 가중치로도 못 뽑는 전선 점."""
    plans = plans or all_plans()
    grid = grid or [i / 1000 for i in range(0, 5001)]
    reached = {}
    for w in grid:
        _, obj = weighted_pick(w, plans)
        reached.setdefault(obj, []).append(w)
    front_pts = {(c, d) for c, d, _ in front}
    return {k: (v[0], v[-1]) for k, v in reached.items() if k in front_pts}, sorted(front_pts - set(reached))


def eps_constraint(max_days, plans=None):
    """배송일이 max_days 이하인 계획 중 비용이 가장 낮은 것."""
    plans = plans or all_plans()
    ok = [t for t in plans if t[1][1] <= max_days + 1e-9]
    return min(ok, key=lambda t: (t[1][0], t[1][1], t[0])) if ok else None


def eps_reaches_all(front, plans=None):
    """전선의 각 점의 배송일을 상한으로 두고 ε-제약을 풀면, 그 점이 하나씩 나오는지 확인한다."""
    plans = plans or all_plans()
    return [eps_constraint(d, plans)[1] == (c, d) for c, d, _ in front]


def duplicate_points(plans=None):
    """같은 (비용, 가장 느린 배송일)을 내는 서로 다른 배분이 있는 점의 수. 전선에서는 하나로 센다."""
    plans = plans or all_plans()
    seen = {}
    for alloc, obj in plans:
        seen.setdefault(obj, []).append(alloc)
    return sum(1 for v in seen.values() if len(v) > 1)


def main():
    plans = all_plans()
    print("전체 배분 수:", len(plans))
    front = pareto_front(plans)
    print("파레토 전선 점 수:", len(front))
    for c, d, a in front:
        print(f"비용={c:5.1f} 가장 느린 지역={d:.3f}일 배분(A,B,C)={a}")
    reach, missed = weights_that_reach(front, plans=plans)
    print("== 가중합이 뽑는 점 (가중치 범위) ==")
    for k in sorted(reach):
        print(k, reach[k])
    print("어떤 가중치로도 못 뽑는 전선 점:", missed)
    print("== 전선 점 사이 손익분기 가중치(비용 증가분당 줄어드는 배송일) ==")
    for (c1, d1, _), (c2, d2, _) in zip(front, front[1:]):
        print(f"비용 {c1:g}->{c2:g}: {(d1 - d2) / (c2 - c1):.4f}")
    print("ε-제약(상한=각 전선 점의 배송일)으로 15개 모두 도달:", all(eps_reaches_all(front, plans)), "/ 점 수", len(eps_reaches_all(front, plans)))
    print("비용·배송일이 같은 배분이 여럿인 점의 수(전선 후보 포함):", duplicate_points(plans))
    print("== 배송일 상한(ε-제약) ==")
    for e in (2.0, 1.5, 1.0, 0.8):
        a, o = eps_constraint(e, plans)
        print(f"배송일 {e} 이하 -> 비용 최소 {o[0]} 배분 {a} 가장 느린 {o[1]:.3f}일")


if __name__ == "__main__":
    main()
