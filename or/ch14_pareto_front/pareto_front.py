"""14장: 배송 속도 개선 인력을 세 지역에 나눌 때 비용과 지역 간 격차 중 무엇을 줄일까.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 완전 탐색(모든 정수 배분을 나열)한다.

재는 것: 두 목표(총비용, 가장 느린 지역의 배송일)를 함께 줄이는 배분 중 서로 지배당하지 않는 점(파레토 전선)과,
가중합·ε-제약(배송일 상한) 두 방식이 그 전선의 어느 점을 뽑는지.
입력: 지역 3곳 x 인원 0~6명(7^3 = 343가지). 출력: 전선 점(비용, 배송일, 배분).
대응(README 기대 출력): 전체 배분 343, 전선 점 15개, 가중합이 못 뽑는 7개
[(8.0, 2.94), (13.0, 2.058), (16.0, 2.0), (19.0, 1.4406), (22.0, 1.4), (25.0, 1.00842), (28.0, 0.98)],
배송일 2.0 이하의 최소 비용 16.0(배분 (0, 2, 4)).
"""
from itertools import product

REGIONS = ("A", "B", "C")
BASE_DAYS = {"A": 2.0, "B": 3.0, "C": 6.0}   # 인력을 더하기 전 평균 배송일(가상)
COST = {"A": 1.0, "B": 2.0, "C": 3.0}         # 한 명 추가 비용(백만원/월, 가상)
CUT = 0.30                                    # 한 명 추가할 때마다 배송일이 곱해지는 감소율: days = base * (1-CUT)^x
MAX_PER_REGION = 6


def days(region, x):
    # 한 명 늘 때마다 30%씩 줄어드는 지수 감소(수확 체감). 예: C에 2명이면 6 x 0.7^2 = 2.94일. 현실의 효과 곡선은 아님
    return BASE_DAYS[region] * (1 - CUT) ** x


def objectives(alloc):
    """(총비용, 가장 느린 지역의 배송일). 둘 다 작을수록 좋다."""
    cost = sum(COST[r] * x for r, x in zip(REGIONS, alloc))
    # 공정성을 "가장 느린 지역"으로 정의했다(모형의 선택). 평균이 아니라 최댓값이라 가장 느린 지역이 줄어야 값이 준다
    worst = max(days(r, x) for r, x in zip(REGIONS, alloc))
    # 6자리 반올림: 부동소수점 오차로 같은 점이 다른 점으로 갈리지 않게 한다(아래 dict 키·동점 비교에 쓰임)
    return round(cost, 6), round(worst, 6)


def all_plans():
    # 0..6명을 지역 3곳에 모두 나열: 7^3 = 343개 계획
    return [(a, objectives(a)) for a in product(range(MAX_PER_REGION + 1), repeat=3)]


def dominates(p, q):
    """p가 q를 지배: 두 목표 모두 같거나 좋고 하나는 확실히 좋다."""
    return p[0] <= q[0] and p[1] <= q[1] and (p[0] < q[0] or p[1] < q[1])


def pareto_front(plans=None):
    """같은 (비용, 배송일) 점은 하나만 남기고 지배당하지 않는 점을 비용 순으로 돌려준다."""
    plans = plans or all_plans()
    # 같은 (비용, 배송일)을 내는 배분이 여럿이면 사전순으로 가장 작은 배분 하나만 대표로 둔다(전선은 점을 센다, 배분 수가 아님)
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
    # weight = 비용 1(백만원)을 배송일 몇 일과 바꿀지. 동점이면 목표값, 그다음 배분 순으로 정해 결정적
    return min(plans, key=lambda t: (weight * t[1][0] + t[1][1], t[1], t[0]))


def weights_that_reach(front, grid=None, plans=None):
    """가중치를 훑어 가중합이 실제로 뽑는 전선 점과, 어떤 가중치로도 못 뽑는 전선 점."""
    plans = plans or all_plans()
    # 가중치 0~5를 0.001 간격으로 훑는다(경계는 0.001 단위로만 안다). 가중합은 전선 중 볼록한 쪽 점만 뽑아 7개를 놓친다
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
    # 1e-9: 배송일 부동소수점 비교에서 상한과 정확히 같은 계획이 빠지지 않게 하는 허용 오차
    ok = [t for t in plans if t[1][1] <= max_days + 1e-9]
    return min(ok, key=lambda t: (t[1][0], t[1][1], t[0])) if ok else None


def eps_reaches_all(front, plans=None):
    """전선의 각 점의 배송일을 상한으로 두고 ε-제약을 풀면, 그 점이 하나씩 나오는지 확인한다."""
    plans = plans or all_plans()
    # 가중합이 못 뽑은 점까지 ε-제약은 모두 뽑는다(README의 15개 전부 도달)
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
    # 이웃한 전선 점 사이의 손익분기 가중치 = 줄어든 배송일 / 늘어난 비용. 가중치가 이보다 작으면 비싼 쪽 점이 가중합에서 이긴다(비용을 덜 따지므로)
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
