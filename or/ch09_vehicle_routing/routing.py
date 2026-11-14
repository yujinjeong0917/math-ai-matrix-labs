"""9장. 배송 경로 묶기 (VRP). 표준 라이브러리만 사용, 정지점 8곳을 전수 탐색으로 푼다.
모든 좌표·수요·시간창은 학습용 가상값이다."""
from itertools import permutations
from functools import lru_cache

DEPOT = (0, 0)
# 정지점: 이름 -> (x, y, 수요(박스), 시간창 (이른 도착, 늦은 도착 한도, 분))
STOPS = {
    1: (2, 6, 3, None), 2: (5, 7, 4, None), 3: (8, 6, 2, None), 4: (9, 2, 3, None),
    5: (6, 1, 4, None), 6: (3, 2, 2, None), 7: (-4, 3, 3, None), 8: (-5, 7, 4, None),
}
CAPACITY = 10
SERVICE = 5  # 정지점마다 내려놓는 데 걸리는 분


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pos(i):
    return DEPOT if i == 0 else STOPS[i][:2]


def route_len(order, speed=1.0):
    p, total = 0, 0.0
    for s in order:
        total += dist(pos(p), pos(s)) / speed
        p = s
    return total + dist(pos(p), pos(0)) / speed


def demand(subset):
    return sum(STOPS[s][2] for s in subset)


def best_route(subset, windows=None, speed=1.0):
    """subset 안 모든 순서를 훑어 가장 짧은(분 기준) 실행 가능한 경로를 돌려준다. 없으면 (None, None)."""
    best = (None, None)
    for order in permutations(subset):
        if windows is not None and not feasible_windows(order, windows, speed):
            continue
        d = route_len(order, speed)
        if best[0] is None or d < best[0]:
            best = (d, order)
    return best


def feasible_windows(order, windows, speed=1.0):
    t, p = 0.0, 0
    windows = windows or {}
    for s in order:
        t += dist(pos(p), pos(s)) / speed
        lo, hi = windows.get(s, (0, 10**9))
        t = max(t, lo)
        if t > hi:
            return False
        t += SERVICE
        p = s
    return True


def solve(max_vehicles=3, windows=None, speed=1.0, capacity=CAPACITY):
    """모든 정지점을 용량 이하의 경로 max_vehicles개 이하로 나눠 총 이동거리를 최소화한다(전수 탐색)."""
    n = len(STOPS)
    full = (1 << n) - 1
    ids = sorted(STOPS)
    cost = {}
    for mask in range(1, full + 1):
        sub = [ids[i] for i in range(n) if mask >> i & 1]
        if demand(sub) > capacity:
            continue
        c, o = best_route(sub, windows, speed)
        if c is not None:
            cost[mask] = (c, o)

    @lru_cache(None)
    def go(rest, k):
        if rest == 0:
            return (0.0, ())
        if k == 0:
            return (float("inf"), ())
        low = rest & -rest  # 남은 점 중 가장 낮은 번호를 반드시 포함해 중복 분할을 막는다
        best = (float("inf"), ())
        sub = rest
        while sub:
            if sub & low and sub in cost:
                c, o = cost[sub]
                rc, rr = go(rest ^ sub, k - 1)
                if c + rc < best[0]:
                    best = (c + rc, (o,) + rr)
            sub = (sub - 1) & rest
        return best

    total, routes = go(full, max_vehicles)
    return (None, None) if total == float("inf") else (total, list(routes))


def one_big_route_cost():
    """용량 무시하고 한 대가 전부 도는 최단 경로(하한 비교용)."""
    return best_route(sorted(STOPS))


def sweep_routes(capacity=CAPACITY):
    """번호 순서대로 싣다가 용량이 차면 새 차를 쓰는 단순 규칙."""
    routes, cur, load = [], [], 0
    for s in sorted(STOPS):
        if load + STOPS[s][2] > capacity:
            routes.append(tuple(cur)); cur, load = [], 0
        cur.append(s); load += STOPS[s][2]
    routes.append(tuple(cur))
    return sum(route_len(r) for r in routes), routes


def nearest_neighbor(capacity=CAPACITY):
    left, routes = set(STOPS), []
    while left:
        p, load, r = 0, 0, []
        while True:
            cand = [s for s in left if load + STOPS[s][2] <= capacity]
            if not cand:
                break
            s = min(cand, key=lambda q: (dist(pos(p), pos(q)), q))
            r.append(s); left.remove(s); load += STOPS[s][2]; p = s
        routes.append(tuple(r))
    return sum(route_len(r) for r in routes), routes


def feasible_counts():
    """용량 안에 드는 정지점 묶음의 수와, 그 묶음 안에서 훑어야 하는 방문 순서의 총수."""
    from itertools import combinations
    from math import factorial
    n_sets = n_orders = 0
    for k in range(1, len(STOPS) + 1):
        for sub in combinations(sorted(STOPS), k):
            if demand(sub) <= CAPACITY:
                n_sets += 1
                n_orders += factorial(k)
    return n_sets, n_orders


WINDOWS_LOOSE = {1: (0, 9), 4: (0, 12), 7: (0, 10)}            # 출발 후 이 분까지 도착해야 한다(가상)
WINDOWS_TIGHT = {1: (0, 9), 4: (0, 12), 5: (0, 8), 7: (0, 10)}


if __name__ == "__main__":
    print("총 수요", demand(STOPS), "박스, 용량", CAPACITY)
    print("단순 순서 묶기", sweep_routes())
    print("가까운 곳 먼저", nearest_neighbor())
    print("전수 탐색(최대 3대)", solve(3))
    print("전수 탐색(최대 2대)", solve(2))
    print("한 대, 용량 무시", one_big_route_cost())
    print("시간창 느슨, 3대", solve(3, WINDOWS_LOOSE))
    print("시간창 빡빡, 3대", solve(3, WINDOWS_TIGHT))
    print("시간창 빡빡, 4대", solve(4, WINDOWS_TIGHT))
    print("교통 혼잡(속도 0.7배), 느슨한 시간창, 3대", solve(3, WINDOWS_LOOSE, 0.7))
    print("교통 혼잡(속도 0.7배), 느슨한 시간창, 4대", solve(4, WINDOWS_LOOSE, 0.7))
    print("용량 13박스 차량, 최대 2대", solve(2, None, 1.0, 13))
    print("용량 10박스로 짠 부분집합 수/순서 수", feasible_counts())
