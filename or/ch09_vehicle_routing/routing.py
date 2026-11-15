"""9장. 배송 경로 묶기 (VRP, 차량 경로 문제). 표준 라이브러리만 사용, 정지점 8곳을 전수 탐색으로 푼다.
모든 좌표·수요·시간창은 학습용 가상값이다.

재는 것: 정지점 8곳을 용량 10박스 이하의 경로 몇 개로 나눠 총 이동거리를 최소화했을 때의 값, 그리고 단순 규칙(번호 순서, 가까운 곳 먼저)과의 차이.
입력: STOPS(좌표, 수요), 차량 수, 용량, 시간창, 속도. 출력: (총 이동거리, 경로 목록). 거리는 맨해튼 거리, 속도 1이면 거리 1당 1분.
대응(README 기대 출력): 단순 순서 78, 가까운 곳 먼저 82, 전수 탐색(최대 3대) 76, 최대 2대는 해 없음(총 수요 25 > 2대 x 10),
시간창 느슨 3대 78 / 빡빡 3대 해 없음 / 빡빡 4대 88, 용량 13박스 최대 2대 62, 평가하는 묶음 85개와 방문 순서 412가지,
속도 0.7배는 3대·4대 모두 해 없음.
"""
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
    # 맨해튼 거리(격자 도로 가정). 실제 도로망·일방통행은 반영하지 않는다
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pos(i):
    return DEPOT if i == 0 else STOPS[i][:2]


def route_len(order, speed=1.0):
    # 창고(0)에서 출발해 순서대로 방문하고 창고로 돌아온다. speed로 나눠 분 단위로 바꾼다
    p, total = 0, 0.0
    for s in order:
        total += dist(pos(p), pos(s)) / speed
        p = s
    return total + dist(pos(p), pos(0)) / speed


def demand(subset):
    return sum(STOPS[s][2] for s in subset)


def best_route(subset, windows=None, speed=1.0):
    """subset 안 모든 순서를 훑어 가장 짧은(분 기준) 실행 가능한 경로를 돌려준다. 없으면 (None, None)."""
    # 점 k개면 k! 가지 순서를 모두 본다(8곳 이하여서 가능)
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
        # (이른 도착, 늦은 도착 한도). 시간창이 없으면 사실상 무제한. 이르면 기다리고(max), 한도를 넘으면 불가능
        lo, hi = windows.get(s, (0, 10**9))
        t = max(t, lo)
        if t > hi:
            return False
        t += SERVICE   # 내려놓는 시간 5분은 다음 점으로 가기 전에 더해진다. 비용(총 이동거리)에는 포함하지 않는다
        p = s
    return True


def solve(max_vehicles=3, windows=None, speed=1.0, capacity=CAPACITY):
    """모든 정지점을 용량 이하의 경로 max_vehicles개 이하로 나눠 총 이동거리를 최소화한다(전수 탐색)."""
    # 점 집합을 비트마스크로 표현(비트 i = ids[i]번 점). 집합 분할 DP: 8점이면 마스크 255개
    n = len(STOPS)
    full = (1 << n) - 1
    ids = sorted(STOPS)
    cost = {}
    for mask in range(1, full + 1):
        sub = [ids[i] for i in range(n) if mask >> i & 1]
        # 용량을 넘는 묶음, 시간창으로 어떤 순서도 불가능한 묶음은 cost에 넣지 않는다(후보에서 제외)
        if demand(sub) > capacity:
            continue
        c, o = best_route(sub, windows, speed)
        if c is not None:
            cost[mask] = (c, o)

    # go(rest, k): 남은 점 rest를 차량 k대 이하로 덮는 최소 거리. 반환 (거리, 경로들)
    @lru_cache(None)
    def go(rest, k):
        if rest == 0:
            return (0.0, ())
        if k == 0:
            return (float("inf"), ())
        low = rest & -rest  # 남은 점 중 가장 낮은 번호를 반드시 포함해 중복 분할을 막는다
        best = (float("inf"), ())
        # (sub - 1) & rest 로 rest의 모든 부분집합을 큰 쪽부터 훑는다
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
    # inf = 차량 수나 용량·시간창 때문에 모든 점을 덮을 수 없다 -> (None, None)
    return (None, None) if total == float("inf") else (total, list(routes))


def one_big_route_cost():
    """용량 무시하고 한 대가 전부 도는 최단 경로(하한 비교용). 8! = 40,320가지 순서를 훑는다."""
    return best_route(sorted(STOPS))


def sweep_routes(capacity=CAPACITY):
    """번호 순서대로 싣다가 용량이 차면 새 차를 쓰는 단순 규칙."""
    # 번호 순서로 싣다 다음 점이 용량을 넘기면 새 차: 위치를 보지 않아 경로가 비효율적(78 대 최적 76)
    routes, cur, load = [], [], 0
    for s in sorted(STOPS):
        if load + STOPS[s][2] > capacity:
            routes.append(tuple(cur)); cur, load = [], 0
        cur.append(s); load += STOPS[s][2]
    routes.append(tuple(cur))
    return sum(route_len(r) for r in routes), routes


def nearest_neighbor(capacity=CAPACITY):
    # 현재 위치에서 가장 가까운 점을 계속 고른다. 같은 거리면 번호가 작은 쪽(결정적). 탐욕이라 최적 보장 없음(82)
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
    # 시간창·속도와 무관하게 용량만으로 센 값(README의 85개, 412가지)
    from itertools import combinations
    from math import factorial
    n_sets = n_orders = 0
    for k in range(1, len(STOPS) + 1):
        for sub in combinations(sorted(STOPS), k):
            if demand(sub) <= CAPACITY:
                n_sets += 1
                n_orders += factorial(k)
    return n_sets, n_orders


# 시간창은 STOPS 안의 None 자리가 아니라 이 딕셔너리로 따로 넘긴다. 형식 {점: (이른, 늦은)}
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
