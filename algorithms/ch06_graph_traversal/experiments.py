"""알고리즘 6장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch06_graph_traversal/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 다이아몬드 3개 사슬(정점 10, 간선 12)에서 표시 없는 DFS 와 표시 있는 DFS, 그리고 Königsberg 차수
E1 실패 최소 예제 1: 다이아몬드 사슬 k = 5, 10, 15, 20 에서 방문 표시 유무의 방문 횟수·간선 검사·시간, 식 2^(k+2)-3 대조
E2 실패 최소 예제 2: 순환이 있는 그래프(방향 3-순환, 무방향 격자, G(n,p) 시드 10개)에서 표시 없는 탐색은 끝나지 않는다.
   방문 상한 10^6 에서 끊고 그때까지 걸린 시간을 적는다. 표시가 있으면 방문 = 닿는 정점 수
E3 BFS 와 DFS 의 프런티어 최대 크기(메모리): 격자, G(n,p), 경로, 별, 완전 이진 트리
E4 O(V+E): G(n, 평균 차수 4)에서 n = 10^3 .. 2x10^5, 간선 검사 = 2E 와 시간/(V+E)
E5 오일러: Königsberg(불가능), Euler E53 §15 의 다리 15개(D 또는 E 에서 출발하면 가능), Hierholzer 로 길 만들기,
   작은 무작위 다중그래프 300개에서 차수 판정 = 전수 탐색, 그리고 전수 탐색이 치른 걸음 수
E6 재귀 DFS 는 긴 경로에서 RecursionError: 몇 번째 정점에서 나는가
E7 NumPy 대조: BFS 거리 = Floyd-Warshall(간선 길이 1), BFS = 불리언 층 BFS
E8 위젯 트레이스: 정점 8개 그래프의 BFS·DFS, 다이아몬드 3개 사슬의 표시 없는 DFS
E9 흔한 실패: 큐에서 꺼낼 때 표시하는 BFS(거리는 같고 큐에 넣는 횟수가 늘어난다)

결과는 results/ch06.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
방문 횟수·간선 검사 수·프런티어 크기는 입력과 시드가 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

# NumPy 가 쓰는 스레드 수를 2개로 고정해 시간 측정의 흔들림을 줄인다(NumPy import 전에 설정해야 효과가 있다).
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch06_graph as G  # noqa: E402

OUT = HERE / "results" / "ch06.json"
# 표시 없는 탐색을 끊는 방문 상한. 이 값에 닿으면 '끝나지 않음'으로 본다.
CAP = 10 ** 6
# 무작위 그래프 반복 수. 시드는 0..9 처럼 고정해 같은 숫자가 다시 나오게 한다.
SEEDS = 10


def _ms(fn, reps=1):
    # 반복 측정의 중앙값을 쓴다(평균보다 일시적 부하에 덜 흔들림). 단위 ms.
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return out, statistics.median(ts)


def e0_first_screen():
    k = 3
    g = G.diamond_chain(k)
    nv = G.dfs_no_visited(g, 0, limit=10 ** 7)
    c = G.Counter()
    order, _, _ = G.dfs_iterative(g, 0, c)
    # 정점별 도착 횟수(표시 없을 때) = 시작에서 그 정점까지 길 수
    counts = G.trace_no_visited(g, 0, limit=10 ** 4)["steps"][-1][1]
    kg = G.konigsberg()
    return {"k": k, "V": g.n, "E": g.m, "paths_to_end": 2 ** k,
            "no_visited": nv, "formula": G.diamond_no_visited_formula(k),
            "visited": {"visits": c.visits, "edge_checks": c.edge_checks},
            "arrivals_per_vertex": counts,
            "k20": {"V": 3 * 20 + 1, "E": 80, "no_visited_formula": G.diamond_no_visited_formula(20)},
            "konigsberg": {"names": G.KONIG_NAMES, "degrees": G.degree_parity(kg), "bridges": 7,
                           "verdict": G.euler_check(kg)[0]}}


def e1_diamond():
    rows = []
    for k in (5, 10, 15, 20):
        g = G.diamond_chain(k)
        reps = 1 if k == 20 else 3
        nv, ms_nv = _ms(lambda: G.dfs_no_visited(g, 0, limit=10 ** 8), reps)
        c = G.Counter()
        # 같은 Counter c 를 5번 재사용하므로 c.visits 는 5회 누적값. 아래에서 // 5 로 1회분을 구한다.
        _, ms_v = _ms(lambda: G.dfs_iterative(g, 0, c), 5)
        cb = G.Counter()
        G.bfs(g, 0, cb)
        # k=20 은 표시 없는 탐색이 약 4x10^6 번 도착이라 반복을 1번만 한다.
        rows.append({"k": k, "V": g.n, "E": g.m, "paths": 2 ** k,
                     "no_visited_visits": nv["visits"], "no_visited_edge_checks": nv["edge_checks"],
                     "no_visited_capped": nv["capped"], "formula": G.diamond_no_visited_formula(k),
                     "visited_visits": c.visits // 5, "visited_edge_checks": c.edge_checks // 5,
                     "bfs_visits": cb.visits, "bfs_edge_checks": cb.edge_checks,
                     "ms_no_visited": ms_nv, "ms_visited": ms_v,
                     "ratio_visits": nv["visits"] / (c.visits // 5), "ratio_ms": ms_nv / ms_v})
    return {"rows": rows}


def e2_cycles():
    out = {"cap": CAP}
    # 순환 하나만 있어도 표시 없는 DFS 는 상한까지 돈다.
    # 방향 3-순환
    g = G.cycle_graph(3, directed=True)
    r, ms = _ms(lambda: G.dfs_no_visited(g, 0, limit=CAP))
    c = G.Counter()
    G.dfs_iterative(g, 0, c)
    out["cycle3"] = {"V": 3, "E": 3, "no_visited": r, "ms_to_cap": ms, "visited_visits": c.visits,
                     "visited_edge_checks": c.edge_checks}
    # 무방향 간선 하나(0-1): 표시가 없으면 0 -> 1 -> 0 -> ... 왕복
    g = G.path_graph(2)
    r, ms = _ms(lambda: G.dfs_no_visited(g, 0, limit=CAP))
    out["one_undirected_edge"] = {"V": 2, "E": 1, "no_visited": r, "ms_to_cap": ms}
    # 무방향 격자 30 x 30
    g = G.grid_graph(30, 30)
    r, ms = _ms(lambda: G.dfs_no_visited(g, 0, limit=CAP))
    c = G.Counter()
    _, msv = _ms(lambda: G.dfs_iterative(g, 0, c))
    out["grid30"] = {"V": g.n, "E": g.m, "no_visited": r, "ms_to_cap": ms, "visited_visits": c.visits,
                     "visited_edge_checks": c.edge_checks, "ms_visited": msv,
                     "distinct_reached_before_cap": None}
    # dfs_no_visited 안의 반복을 그대로 재현해 도착한 서로 다른 정점을 센다(그 함수는 이 정보를 돌려주지 않는다).
    # 끊기 전까지 서로 다른 정점을 몇 개 들렀나(표시 없는 탐색이 '어디까지 갔나')
    seen = set()
    stack, vis = [0], 0
    while stack and vis < CAP:
        u = stack.pop()
        vis += 1
        seen.add(u)
        for v, _ in reversed(g.adj[u]):
            stack.append(v)
    out["grid30"]["distinct_reached_before_cap"] = len(seen)
    # G(n, p), n = 1000, 평균 차수 4, 시드 10
    runs = []
    for s in range(SEEDS):
        g = G.erdos_renyi(1000, 4 / 999, random.Random(s))
        r, ms = _ms(lambda: G.dfs_no_visited(g, 0, limit=CAP))
        c = G.Counter()
        _, msv = _ms(lambda: G.dfs_iterative(g, 0, c))
        runs.append({"seed": s, "E": g.m, "reachable": G.reachable_count(g, 0), "capped": r["capped"],
                     "no_visited_visits": r["visits"], "ms_to_cap": ms, "visited_visits": c.visits,
                     "visited_edge_checks": c.edge_checks, "ms_visited": msv})
    out["gnp1000"] = {"runs": runs, "all_capped": all(x["capped"] or x["reachable"] == 1 for x in runs),
                      "ms_to_cap_median": statistics.median(x["ms_to_cap"] for x in runs),
                      "ms_visited_median": statistics.median(x["ms_visited"] for x in runs),
                      "reachable_min": min(x["reachable"] for x in runs),
                      "reachable_max": max(x["reachable"] for x in runs)}
    return out


def _frontiers(g, s):
    cb, cd = G.Counter(), G.Counter()
    dist, _ = G.bfs(g, s, cb)
    G.dfs_iterative(g, s, cd)
    # ecc(이심률) = 출발점에서 가장 먼 정점까지 거리. 층 수 = ecc + 1.
    ecc = max(d for d in dist if d is not None)
    return {"V": g.n, "E": g.m, "bfs_max_queue": cb.max_frontier, "dfs_max_stack": cd.max_frontier,
            "bfs_levels": ecc + 1, "reachable": sum(d is not None for d in dist)}


def e3_frontier():
    out = {}
    g = G.grid_graph(100, 100)
    out["grid100_corner"] = _frontiers(g, 0)
    # 격자 한가운데(행 50, 열 50)에서 시작하면 BFS 층이 사방으로 퍼져 큐가 더 커진다.
    out["grid100_center"] = _frontiers(g, 50 * 100 + 50)
    out["path10000"] = _frontiers(G.path_graph(10_000), 0)
    out["star10000"] = _frontiers(G.star_graph(10_000), 0)
    out["bintree16383"] = _frontiers(G.binary_tree(2 ** 14 - 1), 0)
    gnp = []
    for s in range(SEEDS):
        g = G.erdos_renyi(10_000, 4 / 9999, random.Random(100 + s))
        gnp.append(_frontiers(g, 0))
    out["gnp10000"] = {"runs": gnp,
                       "bfs_max_queue_mean": statistics.mean(x["bfs_max_queue"] for x in gnp),
                       "dfs_max_stack_mean": statistics.mean(x["dfs_max_stack"] for x in gnp),
                       "reachable_mean": statistics.mean(x["reachable"] for x in gnp)}
    return out


def e4_linear():
    rows = []
    for n in (1_000, 10_000, 50_000, 200_000):
        # 평균 차수 약 4: 기대 간선 수 p x n(n-1)/2 = 2n. 시드 7 고정.
        g = G.erdos_renyi(n, 4 / (n - 1), random.Random(7))
        cb = G.Counter()
        _, msb = _ms(lambda: G.bfs(g, 0, G.Counter()), 5)
        G.bfs(g, 0, cb)
        cd = G.Counter()
        _, msd = _ms(lambda: G.dfs_iterative(g, 0, G.Counter()), 5)
        G.dfs_iterative(g, 0, cd)
        # (무방향이라 간선마다 양 끝에서 한 번씩 본다)
        # 연결 요소가 여럿이라도 전체를 덮는 BFS: 간선 검사가 정확히 2E, 방문이 정확히 V
        total = G.Counter()
        (_, comp), ms_all = _ms(lambda: G.bfs_all(g, G.Counter()), 3)
        G.bfs_all(g, total)
        comps = max(comp) + 1
        # 흔한 실패: 요소마다 표시판을 새로 만들면
        fresh = G.Counter()
        _, ms_fresh = _ms(lambda: G.bfs_all_fresh_marks(g, fresh))
        rows.append({"n": n, "E": g.m, "reachable_from_0": cb.visits, "bfs_edge_checks_from_0": cb.edge_checks,
                     "dfs_edge_checks_from_0": cd.edge_checks, "ms_bfs_from_0": msb, "ms_dfs_from_0": msd,
                     "all_components": comps, "all_visits": total.visits, "all_edge_checks": total.edge_checks,
                     "two_E": 2 * g.m, "ms_all": ms_all, "ns_per_V_plus_E": ms_all * 1e6 / (g.n + g.m),
                     "fresh_marks_edge_checks": fresh.edge_checks, "ms_fresh_marks": ms_fresh,
                     "fresh_marks_array_cells": comps * g.n, "ratio_fresh_over_shared_ms": ms_fresh / ms_all})
    return {"rows": rows}


def e5_euler():
    out = {}
    kg = G.konigsberg()
    kind, odd = G.euler_check(kg)
    bf, steps = G.euler_bruteforce(kg)
    out["konigsberg"] = {"degrees": dict(zip(G.KONIG_NAMES, G.degree_parity(kg))), "odd": [G.KONIG_NAMES[v] for v in odd],
                         "verdict": kind, "bruteforce": bf, "bruteforce_steps": steps}
    g, names, bridges, lands, eids = G.euler_15_bridges()
    kind, odd = G.euler_check(g)
    vs, es = G.hierholzer(g)
    # E53 본문 길이 15개 다리를 빠짐없이 한 번씩 쓰는지 확인.
    route_ok = sorted(eids) == list(range(15)) and len(set(eids)) == 15
    out["euler15"] = {"lands": names, "bridges": 15, "degrees": dict(zip(names, G.degree_parity(g))),
                      "odd": [names[v] for v in odd], "verdict": kind,
                      "euler_route_valid": route_ok, "euler_route_start": names[lands[0]], "euler_route_end": names[lands[-1]],
                      "hierholzer_route": "".join(names[v] for v in vs), "hierholzer_edges": len(es),
                      "hierholzer_uses_each_once": sorted(es) == list(range(15))}
    # 전수 탐색이 지수적이라 정점 2~5, 간선 1~8 개로 제한한다.
    # 무작위 작은 다중그래프: 차수 판정 = 전수 탐색
    rng = random.Random(6)
    agree, total, kinds, max_steps = 0, 0, {"circuit": 0, "path": 0, "none": 0}, 0
    steps_by_m = {}
    for t in range(300):
        n = rng.randint(2, 5)
        m = rng.randint(1, 8)
        h = G.Graph(n)
        for _ in range(m):
            h.add_edge(rng.randrange(n), rng.randrange(n))
        a = G.euler_check(h)[0]
        b, st = G.euler_bruteforce(h)
        total += 1
        agree += a == b
        kinds[a] += 1
        max_steps = max(max_steps, st)
        steps_by_m.setdefault(m, []).append(st)
        if a != "none":
            vs, es = G.hierholzer(h)
            assert sorted(es) == list(range(h.m))
    out["random_small"] = {"graphs": total, "agree": agree, "kinds": kinds, "bruteforce_max_steps": max_steps,
                           "bruteforce_mean_steps_by_m": {str(m): statistics.mean(v) for m, v in sorted(steps_by_m.items())}}
    return out


def e6_recursion():
    limit = sys.getrecursionlimit()
    res = {"recursion_limit": limit}
    # 500 은 보통 한도(1000) 아래, 2000 은 위라 대비가 난다. 실제 한도는 res 의 recursion_limit 참고.
    for n in (500, 2000):
        g = G.path_graph(n)
        c = G.Counter()
        try:
            G.dfs_recursive(g, 0, c)
            res[f"path{n}"] = {"ok": True, "visits": c.visits}
        except RecursionError:
            res[f"path{n}"] = {"ok": False, "visits_before_error": c.visits, "depth_before_error": c.max_frontier}
        c2 = G.Counter()
        G.dfs_iterative(g, 0, c2)
        res[f"path{n}"]["iterative_visits"] = c2.visits
        res[f"path{n}"]["iterative_max_stack"] = c2.max_frontier
    return res


def e7_numpy():
    agree_fw, graphs = 0, 0
    for s in range(50):
        rng = random.Random(500 + s)
        n = rng.randint(5, 60)
        directed = s % 2 == 1
        g = G.Graph(n, directed=directed)
        for _ in range(rng.randint(0, 3 * n)):
            g.add_edge(rng.randrange(n), rng.randrange(n))
        # 정답 대조용 Floyd-Warshall. n 이 60 이하라 O(n^3) 도 순식간.
        D = G.floyd_warshall_np(g)
        src = rng.randrange(n)
        dist, _ = G.bfs(g, src)
        mine = np.array([np.inf if d is None else d for d in dist])
        graphs += 1
        agree_fw += bool(np.array_equal(mine, D[src]))
    g = G.erdos_renyi(20_000, 3 / 19_999, random.Random(9))
    dist, _ = G.bfs(g, 0)
    a = np.array([-1 if d is None else d for d in dist])
    b = G.bfs_levels_np(g, 0)
    return {"floyd_warshall_graphs": graphs, "floyd_warshall_agree": agree_fw,
            "levels_np_n": g.n, "levels_np_agree": bool(np.array_equal(a, b)), "levels": int(b.max()) + 1}


def e9_mark_on_pop():
    """흔한 실패: 꺼낼 때 표시하는 BFS. 거리는 같지만 큐에 넣는 횟수와 큐 크기가 늘어난다."""
    rows = []
    cases = [("grid30", G.grid_graph(30, 30)), ("grid100", G.grid_graph(100, 100)),
             ("complete200", G.erdos_renyi(200, 1.0, random.Random(0))),
             ("gnp10000", G.erdos_renyi(10_000, 4 / 9999, random.Random(100)))]
    for name, g in cases:
        c1, c2 = G.Counter(), G.Counter()
        d1, _ = G.bfs(g, 0, c1)
        d2, pushes = G.bfs_mark_on_pop(g, 0, c2)
        rows.append({"graph": name, "V": g.n, "E": g.m, "same_dist": d1 == d2, "pushes_mark_on_push": c1.visits,
                     "pushes_mark_on_pop": pushes, "max_queue_mark_on_push": c1.max_frontier,
                     "max_queue_mark_on_pop": c2.max_frontier})
    return {"rows": rows}


# 좌표는 화면용 픽셀(x, y)이고 알고리즘과 무관.
# 위젯 그래프: 정점 8개, 순환 2개. 좌표는 위젯이 그린다.
WIDGET_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 5), (4, 6), (5, 6), (6, 7)]
WIDGET_POS = [[60, 150], [170, 70], [170, 230], [290, 150], [300, 40], [410, 230], [430, 110], [540, 110]]


def e8_traces():
    g = G.Graph(8)
    for u, v in WIDGET_EDGES:
        g.add_edge(u, v)
    d = G.diamond_chain(3)
    return {"edges": WIDGET_EDGES, "pos": WIDGET_POS, "bfs": G.trace_bfs(g, 0), "dfs": G.trace_dfs(g, 0),
            "diamond": {"edges": d.edges, "no_visited": G.trace_no_visited(d, 0, limit=100),
                        "visited": G.trace_dfs(d, 0)}}


def main():
    t0 = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                   "platform": platform.platform(), "recursion_limit": sys.getrecursionlimit(),
                   # 아래 note 문자열은 결과 json 에 그대로 저장된다.
                   "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. NumPy 스레드 2개"}}
    for key, fn in (("E0_first_screen", e0_first_screen), ("E5_euler", e5_euler), ("E6_recursion", e6_recursion),
                    ("E7_numpy", e7_numpy), ("E8_widget", e8_traces), ("E9_mark_on_pop", e9_mark_on_pop), ("E3_frontier", e3_frontier),
                    ("E2_cycles", e2_cycles), ("E4_linear", e4_linear), ("E1_diamond", e1_diamond)):
        res[key] = fn()
        print(key, round(time.perf_counter() - t0, 1), flush=True)
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    show = {k: v for k, v in res.items() if k != "E8_widget"}
    print(json.dumps(show, ensure_ascii=False, indent=1)[:15000])


if __name__ == "__main__":
    main()
