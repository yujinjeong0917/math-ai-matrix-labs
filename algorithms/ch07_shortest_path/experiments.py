"""알고리즘 7장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch07_shortest_path/experiments.py` 로 실행한다.

E0 첫 화면 손계산: (a) 간선 개수로 재는 BFS 가 틀리는 3정점 그래프, (b) 음수 간선 하나로 다익스트라가 틀리는 3정점 그래프
E1 BFS(간선 개수) vs 실제 최단 거리: 정점 50, 간선 200, 길이 1~20, 시드 100개
E2 실패 최소 예제 1(음수 간선): 음수 간선 비율 0, 5, 10, 20% (음수 순환 없음, 거절 표집), 시드 100개.
   다익스트라 답이 Bellman-Ford 와 다른 정점 비율(출발점에서 닿는 정점, 출발점 제외), 다시 넣기 변형의 답과 꺼낸 횟수
E3 비교 측정: 음수 없는 같은 그래프에서 다익스트라·Bellman-Ford 의 이완 횟수와 시간, 그리고 크기를 키운 그래프
E4 음수 순환 감지: 닿는 쪽에 음수 순환을 넣은 그래프 50개
E5 실패 최소 예제 2(과대 휴리스틱): 30x30 격자, 장애물 25%, 시드 100개. h=0(다익스트라), 맨해튼, 맨해튼 2배
E6 손으로 볼 수 있는 4x5 격자에서 맨해튼 2배가 틀리는 경우(위젯용)
E7 위젯 트레이스: 음수 없는 5정점 그래프와 음수 간선 3정점 그래프의 다익스트라, E6 격자의 A* 두 가지
E8 흔한 실패: (i) 모든 간선에 같은 수를 더해 음수를 없애기, (ii) 다익스트라를 목표에 처음 거리가 적힐 때 멈추기
E9 NumPy 대조: Floyd-Warshall(NumPy) = Bellman-Ford, 음수 간선 포함 작은 그래프

결과는 results/ch07.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
이완·꺼내기·확장 횟수는 입력과 시드가 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

import heapq
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch07_shortest_path as A  # noqa: E402

OUT = HERE / "results" / "ch07.json"
SEEDS = 100
N, M = 50, 200


def _ms(fn, reps=3):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return out, statistics.median(ts)


def r4(x):
    return round(float(x), 4)


# ---------------------------------------------------------------- E0
def e0_first_screen():
    # (a) s=0, a=1, b=2 : s->a 10, s->b 1, b->a 2
    ga = A.from_edges(3, [(0, 1, 10), (0, 2, 1), (2, 1, 2)])
    hops, cost = A.bfs_hops(ga, 0)
    true_a = A.dijkstra(ga, 0)[0]
    # (b) s->a 1, s->b 2, b->a -2
    gb = A.from_edges(3, [(0, 1, 1), (0, 2, 2), (2, 1, -2)])
    dj = A.dijkstra(gb, 0)[0]
    bf = A.bellman_ford(gb, 0)[0]
    return {
        "bfs_example": {"edges": "s->a 10, s->b 1, b->a 2", "bfs_hops_a": hops[1], "bfs_cost_a": cost[1],
                        "true_a": true_a[1]},
        "neg_example": {"edges": "s->a 1, s->b 2, b->a -2", "dijkstra": dj, "bellman_ford": bf},
    }


# ---------------------------------------------------------------- E1
def e1_bfs_vs_weighted():
    wrong = tot = graphs_wrong = 0
    excess = []
    for seed in range(SEEDS):
        rng = A.seeded(seed)
        g = A.from_edges(N, A.random_digraph(N, M, rng))
        _, bc = A.bfs_hops(g, 0)
        d = A.dijkstra(g, 0)[0]
        w = 0
        for v in range(1, N):
            if d[v] < A.INF:
                tot += 1
                if bc[v] > d[v]:
                    w += 1
                    excess.append(bc[v] / d[v] - 1)
        wrong += w
        graphs_wrong += w > 0
    return {"seeds": SEEDS, "n": N, "m": M, "weights": "1..20", "reachable_vertices": tot, "wrong_vertices": wrong,
            "wrong_ratio": r4(wrong / tot), "graphs_with_wrong": graphs_wrong,
            "mean_excess_ratio_when_wrong": r4(statistics.mean(excess)), "max_excess_ratio": r4(max(excess))}


# ---------------------------------------------------------------- E2
def e2_negative_edges():
    rows = []
    for ratio in (0.0, 0.05, 0.10, 0.20):
        rej = wrong = tot = graphs_wrong = 0
        errs = []
        re_ok = 0
        fw_ok = 0
        pops_dj, pops_re, relax_dj, relax_re = [], [], [], []
        neg_edges = 0
        for seed in range(SEEDS):
            rng = A.seeded(1000 + seed)
            edges, r = A.random_negative_graph(N, M, ratio, rng)
            rej += r
            neg_edges += sum(1 for e in edges if e[2] < 0)
            g = A.from_edges(N, edges)
            bf = A.bellman_ford(g, 0)[0]
            D = floyd_warshall_np(N, edges)
            fw_ok += all((bf[v] == A.INF and np.isinf(D[0, v])) or bf[v] == D[0, v] for v in range(N))
            c1, c2 = A.Counter(), A.Counter()
            dj = A.dijkstra(g, 0, c1)[0]
            re = A.dijkstra_reinsert(g, 0, c2)[0]
            reach = A.reachable(g, 0)
            w = 0
            for v in range(1, N):
                if reach[v]:
                    tot += 1
                    if dj[v] != bf[v]:
                        w += 1
                        errs.append(dj[v] - bf[v])
            wrong += w
            graphs_wrong += w > 0
            re_ok += re == bf
            pops_dj.append(c1.pop)
            pops_re.append(c2.pop)
            relax_dj.append(c1.relax)
            relax_re.append(c2.relax)
        rows.append({
            "neg_ratio_target": ratio, "neg_edges_per_graph": neg_edges / SEEDS,
            "rejected_draws": rej, "reachable_vertices": tot, "wrong_vertices": wrong,
            "wrong_ratio": r4(wrong / tot), "graphs_with_wrong": graphs_wrong,
            "mean_error_when_wrong": r4(statistics.mean(errs)) if errs else 0,
            "max_error": max(errs) if errs else 0,
            "reinsert_correct_graphs": re_ok, "bf_matches_floyd_graphs": fw_ok,
            "pops_dijkstra_mean": r4(statistics.mean(pops_dj)), "pops_reinsert_mean": r4(statistics.mean(pops_re)),
            "pops_reinsert_max": max(pops_re),
            "relax_dijkstra_mean": r4(statistics.mean(relax_dj)), "relax_reinsert_mean": r4(statistics.mean(relax_re)),
        })
    return {"seeds": SEEDS, "n": N, "m": M, "pos_weights": "1..20", "neg_weights": "-5..-1",
            "method": "rejection sampling: draw graph, pick exact count of edges to negate, reject if any negative cycle",
            "rows": rows}


# ---------------------------------------------------------------- E3
def e3_compare():
    dj_relax, bf_relax, bf_rounds, dj_ms, bf_ms = [], [], [], [], []
    for seed in range(SEEDS):
        rng = A.seeded(seed)
        g = A.from_edges(N, A.random_digraph(N, M, rng))
        c1, c2 = A.Counter(), A.Counter()
        d1 = A.dijkstra(g, 0, c1)[0]
        d2 = A.bellman_ford(g, 0, c2)[0]
        assert d1 == d2
        dj_relax.append(c1.relax)
        bf_relax.append(c2.relax)
        bf_rounds.append(c2.rounds)
        dj_ms.append(_ms(lambda: A.dijkstra(g, 0), 5)[1])
        bf_ms.append(_ms(lambda: A.bellman_ford(g, 0), 5)[1])
    small = {
        "seeds": SEEDS, "n": N, "m": M,
        "dijkstra_relax_mean": r4(statistics.mean(dj_relax)),
        "bf_relax_mean": r4(statistics.mean(bf_relax)), "bf_rounds_mean": r4(statistics.mean(bf_rounds)),
        "bf_rounds_min": min(bf_rounds), "bf_rounds_max": max(bf_rounds),
        "bf_worst_bound_relax": (N - 1) * M + M,
        "dijkstra_ms_median": r4(statistics.median(dj_ms)), "bf_ms_median": r4(statistics.median(bf_ms)),
    }
    scale = []
    for n in (1000, 10000, 50000):
        rng = A.seeded(7)
        m = 4 * n
        g = A.from_edges(n, A.random_digraph(n, m, rng))
        c1, c2 = A.Counter(), A.Counter()
        d1 = A.dijkstra(g, 0, c1)[0]
        d2 = A.bellman_ford(g, 0, c2)[0]
        assert d1 == d2
        reps = 3 if n <= 10000 else 1
        _, t1 = _ms(lambda: A.dijkstra(g, 0), reps)
        _, t2 = _ms(lambda: A.bellman_ford(g, 0), reps)
        scale.append({"n": n, "m": m, "reachable": sum(A.reachable(g, 0)),
                      "dijkstra_relax": c1.relax, "dijkstra_pops": c1.pop,
                      "bf_relax": c2.relax, "bf_rounds": c2.rounds,
                      "dijkstra_ms": r4(t1), "bf_ms": r4(t2), "reps": reps})
    return {"small": small, "scale": scale}


# ---------------------------------------------------------------- E4
def e4_negative_cycle():
    detected = 0
    rounds = []
    for seed in range(50):
        rng = A.seeded(5000 + seed)
        edges = A.random_digraph(N, M, rng)
        g = A.from_edges(N, edges)
        reach = [v for v in range(N) if A.reachable(g, 0)[v] and v != 0]
        a, b, c = rng.sample(reach, 3)
        # a->b 1, b->c 1, c->a -5 : 순환 합 -3
        g.add(a, b, 1)
        g.add(b, c, 1)
        g.add(c, a, -5)
        cnt = A.Counter()
        try:
            A.bellman_ford(g, 0, cnt)
        except A.NegativeCycle:
            detected += 1
        rounds.append(cnt.rounds)
    return {"graphs": 50, "detected": detected, "rounds_all": sorted(set(rounds)), "cycle": "a->b 1, b->c 1, c->a -5"}


# ---------------------------------------------------------------- E5
def e5_grid():
    R = C = 30
    s, t = (0, 0), (R - 1, C - 1)
    used = skipped = 0
    rec = {"h0": [], "man": [], "man2": []}
    cost2, opt_list, excess = [], [], []
    sub = 0
    for seed in range(SEEDS):
        rng = A.seeded(seed)
        grid = A.random_grid(R, C, 0.25, rng, keep=[s, t])
        opt = A.grid_dijkstra_cost(grid, s, t)
        if opt is None:
            skipped += 1
            continue
        used += 1
        for key, h in (("h0", A.zero), ("man", A.manhattan), ("man2", A.scaled(A.manhattan, 2))):
            cnt = A.Counter()
            cost, path, order = A.a_star(grid, s, t, h, cnt)
            rec[key].append(cnt.expand)
            if key in ("h0", "man"):
                assert cost == opt
            else:
                cost2.append(cost)
                opt_list.append(opt)
                if cost > opt:
                    sub += 1
                    excess.append(cost - opt)
    free_cells = None
    out = {"rows": R, "cols": C, "block_p": 0.25, "seeds": SEEDS, "used": used, "skipped_no_path": skipped,
           "tie_break": "(f, h, insertion order); closed cells reopened when g improves; stop when goal is popped",
           "expand_mean": {k: r4(statistics.mean(v)) for k, v in rec.items()},
           "expand_median": {k: statistics.median(v) for k, v in rec.items()},
           "man2_suboptimal": sub, "man2_excess_mean": r4(statistics.mean(excess)) if excess else 0,
           "man2_excess_max": max(excess) if excess else 0,
           "man2_excess_ratio_mean": r4(statistics.mean([(c - o) / o for c, o in zip(cost2, opt_list) if c > o])) if excess else 0,
           "opt_cost_mean": r4(statistics.mean(opt_list)),
           "excess_hist": {str(k): excess.count(k) for k in sorted(set(excess))}}
    del free_cells
    return out


# ---------------------------------------------------------------- E6/E7 위젯
SMALL_GRID = ["S..##", "....#", ".#...", "...#T"]


def e6_small_grid():
    grid, s, t = A.grid_from_strings(SMALL_GRID)
    out = {"grid": SMALL_GRID, "opt": A.grid_dijkstra_cost(grid, s, t)}
    for key, h in (("h0", A.zero), ("man", A.manhattan), ("man2", A.scaled(A.manhattan, 2))):
        cnt = A.Counter()
        tr = []
        cost, path, order = A.a_star(grid, s, t, h, cnt, trace=tr)
        out[key] = {"cost": cost, "path": [list(p) for p in path], "expanded": cnt.expand,
                    "steps": [[list(u), gg, hh] for u, gg, hh in tr]}
    return out


def dijkstra_widget(n, edges, pos, names):
    g = A.from_edges(n, edges)
    true = A.bellman_ford(g, 0)[0]
    tr = []
    A.dijkstra(g, 0, trace=tr)
    steps = []
    for kind, u, e, dist, done, heap in tr:
        inv1, inv2 = A.check_settled_invariants(dist, done, heap, true)
        d = [None if x == A.INF else x for x in dist]
        steps.append([kind, u, list(e) if e else None, d, [1 if x else 0 for x in done],
                      [[None if x == A.INF else x, v] for x, v in heap if x == dist[v] and not done[v]], inv1, inv2])
    return {"n": n, "edges": [list(x) for x in edges], "pos": pos, "names": names,
            "true": [None if x == A.INF else x for x in true], "steps": steps}


def e7_widget():
    pos_graph = dijkstra_widget(
        5, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5), (3, 4, 3), (1, 4, 6)],
        [[60, 140], [250, 50], [250, 230], [420, 140], [560, 140]], ["s", "a", "b", "c", "t"])
    neg_graph = dijkstra_widget(
        3, [(0, 1, 1), (0, 2, 2), (2, 1, -2)],
        [[90, 150], [300, 60], [500, 200]], ["s", "a", "b"])
    return {"pos": pos_graph, "neg": neg_graph}


# ---------------------------------------------------------------- E8 흔한 실패
def e8_common_failures():
    # (i) 모든 간선에 |최소 음수| 를 더해 다익스트라
    wrong = tot = graphs_wrong = 0
    for seed in range(SEEDS):
        rng = A.seeded(1000 + seed)
        edges, _ = A.random_negative_graph(N, M, 0.10, rng)
        g = A.from_edges(N, edges)
        bf = A.bellman_ford(g, 0)[0]
        shift = -min(w for _, _, w in edges)
        g2 = A.from_edges(N, [(u, v, w + shift) for u, v, w in edges])
        _, par = A.dijkstra(g2, 0)
        reach = A.reachable(g, 0)
        w_ = 0
        for v in range(1, N):
            if not reach[v]:
                continue
            tot += 1
            path = [v]
            while par[path[-1]] is not None:
                path.append(par[path[-1]])
            path = path[::-1]
            if A.path_cost(g, path) != bf[v]:
                w_ += 1
        wrong += w_
        graphs_wrong += w_ > 0
    # 손계산 예: s->a 1, s->b 2, b->a -2 에 2를 더하면 s->a 3, s->b 4, b->a 0 → 다익스트라 경로 s->a (원래 길이 1, 참 0)
    shift_ex = A.from_edges(3, [(0, 1, 3), (0, 2, 4), (2, 1, 0)])
    _, par_ex = A.dijkstra(shift_ex, 0)
    shift_res = {"seeds": SEEDS, "neg_ratio": 0.10, "reachable_vertices": tot, "wrong_vertices": wrong,
                 "wrong_ratio": r4(wrong / tot), "graphs_with_wrong": graphs_wrong,
                 "tiny_example_parent_of_a": par_ex[1]}
    # (ii) 다익스트라를 목표에 처음 라벨이 붙을 때 멈추기(꺼낼 때가 아니라)
    early_wrong = early_tot = 0
    for seed in range(SEEDS):
        rng = A.seeded(seed)
        g = A.from_edges(N, A.random_digraph(N, M, rng))
        d = A.dijkstra(g, 0)[0]
        for t in range(1, N):
            if d[t] == A.INF:
                continue
            early_tot += 1
            early_wrong += dijkstra_first_label(g, 0, t) > d[t]
    return {"shift_all_edges": shift_res,
            "stop_on_first_label": {"seeds": SEEDS, "targets": early_tot, "wrong": early_wrong,
                                    "wrong_ratio": r4(early_wrong / early_tot)}}


def dijkstra_first_label(g, s, t):
    """흔한 실패: 목표 t 에 처음 거리가 적히는 순간 그 값을 답으로 낸다."""
    dist = [A.INF] * g.n
    done = [False] * g.n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in g.adj[u]:
            if not done[v] and d + w < dist[v]:
                first = dist[v] == A.INF
                dist[v] = d + w
                if v == t and first:
                    return dist[v]
                heapq.heappush(heap, (dist[v], v))
    return dist[t]


# ---------------------------------------------------------------- E9 NumPy
def floyd_warshall_np(n, edges):
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    for u, v, w in edges:
        D[u, v] = min(D[u, v], w)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def e9_numpy():
    agree = 0
    total = 0
    for seed in range(50):
        rng = A.seeded(9000 + seed)
        edges, _ = A.random_negative_graph(30, 100, 0.10, rng)
        g = A.from_edges(30, edges)
        D = floyd_warshall_np(30, edges)
        for s in range(30):
            bf = A.bellman_ford(g, s)[0]
            total += 1
            agree += all((bf[v] == A.INF and np.isinf(D[s, v])) or bf[v] == D[s, v] for v in range(30))
    return {"graphs": 50, "sources_checked": total, "agree": agree}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "omp_threads": os.environ.get("OMP_NUM_THREADS")},
        "E0_first_screen": e0_first_screen(),
        "E1_bfs_vs_weighted": e1_bfs_vs_weighted(),
        "E2_negative_edges": e2_negative_edges(),
        "E3_compare": e3_compare(),
        "E4_negative_cycle": e4_negative_cycle(),
        "E5_grid": e5_grid(),
        "E6_small_grid": e6_small_grid(),
        "E7_widget": e7_widget(),
        "E8_common_failures": e8_common_failures(),
        "E9_numpy": e9_numpy(),
    }
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("E6_small_grid", "E7_widget")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
