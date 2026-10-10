"""알고리즘 6장 검증. `uv run pytest -q algorithms/ch06_graph_traversal` 로 실행한다.

설계도는 Hypothesis 성질 테스트를 적었지만 의존성을 늘리지 않으려고, 시드를 고정한 무작위 그래프를 반복문으로 만들어
Floyd-Warshall(간선 길이 1), 전수 탐색과 나란히 돌려 같은 성질을 확인한다.
"""

import random
import sys

import numpy as np
import pytest

import algorithms_ch06_graph as G


def _random_graph(rng, directed=False, nmax=25):
    n = rng.randint(1, nmax)
    g = G.Graph(n, directed=directed)
    for _ in range(rng.randint(0, 3 * n)):
        g.add_edge(rng.randrange(n), rng.randrange(n))
    return g


def test_first_screen_numbers():
    g = G.diamond_chain(3)
    assert (g.n, g.m) == (10, 12)
    r = G.dfs_no_visited(g, 0, limit=10 ** 6)
    assert r["visits"] == 29 == 1 + (1 + 1 + 2) + (2 + 2 + 4) + (4 + 4 + 8) and not r["capped"]
    c = G.Counter()
    G.dfs_iterative(g, 0, c)
    assert (c.visits, c.edge_checks) == (10, 12)
    assert G.trace_no_visited(g, 0)["steps"][-1][1] == [1, 1, 1, 2, 2, 2, 4, 4, 4, 8]
    assert G.diamond_no_visited_formula(20) == 4_194_301
    kg = G.konigsberg()
    assert G.degree_parity(kg) == [5, 3, 3, 3] and G.euler_check(kg)[0] == "none"


@pytest.mark.parametrize("k", range(1, 15))
def test_diamond_without_marks_matches_path_count_formula(k):
    g = G.diamond_chain(k)
    r = G.dfs_no_visited(g, 0, limit=10 ** 7)
    assert r["visits"] == 2 ** (k + 2) - 3 and r["edge_checks"] == 2 ** (k + 2) - 4
    for fn in (G.dfs_iterative, G.bfs):
        c = G.Counter()
        fn(g, 0, c)
        assert c.visits == 3 * k + 1 and c.edge_checks == 4 * k


def test_no_marks_never_ends_on_a_cycle():
    for g in (G.cycle_graph(3, directed=True), G.path_graph(2), G.grid_graph(4, 4)):
        r = G.dfs_no_visited(g, 0, limit=5000)
        assert r["capped"] and r["visits"] == 5000


def test_bfs_matches_floyd_warshall_and_visits_once():
    for t in range(80):
        rng = random.Random(t)
        g = _random_graph(rng, directed=bool(t % 2))
        s = rng.randrange(g.n)
        c = G.Counter()
        dist, parent = G.bfs(g, s, c)
        D = G.floyd_warshall_np(g)
        mine = np.array([np.inf if d is None else d for d in dist])
        assert np.array_equal(mine, D[s])
        reach = [v for v in range(g.n) if dist[v] is not None]
        assert c.visits == len(reach)  # 닿는 정점은 정확히 한 번
        assert c.edge_checks == sum(len(g.adj[v]) for v in reach)
        assert all(ok for _, ok, _ in G.check_bfs_distances(g, dist))
        for v in reach:  # 부모를 따라가면 거리만큼 걸어 출발점에 닿는다
            steps, x = 0, v
            while x != s:
                x = parent[x]
                steps += 1
            assert steps == dist[v]


def test_dfs_iterative_matches_recursive_and_reaches_same_set():
    for t in range(80):
        rng = random.Random(1000 + t)
        g = _random_graph(rng, directed=bool(t % 2))
        s = rng.randrange(g.n)
        order, parent, finish = G.dfs_iterative(g, s)
        assert order == G.dfs_recursive(g, s)
        dist, _ = G.bfs(g, s)
        assert sorted(order) == [v for v in range(g.n) if dist[v] is not None]
        assert sorted(finish) == sorted(order)
        assert finish[-1] == s


def test_trace_invariants_hold_every_step():
    for t in range(40):
        rng = random.Random(2000 + t)
        g = _random_graph(rng, nmax=12)
        s = rng.randrange(g.n)
        for tr, has_queue in ((G.trace_bfs(g, s), True), (G.trace_dfs(g, s), False)):
            for st in tr["steps"]:
                assert st[8] is True  # 검정 정점의 이웃은 흰색이 아니다
                if has_queue:
                    assert st[7] is True  # 큐의 거리는 비감소, 많아야 두 종류
            assert tr["steps"][-1][1].count("g") == 0  # 끝나면 회색이 남지 않는다


def test_euler_check_matches_bruteforce_and_hierholzer_builds_it():
    for t in range(200):
        rng = random.Random(3000 + t)
        n = rng.randint(1, 5)
        g = G.Graph(n)
        for _ in range(rng.randint(1, 7)):
            g.add_edge(rng.randrange(n), rng.randrange(n))
        kind, odd = G.euler_check(g)
        assert kind == G.euler_bruteforce(g)[0]
        if kind != "none":
            vs, es = G.hierholzer(g)
            assert sorted(es) == list(range(g.m))
            for i, e in enumerate(es):  # 이어진 길인가
                assert {vs[i], vs[i + 1]} == set(g.edges[e])
            assert (vs[0] == vs[-1]) == (kind == "circuit")


def test_euler_e53_examples():
    kg = G.konigsberg()
    assert G.euler_check(kg) == ("none", [0, 1, 2, 3])
    g, names, bridges, lands, eids = G.euler_15_bridges()
    assert g.m == 15 and len(set(bridges)) == 15
    deg = dict(zip(names, G.degree_parity(g)))
    assert deg == {"A": 8, "B": 4, "C": 4, "D": 3, "E": 5, "F": 6}
    kind, odd = G.euler_check(g)
    assert kind == "path" and [names[v] for v in odd] == ["D", "E"]
    assert names[lands[0]] == "E" and names[lands[-1]] == "D"


def test_recursive_dfs_hits_recursion_limit_on_long_path():
    g = G.path_graph(sys.getrecursionlimit() + 500)
    with pytest.raises(RecursionError):
        G.dfs_recursive(g, 0)
    c = G.Counter()
    G.dfs_iterative(g, 0, c)
    assert c.visits == g.n and c.max_frontier == g.n


def test_frontier_shapes():
    cb, cd = G.Counter(), G.Counter()
    G.bfs(G.star_graph(100), 0, cb)
    G.dfs_iterative(G.star_graph(100), 0, cd)
    assert cb.max_frontier == 99 and cd.max_frontier == 2
    cb, cd = G.Counter(), G.Counter()
    G.bfs(G.path_graph(100), 0, cb)
    G.dfs_iterative(G.path_graph(100), 0, cd)
    assert cb.max_frontier == 1 and cd.max_frontier == 100


def test_bfs_all_is_linear_and_levels_np_agree():
    g = G.erdos_renyi(3000, 3 / 2999, random.Random(1))
    c = G.Counter()
    dist, comp = G.bfs_all(g, c)
    assert c.visits == g.n and c.edge_checks == 2 * g.m
    f = G.Counter()
    comps = G.bfs_all_fresh_marks(g, f)
    assert comps == max(comp) + 1 and f.edge_checks == 2 * g.m
    d0, _ = G.bfs(g, 0)
    assert np.array_equal(np.array([-1 if d is None else d for d in d0]), G.bfs_levels_np(g, 0))
