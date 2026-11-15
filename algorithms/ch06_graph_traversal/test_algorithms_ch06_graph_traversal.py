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
    # 시드가 고정된 rng 를 받아 재현 가능한 무작위 그래프를 만든다. 간선 수 0..3n, 고리와 평행 간선도 나올 수 있다(일부러 허용).
    n = rng.randint(1, nmax)
    g = G.Graph(n, directed=directed)
    for _ in range(rng.randint(0, 3 * n)):
        g.add_edge(rng.randrange(n), rng.randrange(n))
    return g


def test_first_screen_numbers():
    g = G.diamond_chain(3)
    # 다이아몬드 3개: 정점 3k+1 = 10, 간선 4k = 12.
    assert (g.n, g.m) == (10, 12)
    r = G.dfs_no_visited(g, 0, limit=10 ** 6)
    # 표시 없는 DFS 의 도착 횟수 = 출발점에서 각 정점까지 길 수의 합. 마디 i 는 2^i 번, 위·아래 정점은 각 2^(i-1) 번 -> 2^(k+2)-3 = 29.
    assert r["visits"] == 29 == 1 + (1 + 1 + 2) + (2 + 2 + 4) + (4 + 4 + 8) and not r["capped"]
    c = G.Counter()
    G.dfs_iterative(g, 0, c)
    # 표시판이 있으면 정점 V=10 번, 방향 간선 E=12 번만 본다.
    assert (c.visits, c.edge_checks) == (10, 12)
    # 정점 번호 순서(0, 1, 2, 3, ...)로 본 도착 횟수. 마디 0,3,6,9 는 1,2,4,8, 사이 정점은 1,1 / 2,2 / 4,4.
    assert G.trace_no_visited(g, 0)["steps"][-1][1] == [1, 1, 1, 2, 2, 2, 4, 4, 4, 8]
    # 2^22 - 3 = 4194304 - 3.
    assert G.diamond_no_visited_formula(20) == 4_194_301
    kg = G.konigsberg()
    # Königsberg 차수 A5, B3, C3, D3 (합 14 = 2 x 다리 7개). 홀수 정점 4개라 오일러 길이 없다.
    assert G.degree_parity(kg) == [5, 3, 3, 3] and G.euler_check(kg)[0] == "none"


@pytest.mark.parametrize("k", range(1, 15))
def test_diamond_without_marks_matches_path_count_formula(k):
    g = G.diamond_chain(k)
    r = G.dfs_no_visited(g, 0, limit=10 ** 7)
    # edge_checks = visits - 1: 출발점을 뺀 모든 도착은 간선 하나를 본 결과다.
    assert r["visits"] == 2 ** (k + 2) - 3 and r["edge_checks"] == 2 ** (k + 2) - 4
    for fn in (G.dfs_iterative, G.bfs):
        c = G.Counter()
        fn(g, 0, c)
        # 표시판이 있는 BFS·DFS 는 도착 횟수가 V = 3k+1, 간선 검사가 E = 4k 로 선형.
        assert c.visits == 3 * k + 1 and c.edge_checks == 4 * k


def test_no_marks_never_ends_on_a_cycle():
    for g in (G.cycle_graph(3, directed=True), G.path_graph(2), G.grid_graph(4, 4)):
        # 순환이 있으면 표시 없는 탐색은 끝나지 않으니 limit 에서 끊긴 것이 정상 결과(capped). 무방향 간선은 양방향이라 간선 하나도 순환.
        r = G.dfs_no_visited(g, 0, limit=5000)
        assert r["capped"] and r["visits"] == 5000


def test_bfs_matches_floyd_warshall_and_visits_once():
    for t in range(80):
        rng = random.Random(t)
        g = _random_graph(rng, directed=bool(t % 2))
        s = rng.randrange(g.n)
        c = G.Counter()
        dist, parent = G.bfs(g, s, c)
        # Floyd-Warshall 을 독립된 정답으로 사용(간선 길이 1, 닿지 않으면 inf). BFS 거리와 같아야 한다.
        D = G.floyd_warshall_np(g)
        mine = np.array([np.inf if d is None else d for d in dist])
        # None 을 inf 로 바꿔 비교. 값이 정수이므로 허용 오차 없이 정확히 비교해도 된다.
        assert np.array_equal(mine, D[s])
        reach = [v for v in range(g.n) if dist[v] is not None]
        # 큐에서 꺼낸 횟수 = 닿는 정점 수.
        assert c.visits == len(reach)  # 닿는 정점은 정확히 한 번
        # 닿는 정점의 인접 리스트 길이 합(무방향은 2E, 고리는 2로 셈).
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
        # 반복판과 재귀판의 방문 순서가 같아야 한다(스택에 이웃 번호를 둔 설계의 확인).
        assert order == G.dfs_recursive(g, s)
        dist, _ = G.bfs(g, s)
        assert sorted(order) == [v for v in range(g.n) if dist[v] is not None]
        assert sorted(finish) == sorted(order)
        # 출발점은 가장 마지막에 검정이 된다.
        assert finish[-1] == s


def test_trace_invariants_hold_every_step():
    for t in range(40):
        rng = random.Random(2000 + t)
        g = _random_graph(rng, nmax=12)
        s = rng.randrange(g.n)
        for tr, has_queue in ((G.trace_bfs(g, s), True), (G.trace_dfs(g, s), False)):
            for st in tr["steps"]:
                # 단계마다 성립해야 하는 불변식(최종 상태만이 아님).
                assert st[8] is True  # 검정 정점의 이웃은 흰색이 아니다
                if has_queue:
                    assert st[7] is True  # 큐의 거리는 비감소, 많아야 두 종류
            # 색 문자열의 g 개수. 끝나면 모든 닿은 정점이 b.
            assert tr["steps"][-1][1].count("g") == 0  # 끝나면 회색이 남지 않는다


def test_euler_check_matches_bruteforce_and_hierholzer_builds_it():
    for t in range(200):
        rng = random.Random(3000 + t)
        n = rng.randint(1, 5)
        g = G.Graph(n)
        for _ in range(rng.randint(1, 7)):
            g.add_edge(rng.randrange(n), rng.randrange(n))
        kind, odd = G.euler_check(g)
        # 작은 다중그래프(정점 5, 간선 7 이하)에서 차수 규칙과 전수 탐색이 일치해야 한다.
        assert kind == G.euler_bruteforce(g)[0]
        if kind != "none":
            vs, es = G.hierholzer(g)
            # 모든 간선을 정확히 한 번씩.
            assert sorted(es) == list(range(g.m))
            # i 번째 간선이 경로의 i, i+1 번째 정점을 잇는지(집합 비교는 무방향이라 순서 무관).
            for i, e in enumerate(es):  # 이어진 길인가
                assert {vs[i], vs[i + 1]} == set(g.edges[e])
            assert (vs[0] == vs[-1]) == (kind == "circuit")


def test_euler_e53_examples():
    kg = G.konigsberg()
    assert G.euler_check(kg) == ("none", [0, 1, 2, 3])
    g, names, bridges, lands, eids = G.euler_15_bridges()
    assert g.m == 15 and len(set(bridges)) == 15
    deg = dict(zip(names, G.degree_parity(g)))
    # E53 §15 의 길 문자열에서 읽은 차수. 합 30 = 2 x 15.
    assert deg == {"A": 8, "B": 4, "C": 4, "D": 3, "E": 5, "F": 6}
    kind, odd = G.euler_check(g)
    # 홀수 정점이 D, E 둘뿐이라 회로는 아니고 경로. 본문 길도 E 에서 D 로 끝난다.
    assert kind == "path" and [names[v] for v in odd] == ["D", "E"]
    assert names[lands[0]] == "E" and names[lands[-1]] == "D"


def test_recursive_dfs_hits_recursion_limit_on_long_path():
    # 재귀 한도보다 긴 경로를 만들어 재귀판이 반드시 깊이 초과하게 한다.
    g = G.path_graph(sys.getrecursionlimit() + 500)
    with pytest.raises(RecursionError):
        G.dfs_recursive(g, 0)
    c = G.Counter()
    G.dfs_iterative(g, 0, c)
    # 경로에서 DFS 스택은 정점 전부가 쌓이는 깊이(= n)까지 간다.
    assert c.visits == g.n and c.max_frontier == g.n


def test_frontier_shapes():
    cb, cd = G.Counter(), G.Counter()
    G.bfs(G.star_graph(100), 0, cb)
    G.dfs_iterative(G.star_graph(100), 0, cd)
    # 별 그래프: BFS 큐에는 잎 99개가 한꺼번에, DFS 스택은 중심+잎 2칸.
    assert cb.max_frontier == 99 and cd.max_frontier == 2
    cb, cd = G.Counter(), G.Counter()
    G.bfs(G.path_graph(100), 0, cb)
    G.dfs_iterative(G.path_graph(100), 0, cd)
    # 경로 그래프: BFS 큐는 늘 1칸, DFS 스택은 100칸. 두 탐색의 프런티어 크기가 반대 모양.
    assert cb.max_frontier == 1 and cd.max_frontier == 100


def test_bfs_all_is_linear_and_levels_np_agree():
    # 평균 차수 약 3 인 희소 무작위 그래프. 연결 요소가 여럿이다. 시드 1 고정.
    g = G.erdos_renyi(3000, 3 / 2999, random.Random(1))
    c = G.Counter()
    dist, comp = G.bfs_all(g, c)
    # bfs_all 은 요소가 여럿이어도 V 번 방문, 간선은 양 끝에서 2E 번.
    assert c.visits == g.n and c.edge_checks == 2 * g.m
    f = G.Counter()
    comps = G.bfs_all_fresh_marks(g, f)
    # 표시판을 새로 만드는 판도 간선 검사 수는 같다(느린 건 배열 재생성 쪽이라 이 숫자로는 안 드러남).
    assert comps == max(comp) + 1 and f.edge_checks == 2 * g.m
    d0, _ = G.bfs(g, 0)
    assert np.array_equal(np.array([-1 if d is None else d for d in d0]), G.bfs_levels_np(g, 0))
