"""알고리즘 7장 테스트. `uv run pytest -q algorithms/ch07_shortest_path`

Hypothesis 를 의존성에 넣지 않으려고, 성질 테스트는 시드를 고정한 무작위 입력 반복으로 한다.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import algorithms_ch07_shortest_path as A  # noqa: E402


def floyd(n, edges):
    # 정답 대조용 Floyd-Warshall(O(n^3)). 평행 간선은 더 짧은 쪽만 남긴다.
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)
    for u, v, w in edges:
        D[u, v] = min(D[u, v], w)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def test_bfs_by_hops_is_wrong_on_weighted_example():
    # 0->1 직행은 길이 10 에 간선 1개, 0->2->1 은 길이 3 에 간선 2개. BFS 는 간선 수만 봐서 10 을 고른다.
    g = A.from_edges(3, [(0, 1, 10), (0, 2, 1), (2, 1, 2)])
    hops, cost = A.bfs_hops(g, 0)
    assert hops[1] == 1 and cost[1] == 10
    assert A.dijkstra(g, 0)[0][1] == 3


def test_dijkstra_with_settled_set_fails_on_three_node_negative_edge():
    # 음수 간선 2->1(-2): 참 거리는 d(1) = 2 + (-2) = 0. Dijkstra 는 1 을 거리 1 로 먼저 확정해 놓치고 [0, 1, 2] 를 낸다.
    g = A.from_edges(3, [(0, 1, 1), (0, 2, 2), (2, 1, -2)])
    assert A.dijkstra(g, 0)[0] == [0, 1, 2]          # 틀린 답이 실제로 나온다
    assert A.bellman_ford(g, 0)[0] == [0, 0, 2]      # 참 거리
    assert A.dijkstra_reinsert(g, 0)[0] == [0, 0, 2]  # 다시 넣기 변형은 여기서 맞힌다


@pytest.mark.parametrize("seed", range(40))
def test_dijkstra_equals_bellman_ford_and_floyd_on_nonnegative(seed):
    rng = A.seeded(seed)
    n = rng.randint(2, 25)
    m = rng.randint(1, min(n * (n - 1), 80))
    # 길이 0..15 의 비음수. 0 길이 간선도 포함해 동률 처리를 시험한다.
    edges = A.random_digraph(n, m, rng, 0, 15)
    g = A.from_edges(n, edges)
    d1, p1 = A.dijkstra(g, 0)
    d2, _ = A.bellman_ford(g, 0)
    D = floyd(n, edges)
    # 정수 길이라 부동소수 오차가 없어 == 로 비교해도 된다(inf 끼리도 같음).
    assert d1 == d2
    assert all((d1[v] == A.INF and np.isinf(D[0, v])) or d1[v] == D[0, v] for v in range(n))
    # 최단 거리의 성질(삼각 부등식). 닿는 u 에서 나가는 간선만 검사.
    # 종료 후 모든 간선에서 d(v) <= d(u) + w
    for u, v, w in edges:
        if d1[u] < A.INF:
            assert d1[v] <= d1[u] + w
    # 부모를 따라간 경로의 길이 = 거리
    for v in range(n):
        if d1[v] < A.INF and v != 0:
            path = [v]
            while p1[path[-1]] is not None:
                path.append(p1[path[-1]])
            assert path[-1] == 0
            assert A.path_cost(g, path[::-1]) == d1[v]


@pytest.mark.parametrize("seed", range(20))
def test_bellman_ford_matches_floyd_with_negative_edges(seed):
    rng = A.seeded(100 + seed)
    # 음수 간선 15%(9개), 음수 순환 없음이 보장된 그래프. 시드 100 + seed.
    edges, _ = A.random_negative_graph(20, 60, 0.15, rng)
    g = A.from_edges(20, edges)
    D = floyd(20, edges)
    bf = A.bellman_ford(g, 0)[0]
    assert all((bf[v] == A.INF and np.isinf(D[0, v])) or bf[v] == D[0, v] for v in range(20))
    assert A.dijkstra_reinsert(g, 0)[0] == bf


def test_bellman_ford_early_stop_does_not_change_answer():
    rng = A.seeded(3)
    edges, _ = A.random_negative_graph(30, 90, 0.1, rng)
    g = A.from_edges(30, edges)
    c1, c2 = A.Counter(), A.Counter()
    a = A.bellman_ford(g, 0, c1, early_stop=True)[0]
    b = A.bellman_ford(g, 0, c2, early_stop=False)[0]
    assert a == b
    # 조기 종료는 라운드를 줄일 뿐 결과는 같다. 끄면 V 라운드를 모두 돈다.
    assert c1.relax <= c2.relax


@pytest.mark.parametrize("seed", range(15))
def test_bellman_ford_detects_injected_reachable_negative_cycle(seed):
    rng = A.seeded(500 + seed)
    g = A.from_edges(30, A.random_digraph(30, 100, rng))
    reach = [v for v in range(30) if A.reachable(g, 0)[v] and v != 0]
    # 출발점에서 닿는 두 정점을 골라 a->b(+1), b->a(-3) 를 더하면 합 -2 의 음수 순환이 닿는다.
    a, b = rng.sample(reach, 2)
    g.add(a, b, 1)
    g.add(b, a, -3)
    with pytest.raises(A.NegativeCycle):
        A.bellman_ford(g, 0)


def test_unreachable_negative_cycle_is_not_reported():
    # 출발점에서 닿지 않는 음수 순환은 오류가 아니다. dist 가 inf 인 정점 간선은 이완에서 건너뛴다.
    # 0 -> 1 만 닿고, 2 <-> 3 의 음수 순환은 0 에서 닿지 않는다
    g = A.from_edges(4, [(0, 1, 1), (2, 3, 1), (3, 2, -5)])
    d = A.bellman_ford(g, 0)[0]
    assert d[:2] == [0, 1] and d[2] == A.INF


@pytest.mark.parametrize("seed", range(20))
def test_a_star_manhattan_and_zero_match_bfs_cost(seed):
    rng = A.seeded(seed)
    grid = A.random_grid(15, 15, 0.25, rng, keep=[(0, 0), (14, 14)])
    # BFS 로 구한 정답. 목표에 닿을 수 없으면 None 이고 A* 도 None 이어야 한다(장애물 25%라 그런 시드가 섞일 수 있음).
    opt = A.grid_dijkstra_cost(grid, (0, 0), (14, 14))
    c0, path0, _ = A.a_star(grid, (0, 0), (14, 14), A.zero)
    c1, path1, _ = A.a_star(grid, (0, 0), (14, 14), A.manhattan)
    assert c0 == opt and c1 == opt
    if opt is not None:
        # 경로 칸 수 = 비용 + 1(시작 칸 포함).
        assert len(path1) == opt + 1
        # h=0 A* 비용 = 격자를 그래프로 바꾼 다익스트라 비용
        g = A.grid_to_graph(grid)
        assert A.dijkstra(g, 0)[0][14 * 15 + 14] == c0


def test_manhattan_never_expands_more_than_zero_heuristic_on_sample():
    rng = A.seeded(1)
    grid = A.random_grid(30, 30, 0.2, rng, keep=[(0, 0), (29, 29)])
    ca, cb = A.Counter(), A.Counter()
    A.a_star(grid, (0, 0), (29, 29), A.zero, ca)
    A.a_star(grid, (0, 0), (29, 29), A.manhattan, cb)
    # 이 한 표본(시드 1)에서의 관찰이지 항상 성립하는 법칙으로 쓴 것이 아니다(동률 처리에 따라 달라질 수 있음).
    assert cb.expand <= ca.expand


def test_double_manhattan_is_suboptimal_on_small_grid():
    # '#' 장애물. 손으로 센 최단 거리 7(맨해튼 거리 3+4 와 같음).
    grid, s, t = A.grid_from_strings(["S..##", "....#", ".#...", "...#T"])
    assert A.grid_dijkstra_cost(grid, s, t) == 7
    assert A.a_star(grid, s, t, A.manhattan)[0] == 7
    # h 를 2배로 키우면 허용성이 깨져 최적이 아닌 9 를 낸다(이 격자에서 실제로 확인한 값).
    assert A.a_star(grid, s, t, A.scaled(A.manhattan, 2))[0] == 9


def test_invariant_breaks_exactly_when_wrong_vertex_is_settled():
    g = A.from_edges(3, [(0, 1, 1), (0, 2, 2), (2, 1, -2)])
    true = A.bellman_ford(g, 0)[0]
    tr = []
    A.dijkstra(g, 0, trace=tr)
    # 단계마다 불변식 2(확정 라벨 = 참 거리)가 성립하는지. trace 원소는 (종류, 정점, 간선, dist, done, 힙).
    flags = [A.check_settled_invariants(d, done, h, true)[1] for _, _, _, d, done, h in tr]
    first_bad = flags.index(False)
    kind, u = tr[first_bad][0], tr[first_bad][1]
    # 불변식이 처음 깨지는 순간 = 정점 1 을 잘못된 라벨 1 로 확정하는 단계.
    assert kind == "settle" and u == 1


@pytest.mark.parametrize("seed", range(10))
def test_settled_invariants_hold_every_step_on_nonnegative(seed):
    rng = A.seeded(700 + seed)
    edges = A.random_digraph(12, 30, rng, 0, 9)
    g = A.from_edges(12, edges)
    true = A.bellman_ford(g, 0)[0]
    tr = []
    A.dijkstra(g, 0, trace=tr)
    for _, _, _, d, done, h in tr:
        assert A.check_settled_invariants(d, done, h, true) == (True, True)
