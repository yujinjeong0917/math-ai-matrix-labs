"""알고리즘 8장 테스트. `uv run pytest -q algorithms/ch08_greedy`

Hypothesis 를 의존성에 넣지 않으려고, 성질 테스트는 시드를 고정한 무작위 입력 반복으로 한다.
"""

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import algorithms_ch08_greedy as A  # noqa: E402


# ---------------------------------------------------------------- 거스름돈
def test_first_screen_coin_counterexample():
    assert A.greedy_coins([1, 3, 4], 6) == [4, 1, 1]
    best, combo = A.coin_change_dp([1, 3, 4], 6)
    assert best[6] == 2 and combo == [3, 3]


def test_greedy_coins_sums_to_amount():
    rng = A.seeded(1)
    for _ in range(200):
        cs = sorted({1} | {rng.randint(2, 30) for _ in range(rng.randint(1, 4))})
        x = rng.randint(1, 300)
        assert sum(A.greedy_coins(cs, x)) == x
        best, combo = A.coin_change_dp(cs, x)
        assert sum(combo) == x and len(combo) == best[x] <= len(A.greedy_coins(cs, x))


@pytest.mark.parametrize("coins", [[1, 5, 10, 25], [1, 5, 10, 50, 100, 500], [1, 2, 5, 10, 20, 50, 100, 200]])
def test_canonical_systems_1_to_500(coins):
    best, _ = A.coin_change_dp(coins, 500)
    for x in range(1, 501):
        assert len(A.greedy_coins(coins, x)) == best[x]


def test_dp_matches_brute_force_small():
    for cs in ([1, 3, 4], [1, 7, 10], [1, 2, 5]):
        best, _ = A.coin_change_dp(cs, 20)
        for x in range(21):
            m = min((sum(t) for t in itertools.product(range(21), repeat=len(cs))
                     if sum(a * b for a, b in zip(t, cs)) == x ), default=None)
            assert best[x] == m


def test_kozen_zaks_range_on_random_systems():
    """반례가 있으면 가장 작은 반례는 c3+1 < x < cm+c(m-1) 안에 있다(Kozen·Zaks 1994 Theorem 4)."""
    rng = A.seeded(2)
    for _ in range(150):
        cs = sorted({1} | set(rng.sample(range(2, 40), rng.randint(2, 4))))
        if len(cs) < 3:
            continue
        lo, hi = cs[2] + 1, cs[-1] + cs[-2]
        fails = A.coin_failures(cs, 3 * hi)
        if fails:
            assert lo < fails[0][0] < hi


# ---------------------------------------------------------------- UnionFind
@pytest.mark.parametrize("compress,by_rank", [(True, True), (False, True), (True, False), (False, False)])
def test_unionfind_matches_naive(compress, by_rank):
    rng = A.seeded(3)
    for _ in range(50):
        n = rng.randint(2, 40)
        pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 60))]
        uf = A.UnionFind(n, compress=compress, by_rank=by_rank)
        for a, b in pairs:
            uf.union(a, b)
        rep = A.naive_components(n, pairs)
        for i in range(n):
            for j in range(n):
                assert (uf.find(i) == uf.find(j)) == (rep[i] == rep[j])
        assert uf.count == len(set(rep))


# ---------------------------------------------------------------- MST
def test_first_screen_mst():
    edges = [(1, 0, 1), (2, 1, 2), (3, 0, 2), (4, 2, 3), (5, 1, 3)]
    tr = []
    total, chosen = A.kruskal(4, edges, trace=tr)
    assert total == 7 and [ok for _, ok, _ in tr] == [True, True, False, True]


def test_kruskal_prim_bruteforce_agree():
    rng = A.seeded(4)
    for _ in range(80):
        n = rng.randint(2, 6)
        m = rng.randint(n - 1, n * (n - 1) // 2)
        edges = A.random_connected_graph(n, m, rng, wmax=rng.choice([3, 10, 100]))  # 작은 wmax 는 같은 무게를 만든다
        kw, kc = A.kruskal(n, edges)
        pw, pc = A.prim(n, edges, start=rng.randrange(n))
        assert A.is_spanning_tree(n, kc) and A.is_spanning_tree(n, pc)
        assert kw == pw == A.brute_force_mst(n, edges)  # 같은 무게가 있으면 간선 집합은 다를 수 있어 무게로 비교


def test_kruskal_variants_same_weight():
    rng = A.seeded(5)
    edges = A.random_connected_graph(300, 1200, rng)
    w = A.kruskal(300, edges)[0]
    for comp in (True, False):
        for rank in (True, False):
            assert A.kruskal(300, edges, compress=comp, by_rank=rank)[0] == w


def test_cut_property_every_accepted_edge():
    """받은 간선마다, 받기 직전 그 간선 한쪽 끝의 조각을 절단으로 두면 그 간선이 절단에서 가장 가볍다."""
    rng = A.seeded(6)
    for _ in range(40):
        n = rng.randint(3, 12)
        edges = A.random_connected_graph(n, rng.randint(n - 1, n * (n - 1) // 2), rng, wmax=30)
        tr = []
        A.kruskal(n, edges, trace=tr)
        prev = list(range(n))
        for (w, u, v), ok, comp in tr:
            if ok:
                side = {i for i in range(n) if prev[i] == prev[u]}
                assert w == A.lightest_crossing(n, edges, side)
            else:
                assert prev[u] == prev[v]  # 거부한 간선은 순환을 만든다
            prev = comp


def test_prim_with_dijkstra_key_is_not_mst():
    # s-a 3, s-b 3, a-b 1: MST 는 3+1=4, 최단경로 트리는 s-a, s-b 로 6
    edges = [(3, 0, 1), (3, 0, 2), (1, 1, 2)]
    assert A.prim(3, edges)[0] == 4
    assert A.prim_with_dijkstra_key(3, edges)[0] == 6


# ---------------------------------------------------------------- 같은 탐욕, 다른 구조
def test_greedy_forest_always_optimal_matching_not():
    edges = [(3, 0, 1), (4, 1, 2), (3, 2, 3)]
    assert A.greedy_max_weight(4, edges, A.is_matching)[0] == 4
    assert A.brute_force_max_weight(4, edges, A.is_matching) == 6
    rng = A.seeded(7)
    for _ in range(200):
        edges = A.random_small_graph(5, 0.6, rng)
        if edges:
            assert A.greedy_max_weight(5, edges, A.is_forest)[0] == A.brute_force_max_weight(5, edges, A.is_forest)


# ---------------------------------------------------------------- 최대 덮기
def test_coverage_bound_values():
    assert A.coverage_bound(1) == 1
    assert A.coverage_bound(2) == Fraction(3, 4)
    assert A.coverage_bound(3) == Fraction(19, 27)
    import math
    assert all(float(A.coverage_bound(k)) > 1 - 1 / math.e for k in range(1, 30))


def test_first_screen_coverage_tight():
    sets = [frozenset({1, 3}), frozenset({1, 2}), frozenset({3, 4})]
    assert A.greedy_max_coverage(sets, 2)[0] == 3
    assert A.brute_force_coverage(sets, 2)[0] == 4


def test_greedy_coverage_never_below_bound():
    rng = A.seeded(8)
    for _ in range(300):
        k = rng.randint(1, 4)
        sets = A.random_sets(rng.randint(k, 7), rng.randint(3, 10), rng.uniform(0.1, 0.6), rng)
        ov = A.brute_force_coverage(sets, k)[0]
        if ov == 0:
            continue
        gv = A.greedy_max_coverage(sets, k)[0]
        assert Fraction(gv, ov) >= A.coverage_bound(k)


def test_size_rule_can_fall_below_bound():
    hand = [frozenset({1, 2, 3, 4}), frozenset({1, 2, 3, 5}), frozenset({6, 7, 8})]
    assert A.greedy_max_coverage(hand, 2, by="size")[0] == 5
    assert A.greedy_max_coverage(hand, 2)[0] == 7
    assert Fraction(5, 7) < A.coverage_bound(2)
