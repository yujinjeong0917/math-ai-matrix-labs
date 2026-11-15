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
    # 체계 [1,3,4], 6원: 탐욕은 4+1+1 = 3개, 최적은 3+3 = 2개.
    assert A.greedy_coins([1, 3, 4], 6) == [4, 1, 1]
    best, combo = A.coin_change_dp([1, 3, 4], 6)
    assert best[6] == 2 and combo == [3, 3]


def test_greedy_coins_sums_to_amount():
    rng = A.seeded(1)
    for _ in range(200):
        # 항상 1원을 넣어 모든 금액을 만들 수 있게 한다(그렇지 않으면 greedy_coins 가 ValueError).
        cs = sorted({1} | {rng.randint(2, 30) for _ in range(rng.randint(1, 4))})
        x = rng.randint(1, 300)
        assert sum(A.greedy_coins(cs, x)) == x
        best, combo = A.coin_change_dp(cs, x)
        # DP 는 최소이므로 탐욕 개수 이상일 수 없다(<=).
        assert sum(combo) == x and len(combo) == best[x] <= len(A.greedy_coins(cs, x))


@pytest.mark.parametrize("coins", [[1, 5, 10, 25], [1, 5, 10, 50, 100, 500], [1, 2, 5, 10, 20, 50, 100, 200]])
# 미국 센트, 한국 원 동전, 유로 동전 체계는 1..500 에서 탐욕이 최적이다(이 범위 전수 확인).
def test_canonical_systems_1_to_500(coins):
    best, _ = A.coin_change_dp(coins, 500)
    for x in range(1, 501):
        assert len(A.greedy_coins(coins, x)) == best[x]


def test_dp_matches_brute_force_small():
    for cs in ([1, 3, 4], [1, 7, 10], [1, 2, 5]):
        best, _ = A.coin_change_dp(cs, 20)
        for x in range(21):
            # 각 동전을 0..20 개 쓰는 모든 조합을 직접 나열해 최소 개수를 구한다(DP 의 독립 검증, 느린 전수).
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
        # 정리의 범위: 세 번째로 작은 동전 + 1 < 가장 작은 반례 < 가장 큰 두 동전의 합. 정확한 식은 docstring 의 논문 정리 기준.
        lo, hi = cs[2] + 1, cs[-1] + cs[-2]
        # 범위 위로 넉넉히 3배까지 탐색. 반례가 없으면 fails 가 비어 검사하지 않는다.
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
        # rep[i] = 점 i 의 집합 대표. 같은 집합이면 대표가 같다.
        rep = A.naive_components(n, pairs)
        for i in range(n):
            for j in range(n):
                # 모든 쌍에 대해 같은 집합 여부가 일치하는지. 압축·랭크 옵션 네 조합 모두 같은 답이어야 한다.
                assert (uf.find(i) == uf.find(j)) == (rep[i] == rep[j])
        assert uf.count == len(set(rep))


# ---------------------------------------------------------------- MST
def test_first_screen_mst():
    # 무게순으로 (0,1) 받음, (1,2) 받음, (0,2) 는 순환이라 거부, (2,3) 받음 -> 1+2+4 = 7. 4개 점이므로 3개를 받으면 멈춰 (1,3) 은 보지도 않는다(trace 에 4줄).
    edges = [(1, 0, 1), (2, 1, 2), (3, 0, 2), (4, 2, 3), (5, 1, 3)]
    tr = []
    total, chosen = A.kruskal(4, edges, trace=tr)
    assert total == 7 and [ok for _, ok, _ in tr] == [True, True, False, True]


def test_kruskal_prim_bruteforce_agree():
    rng = A.seeded(4)
    for _ in range(80):
        n = rng.randint(2, 6)
        m = rng.randint(n - 1, n * (n - 1) // 2)
        # wmax 를 작게 하면 같은 무게가 늘어 동률 처리를 시험한다.
        edges = A.random_connected_graph(n, m, rng, wmax=rng.choice([3, 10, 100]))  # 작은 wmax 는 같은 무게를 만든다
        kw, kc = A.kruskal(n, edges)
        pw, pc = A.prim(n, edges, start=rng.randrange(n))
        assert A.is_spanning_tree(n, kc) and A.is_spanning_tree(n, pc)
        # 셋 다 총무게가 같아야 한다.
        assert kw == pw == A.brute_force_mst(n, edges)  # 같은 무게가 있으면 간선 집합은 다를 수 있어 무게로 비교


def test_kruskal_variants_same_weight():
    rng = A.seeded(5)
    # 정점 300, 간선 1200 으로 네 가지 UnionFind 옵션이 총무게를 바꾸지 않는지(옵션은 속도만 바꾼다).
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
        # prev = 직전 단계의 조각 대표 목록(처음엔 모두 따로).
        prev = list(range(n))
        for (w, u, v), ok, comp in tr:
            if ok:
                # 받기 직전 u 가 속한 조각 = 절단의 한쪽. 절단 성질: 이 절단을 가로지르는 가장 가벼운 간선은 MST 에 속한다.
                side = {i for i in range(n) if prev[i] == prev[u]}
                assert w == A.lightest_crossing(n, edges, side)
            else:
                assert prev[u] == prev[v]  # 거부한 간선은 순환을 만든다
            prev = comp


def test_prim_with_dijkstra_key_is_not_mst():
    # (a-b 를 거쳐 가는 길 3+1 = 4 는 직접 3 보다 길어서 최단경로 트리는 a-b 를 안 쓴다.)
    # s-a 3, s-b 3, a-b 1: MST 는 3+1=4, 최단경로 트리는 s-a, s-b 로 6
    edges = [(3, 0, 1), (3, 0, 2), (1, 1, 2)]
    assert A.prim(3, edges)[0] == 4
    assert A.prim_with_dijkstra_key(3, edges)[0] == 6


# ---------------------------------------------------------------- 같은 탐욕, 다른 구조
def test_greedy_forest_always_optimal_matching_not():
    # 경로 0-1-2-3, 무게 3, 4, 3. 탐욕은 가운데 4 를 잡아 끝(총 4), 최적 매칭은 양쪽 3+3 = 6.
    edges = [(3, 0, 1), (4, 1, 2), (3, 2, 3)]
    assert A.greedy_max_weight(4, edges, A.is_matching)[0] == 4
    assert A.brute_force_max_weight(4, edges, A.is_matching) == 6
    rng = A.seeded(7)
    for _ in range(200):
        # 정점 5개, 간선 확률 0.6 이라 간선이 최대 10개. 전수 탐색 2^10 이내라 빠르다.
        edges = A.random_small_graph(5, 0.6, rng)
        if edges:
            assert A.greedy_max_weight(5, edges, A.is_forest)[0] == A.brute_force_max_weight(5, edges, A.is_forest)


# ---------------------------------------------------------------- 최대 덮기
def test_coverage_bound_values():
    assert A.coverage_bound(1) == 1
    assert A.coverage_bound(2) == Fraction(3, 4)
    # 1 - (2/3)^3 = 1 - 8/27 = 19/27.
    assert A.coverage_bound(3) == Fraction(19, 27)
    import math
    # 경계가 1-1/e 보다 항상 큼(k 가 커질수록 1-1/e 에 접근). 여기서만 float 비교라 k < 30 로 제한.
    assert all(float(A.coverage_bound(k)) > 1 - 1 / math.e for k in range(1, 30))


def test_first_screen_coverage_tight():
    # 동률 규칙(번호 작은 쪽)이 탐욕을 나쁜 쪽으로 이끈다: 첫 이득이 모두 2 라 {1,3} 을 고르고 그 뒤 이득이 1 뿐. 3 대 4 = 3/4 는 k=2 경계와 같다(경계가 빡빡함).
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
        # 탐욕 덮은 수 / 최적 덮은 수가 보장 비율 이상이어야 한다(Fraction 이라 정확한 비교).
        assert Fraction(gv, ov) >= A.coverage_bound(k)


def test_size_rule_can_fall_below_bound():
    # 크기순으로 고르면 겹치는 앞 두 집합(5개 덮음), 이득순이면 큰 것 + 서로소인 {6,7,8} (7개). 5/7 < 3/4 라 보장 아래로 떨어진다.
    hand = [frozenset({1, 2, 3, 4}), frozenset({1, 2, 3, 5}), frozenset({6, 7, 8})]
    assert A.greedy_max_coverage(hand, 2, by="size")[0] == 5
    assert A.greedy_max_coverage(hand, 2)[0] == 7
    assert Fraction(5, 7) < A.coverage_bound(2)
