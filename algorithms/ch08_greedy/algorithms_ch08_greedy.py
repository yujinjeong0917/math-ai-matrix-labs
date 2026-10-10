"""알고리즘 8장: 탐욕은 언제 통하나.

이 파일 하나에 장의 구현이 모두 있다(다른 장과 import 를 공유하지 않는다).

- greedy_coins / coin_change_dp: 거스름돈. 탐욕(가장 큰 동전부터)과 최소 동전 수 DP.
  DP 는 10장(동적 계획법)에서 다룰 내용이지만, 장끼리 동시에 쓰이므로 여기에 필요한 만큼 직접 짰다.
- UnionFind: 경로 압축과 랭크(높이 어림)로 합치는 서로소 집합. 두 기능을 끌 수 있다(비교용).
- kruskal: 가벼운 간선부터, 순환을 만들면 건너뛴다(Kruskal 1956 의 Construction A).
- prim: 한 점에서 시작한 조각을 가장 가까운 점으로 넓힌다(Jarník 1930, Prim 1957). 이진 힙 사용.
  7장의 다익스트라와 키만 다르다(d(u)+w 대신 w).
- prim_with_dijkstra_key: 흔한 실패용. 키를 d(u)+w 로 두면 최단경로 트리가 나온다(MST 가 아니다).
- brute_force_mst: 작은 그래프에서 모든 (n-1)개 간선 조합을 보는 전수 탐색(테스트·대조용).
- greedy_max_weight(edges, n, independent): "무거운 것부터, 독립성을 깨면 건너뛴다" 하나로
  숲(그래프 매트로이드)과 매칭(매트로이드 아님)을 같은 규칙으로 고른다.
- greedy_max_coverage / brute_force_coverage: 최대 덮기(집합 k 개로 원소를 최대한 많이 덮기).
  탐욕은 매번 새로 덮는 원소가 가장 많은 집합을 고르고, 같으면 번호가 작은 집합을 고른다.

모든 함수는 선택 인자 counter(Counter)에 비교·꺼내기 등 횟수를 센다.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction

INF = math.inf


@dataclass
class Counter:
    edge_scan: int = 0   # 간선 하나를 보고 받을지 판단한 횟수(Kruskal) / 이웃 간선을 본 횟수(Prim)
    find_calls: int = 0  # UnionFind.find 호출 수
    find_steps: int = 0  # find 가 부모 포인터를 따라 올라간 걸음 수
    unions: int = 0      # 실제로 두 집합을 합친 횟수
    push: int = 0        # 힙에 넣은 횟수
    pop: int = 0         # 힙에서 꺼낸 횟수(낡은 항목 포함)
    gain_evals: int = 0  # 최대 덮기에서 "새로 덮는 원소 수"를 계산한 횟수


def seeded(seed):
    return random.Random(seed)


# ---------------------------------------------------------------- 거스름돈
def greedy_coins(coins, amount):
    """남은 금액 이하인 가장 큰 동전을 되풀이해 고른다. 고른 동전 목록을 돌려준다."""
    cs = sorted(coins, reverse=True)
    out = []
    rest = amount
    for c in cs:
        while rest >= c:
            rest -= c
            out.append(c)
    if rest != 0:
        raise ValueError("이 동전 체계로는 만들 수 없는 금액(1원짜리가 없음)")
    return out


def coin_change_dp(coins, amount):
    """금액 0..amount 각각의 최소 동전 수 표와, amount 를 만드는 최소 동전 목록.

    best[x] = min_c (best[x-c] + 1). 같은 수면 큰 동전을 먼저 쓴 쪽을 남긴다(표시용 규칙).
    """
    best = [0] + [INF] * amount
    last = [0] * (amount + 1)
    for x in range(1, amount + 1):
        for c in sorted(coins, reverse=True):
            if c <= x and best[x - c] + 1 < best[x]:
                best[x] = best[x - c] + 1
                last[x] = c
    combo = []
    x = amount
    while x > 0:
        combo.append(last[x])
        x -= last[x]
    return best, sorted(combo, reverse=True)


def coin_failures(coins, max_amount):
    """1..max_amount 에서 탐욕 동전 수 > 최소 동전 수 인 금액 목록과 (금액, 탐욕 수, 최소 수)."""
    best, _ = coin_change_dp(coins, max_amount)
    rows = []
    for x in range(1, max_amount + 1):
        g = len(greedy_coins(coins, x))
        if g > best[x]:
            rows.append((x, g, best[x]))
    return rows


# ---------------------------------------------------------------- UnionFind
class UnionFind:
    """서로소 집합. compress=경로 압축, by_rank=랭크로 합치기. 둘 다 끄면 순진한 연결 리스트 꼴이 된다."""

    def __init__(self, n, compress=True, by_rank=True, counter=None):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.compress = compress
        self.by_rank = by_rank
        self.c = counter if counter is not None else Counter()
        self.count = n  # 연결 요소 수

    def find(self, x):
        self.c.find_calls += 1
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
            self.c.find_steps += 1
        if self.compress:
            while self.parent[x] != root:
                self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.by_rank:
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        else:
            self.parent[ra] = rb  # 늘 앞 집합의 뿌리를 뒤 집합 밑에 붙인다
        self.c.unions += 1
        self.count -= 1
        return True


def naive_components(n, pairs):
    """대조용: 집합 목록을 직접 합치는 순진한 방법. 각 점의 대표(가장 작은 번호)를 돌려준다."""
    sets = [{i} for i in range(n)]
    where = list(range(n))
    for a, b in pairs:
        ia, ib = where[a], where[b]
        if ia == ib:
            continue
        sets[ia] |= sets[ib]
        for x in sets[ib]:
            where[x] = ia
        sets[ib] = set()
    return [min(sets[where[i]]) for i in range(n)]


# ---------------------------------------------------------------- 그래프
def random_connected_graph(n, m, rng, wmax=1000):
    """연결된 무향 그래프. 먼저 무작위 신장 트리(n-1 간선)를 만든 뒤 서로 다른 간선을 m 개까지 더한다.
    간선은 (w, u, v), u < v, 무게는 1..wmax 정수(같은 무게가 있을 수 있다)."""
    assert n - 1 <= m <= n * (n - 1) // 2
    order = list(range(n))
    rng.shuffle(order)
    pairs = set()
    for i in range(1, n):
        a, b = order[i], order[rng.randrange(i)]
        pairs.add((min(a, b), max(a, b)))
    while len(pairs) < m:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)
    rng.shuffle(pairs)
    return [(rng.randint(1, wmax), u, v) for u, v in pairs]


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for w, u, v in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def kruskal(n, edges, counter=None, trace=None, compress=True, by_rank=True):
    """가벼운 간선부터 보고, 두 끝이 이미 같은 조각이면(순환) 건너뛴다. (총무게, 채택 간선 목록)."""
    c = counter if counter is not None else Counter()
    uf = UnionFind(n, compress=compress, by_rank=by_rank, counter=c)
    chosen = []
    total = 0
    for w, u, v in sorted(edges):
        c.edge_scan += 1
        ok = uf.union(u, v)
        if ok:
            chosen.append((w, u, v))
            total += w
        if trace is not None:
            trace.append(((w, u, v), ok, [uf.find(i) for i in range(n)]))
        if len(chosen) == n - 1:
            break
    return total, chosen


def prim(n, edges, start=0, counter=None, adj=None):
    """한 점에서 시작한 조각에 '조각까지 가장 가까운 점'을 하나씩 붙인다. 이진 힙, 낡은 항목은 건너뛴다."""
    c = counter if counter is not None else Counter()
    adj = adj if adj is not None else adjacency(n, edges)
    inside = [False] * n
    key = [INF] * n
    key[start] = 0
    heap = [(0, start, -1)]
    c.push += 1
    total = 0
    chosen = []
    while heap:
        k, u, p = heapq.heappop(heap)
        c.pop += 1
        if inside[u] or k > key[u]:
            continue
        inside[u] = True
        total += k
        if p >= 0:
            chosen.append((k, min(p, u), max(p, u)))
        for v, w in adj[u]:
            c.edge_scan += 1
            if not inside[v] and w < key[v]:   # 키 = 조각까지의 간선 하나 무게
                key[v] = w
                heapq.heappush(heap, (w, v, u))
                c.push += 1
    return total, chosen


def prim_with_dijkstra_key(n, edges, start=0, adj=None):
    """흔한 실패: 7장 다익스트라처럼 키를 d(u)+w 로 둔다. 결과는 start 에서의 최단경로 트리."""
    adj = adj if adj is not None else adjacency(n, edges)
    inside = [False] * n
    dist = [INF] * n
    dist[start] = 0
    heap = [(0, start, -1, 0)]
    chosen = []
    while heap:
        d, u, p, w_in = heapq.heappop(heap)
        if inside[u] or d > dist[u]:
            continue
        inside[u] = True
        if p >= 0:
            chosen.append((w_in, min(p, u), max(p, u)))
        for v, w in adj[u]:
            if not inside[v] and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v, u, w))
    return sum(e[0] for e in chosen), chosen


def is_spanning_tree(n, chosen):
    if len(chosen) != n - 1:
        return False
    uf = UnionFind(n)
    return all(uf.union(u, v) for _, u, v in chosen)


def brute_force_mst(n, edges):
    """(n-1)개 간선 조합을 모두 보고 신장 트리 중 가장 가벼운 것의 무게."""
    best = INF
    for comb in itertools.combinations(edges, n - 1):
        if is_spanning_tree(n, comb):
            best = min(best, sum(e[0] for e in comb))
    return best


def lightest_crossing(n, edges, side):
    """절단(side 에 든 점 / 나머지)을 가로지르는 간선 중 가장 가벼운 무게."""
    ws = [w for w, u, v in edges if (u in side) != (v in side)]
    return min(ws) if ws else INF


# ---------------------------------------------------------------- 같은 탐욕, 다른 구조
def is_forest(n, chosen):
    uf = UnionFind(n)
    return all(uf.union(u, v) for _, u, v in chosen)


def is_matching(n, chosen):
    used = set()
    for _, u, v in chosen:
        if u in used or v in used:
            return False
        used.add(u)
        used.add(v)
    return True


def greedy_max_weight(n, edges, independent):
    """무거운 간선부터 보고, 넣어도 independent 가 참이면 넣는다. 같은 무게는 (u, v) 순."""
    chosen = []
    for e in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        if independent(n, chosen + [e]):
            chosen.append(e)
    return sum(e[0] for e in chosen), chosen


def brute_force_max_weight(n, edges, independent):
    best = 0
    for r in range(1, len(edges) + 1):
        for comb in itertools.combinations(edges, r):
            if independent(n, list(comb)):
                best = max(best, sum(e[0] for e in comb))
    return best


def random_small_graph(n, p, rng, wmax=10):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((rng.randint(1, wmax), u, v))
    return edges


# ---------------------------------------------------------------- 최대 덮기
def covered(sets, idx):
    s = set()
    for i in idx:
        s |= sets[i]
    return len(s)


def greedy_max_coverage(sets, k, counter=None, by="gain"):
    """k 번, 새로 덮는 원소가 가장 많은 집합을 고른다(같으면 번호가 작은 쪽).
    by="size" 는 흔한 실패용: 이미 덮인 것과 상관없이 크기가 큰 집합부터 고른다."""
    c = counter if counter is not None else Counter()
    if by == "size":
        idx = sorted(range(len(sets)), key=lambda i: (-len(sets[i]), i))[:k]
        return covered(sets, idx), idx
    got = set()
    idx = []
    for _ in range(k):
        best_i, best_gain = None, -1
        for i, s in enumerate(sets):
            if i in idx:
                continue
            c.gain_evals += 1
            g = len(s - got)
            if g > best_gain:
                best_i, best_gain = i, g
        if best_i is None:
            break
        idx.append(best_i)
        got |= sets[best_i]
    return len(got), idx


def brute_force_coverage(sets, k):
    best, arg = -1, None
    for comb in itertools.combinations(range(len(sets)), min(k, len(sets))):
        v = covered(sets, comb)
        if v > best:
            best, arg = v, list(comb)
    return best, arg


def coverage_bound(k):
    """Nemhauser·Wolsey·Fisher(1978) 초록의 1 - ((k-1)/k)^k 를 정확한 분수로."""
    return 1 - Fraction(k - 1, k) ** k


def random_sets(n_sets, universe, p, rng):
    return [frozenset(e for e in range(universe) if rng.random() < p) for _ in range(n_sets)]
