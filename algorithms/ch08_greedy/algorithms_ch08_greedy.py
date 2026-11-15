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
    # 탐욕 규칙: 큰 동전부터 가능한 만큼. 시간 O(|coins| log |coins| + 동전 개수). 정규 체계(예: 1,5,10,25)에서는 최적이지만 [1,3,4] 의 6원처럼 아닌 체계가 있다.
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
    # DP 표. best[x] = x원을 만드는 최소 동전 수(INF = 불가능). 시간 O(amount x |coins|), 공간 O(amount). 불변식: x 를 채울 때 best[0..x-1] 은 이미 최적.
    best = [0] + [INF] * amount
    # last[x] = x원의 최적 해에서 마지막에 쓴 동전(되짚어 목록을 복원하는 용도).
    last = [0] * (amount + 1)
    for x in range(1, amount + 1):
        for c in sorted(coins, reverse=True):
            # 점화식 best[x] = min_c (best[x-c] + 1). 큰 동전부터 보고 '<' 로만 갱신하므로 동수일 때 큰 동전이 남는다.
            if c <= x and best[x - c] + 1 < best[x]:
                best[x] = best[x - c] + 1
                last[x] = c
    # last 를 따라 amount 에서 0 까지 내려오며 동전 목록 복원. best[amount] 가 INF 면 last 가 0 이라 무한 루프가 되는 함정(1원이 없는 체계는 넘기지 말 것).
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
        # 합칠 때마다 1 줄어든다.
        self.count = n  # 연결 요소 수

    def find(self, x):
        self.c.find_calls += 1
        # 불변식: parent 를 따라가면 항상 뿌리에 닿는다. 경로 압축 + 랭크 합치기를 둘 다 쓰면 연산당 분할상환 O(알파(n)) (알파 = 역아커만 함수, 사실상 상수).
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
            self.c.find_steps += 1
        if self.compress:
            while self.parent[x] != root:
                # 경로 위 모든 점을 뿌리에 직접 붙인다. 튜플 대입 순서에 주의: 오른쪽이 먼저 계산돼 옛 부모를 x 로 옮긴다.
                self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.by_rank:
            # 랭크 = 트리 높이의 상한(압축 때문에 실제 높이보다 클 수 있다). 낮은 쪽을 높은 쪽 밑에 붙인다.
            if self.rank[ra] < self.rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        else:
            # (by_rank 를 끈 순진한 방식: 최악에 트리가 사슬이 되어 find 가 O(n)).
            self.parent[ra] = rb  # 늘 앞 집합의 뿌리를 뒤 집합 밑에 붙인다
        self.c.unions += 1
        self.count -= 1
        return True


def naive_components(n, pairs):
    """대조용: 집합 목록을 직접 합치는 순진한 방법. 각 점의 대표(가장 작은 번호)를 돌려준다."""
    # 정답 대조용. 합칠 때 작은 쪽/큰 쪽 구분 없이 옮겨서 느리지만 단순하다.
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
    # 연결되려면 최소 n-1 간선, 단순 그래프의 최대는 n(n-1)/2.
    assert n - 1 <= m <= n * (n - 1) // 2
    order = list(range(n))
    rng.shuffle(order)
    pairs = set()
    for i in range(1, n):
        # 앞서 놓인 점 중 하나에 새 점을 붙여 무작위 신장 트리를 만든다(연결을 보장).
        a, b = order[i], order[rng.randrange(i)]
        pairs.add((min(a, b), max(a, b)))
    while len(pairs) < m:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    pairs = sorted(pairs)
    rng.shuffle(pairs)
    # 간선 형식 (무게, u, v): 튜플 정렬이 곧 무게순이라 Kruskal 이 sorted 한 번으로 끝난다. wmax 가 작으면 같은 무게가 생겨 MST 가 유일하지 않을 수 있다.
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
    # Kruskal 시간 O(E log E) (정렬이 지배, 합집합 연산은 거의 O(E)). 불변식: chosen 은 항상 순환 없는 숲이며 어떤 MST 에 포함된다(절단 성질).
    uf = UnionFind(n, compress=compress, by_rank=by_rank, counter=c)
    chosen = []
    total = 0
    # 같은 무게 간선은 (u, v) 순. 무게가 같으면 MST 가 여러 개일 수 있다.
    for w, u, v in sorted(edges):
        c.edge_scan += 1
        ok = uf.union(u, v)
        if ok:
            chosen.append((w, u, v))
            total += w
        if trace is not None:
            trace.append(((w, u, v), ok, [uf.find(i) for i in range(n)]))
        # n-1 개를 받으면 신장 트리 완성이라 남은 간선은 보지 않고 멈춘다(edge_scan 이 E 보다 작을 수 있는 이유).
        if len(chosen) == n - 1:
            break
    return total, chosen


def prim(n, edges, start=0, counter=None, adj=None):
    """한 점에서 시작한 조각에 '조각까지 가장 가까운 점'을 하나씩 붙인다. 이진 힙, 낡은 항목은 건너뛴다."""
    c = counter if counter is not None else Counter()
    adj = adj if adj is not None else adjacency(n, edges)
    # Prim 시간 O(E log V) (이진 힙, 힙 항목 최대 E + 1). 불변식: inside 는 항상 연결된 트리이고, 다음에 붙는 점은 inside 와 바깥을 잇는 간선 중 최소(절단 성질).
    inside = [False] * n
    key = [INF] * n
    key[start] = 0
    # 힙 원소 (키, 점, 부모). 키 = 조각까지 이어주는 간선 하나의 무게. 부모 -1 은 시작점.
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
            # (7장 다익스트라의 d(u)+w 와 달리 w 만 비교한다. 이 한 줄이 두 알고리즘의 차이.)
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
            # 여기서는 키가 출발점에서의 누적 거리 d + w: 다익스트라와 같아서 MST 가 아니라 최단경로 트리가 나온다.
            if not inside[v] and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v, u, w))
    return sum(e[0] for e in chosen), chosen


def is_spanning_tree(n, chosen):
    # 신장 트리는 간선이 정확히 n-1 개이고 순환이 없어야 한다.
    if len(chosen) != n - 1:
        return False
    uf = UnionFind(n)
    # union 이 False(이미 같은 집합) 면 순환이므로 트리가 아니다.
    return all(uf.union(u, v) for _, u, v in chosen)


def brute_force_mst(n, edges):
    """(n-1)개 간선 조합을 모두 보고 신장 트리 중 가장 가벼운 것의 무게."""
    best = INF
    # 전수 탐색: C(E, n-1) 가지라 정점 6 이하 같은 작은 그래프에서만 쓴다.
    for comb in itertools.combinations(edges, n - 1):
        if is_spanning_tree(n, comb):
            best = min(best, sum(e[0] for e in comb))
    return best


def lightest_crossing(n, edges, side):
    """절단(side 에 든 점 / 나머지)을 가로지르는 간선 중 가장 가벼운 무게."""
    # 절단을 가로지르는 간선 = 한 끝만 side 에 있는 간선(XOR).
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
    # 무거운 것부터. 매트로이드(숲)에서는 이 탐욕이 최적 가중치를 보장하고, 매칭은 매트로이드가 아니라서 보장이 없다. 매번 independent 를 새로 계산하므로 느리다(작은 입력용).
    for e in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        if independent(n, chosen + [e]):
            chosen.append(e)
    return sum(e[0] for e in chosen), chosen


def brute_force_max_weight(n, edges, independent):
    best = 0
    # 모든 부분집합 2^E 가지(대조용, 간선 10개 안팎까지).
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
        # 흔한 실패: 겹침을 무시하고 큰 집합 k 개를 고른다. 겹치면 새로 덮는 것이 거의 없다.
        idx = sorted(range(len(sets)), key=lambda i: (-len(sets[i]), i))[:k]
        return covered(sets, idx), idx
    got = set()
    idx = []
    # 탐욕 한 번에 집합 수 x (원소 비교) 계산 -> 전체 O(k x 집합 수 x 집합 크기). 보장: 최적 덮기의 1 - ((k-1)/k)^k >= 1 - 1/e (약 63%) 이상.
    for _ in range(k):
        best_i, best_gain = None, -1
        for i, s in enumerate(sets):
            # 이미 고른 집합은 건너뜀(idx 는 리스트라 검사가 O(k)).
            if i in idx:
                continue
            c.gain_evals += 1
            g = len(s - got)
            # '>' 로만 갱신하므로 같은 이득이면 번호가 작은 집합이 남는다(문서의 동률 규칙).
            if g > best_gain:
                best_i, best_gain = i, g
        if best_i is None:
            break
        idx.append(best_i)
        got |= sets[best_i]
    return len(got), idx


def brute_force_coverage(sets, k):
    best, arg = -1, None
    # C(집합 수, k) 전수 탐색(대조용).
    for comb in itertools.combinations(range(len(sets)), min(k, len(sets))):
        v = covered(sets, comb)
        if v > best:
            best, arg = v, list(comb)
    return best, arg


def coverage_bound(k):
    """Nemhauser·Wolsey·Fisher(1978) 초록의 1 - ((k-1)/k)^k 를 정확한 분수로."""
    # k=1 이면 1, k=2 이면 3/4, k 가 커지면 1-1/e 에 아래에서 가까워진다. Fraction 으로 반올림 오차 없이 비교.
    return 1 - Fraction(k - 1, k) ** k


def random_sets(n_sets, universe, p, rng):
    return [frozenset(e for e in range(universe) if rng.random() < p) for _ in range(n_sets)]
