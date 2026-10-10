"""알고리즘 7장: 최단경로는 언제 틀리나.

이 파일 하나에 장의 구현이 모두 있다(다른 장과 import 를 공유하지 않는다).

- Graph: 가중치 있는 방향 그래프(인접 리스트)
- bfs_hops: 간선 개수로 잰 BFS(6장의 BFS를 길이 무시용으로 줄여 옮김). 길이가 다르면 틀린다.
- dijkstra: 확정 집합을 두는 다익스트라. 확정된 정점에는 다시 들어가지 않는다(Dijkstra 1959 의 집합 A).
  음수 간선이 있으면 틀릴 수 있다.
- dijkstra_reinsert: 확정 표시 없이 거리가 줄면 언제든 다시 넣는 변형(흔한 실패/대조용).
  음수 순환이 없으면 답은 맞지만 꺼내는 횟수가 늘 수 있다.
- bellman_ford: 모든 간선 이완을 라운드 단위로 반복. 한 라운드에 바뀐 게 없으면 멈춘다(조기 종료).
  V 번째 라운드에도 줄면 출발점에서 닿는 음수 순환이 있다.
- a_star: 4방향 격자(칸 사이 비용 1). Hart·Nilsson·Raphael(1968)의 Step 4 처럼 닫힌 칸도 f 가
  줄면 다시 연다. 꺼내는 순서는 (f, h, 넣은 순서). 목표를 꺼내면 멈춘다.

모든 함수는 선택 인자 counter(Counter)에 이완(간선 검사)·꺼내기·넣기 횟수를 센다.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field

INF = math.inf


@dataclass
class Counter:
    relax: int = 0      # 간선 하나를 보고 d(u)+w 와 d(v)를 비교한 횟수
    improve: int = 0    # 그 비교에서 d(v)가 실제로 줄어든 횟수
    pop: int = 0        # 우선순위 큐에서 꺼낸 횟수(낡은 항목 포함)
    push: int = 0       # 우선순위 큐에 넣은 횟수
    expand: int = 0     # 이웃을 둘러본(확장한) 정점 수
    rounds: int = 0     # Bellman-Ford 라운드 수


class NegativeCycle(Exception):
    """출발점에서 닿는 음수 순환이 있다."""


@dataclass
class Graph:
    n: int
    adj: list = field(default_factory=list)

    def __post_init__(self):
        if not self.adj:
            self.adj = [[] for _ in range(self.n)]

    def add(self, u, v, w):
        self.adj[u].append((v, w))

    def edges(self):
        for u in range(self.n):
            for v, w in self.adj[u]:
                yield u, v, w

    @property
    def m(self):
        return sum(len(a) for a in self.adj)


def from_edges(n, edges):
    g = Graph(n)
    for u, v, w in edges:
        g.add(u, v, w)
    return g


# ---------------------------------------------------------------- BFS(간선 개수)
def bfs_hops(g, s):
    """간선 개수가 가장 적은 경로를 찾고, 그 경로의 실제 길이 합을 돌려준다."""
    hops = [None] * g.n
    cost = [INF] * g.n
    hops[s] = 0
    cost[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v, w in g.adj[u]:
            if hops[v] is None:
                hops[v] = hops[u] + 1
                cost[v] = cost[u] + w
                q.append(v)
    return hops, cost


# ---------------------------------------------------------------- Dijkstra
def dijkstra(g, s, counter=None, trace=None):
    """확정 집합을 두는 다익스트라. trace 가 리스트면 단계 기록을 덧붙인다."""
    c = counter if counter is not None else Counter()
    dist = [INF] * g.n
    parent = [None] * g.n
    done = [False] * g.n
    dist[s] = 0
    heap = [(0, s)]
    c.push += 1
    if trace is not None:
        trace.append(("start", s, None, list(dist), list(done), sorted(heap)))
    while heap:
        d, u = heapq.heappop(heap)
        c.pop += 1
        if done[u] or d > dist[u]:
            continue
        done[u] = True
        c.expand += 1
        if trace is not None:
            trace.append(("settle", u, None, list(dist), list(done), sorted(heap)))
        for v, w in g.adj[u]:
            c.relax += 1  # 확정된 정점으로 가는 간선도 '보기는' 하므로 센다
            if done[v]:
                if trace is not None:
                    trace.append(("skip", u, (v, w), list(dist), list(done), sorted(heap)))
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
                c.push += 1
                c.improve += 1
                if trace is not None:
                    trace.append(("improve", u, (v, w), list(dist), list(done), sorted(heap)))
            elif trace is not None:
                trace.append(("keep", u, (v, w), list(dist), list(done), sorted(heap)))
    return dist, parent


def dijkstra_reinsert(g, s, counter=None, pop_limit=None):
    """확정 표시 없이, 거리가 줄면 언제든 다시 넣는 변형. 낡은 항목만 건너뛴다."""
    c = counter if counter is not None else Counter()
    dist = [INF] * g.n
    parent = [None] * g.n
    dist[s] = 0
    heap = [(0, s)]
    c.push += 1
    while heap:
        d, u = heapq.heappop(heap)
        c.pop += 1
        if pop_limit is not None and c.pop > pop_limit:
            raise RuntimeError("pop limit")
        if d > dist[u]:
            continue
        c.expand += 1
        for v, w in g.adj[u]:
            c.relax += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
                c.push += 1
                c.improve += 1
    return dist, parent


# ---------------------------------------------------------------- Bellman-Ford
def bellman_ford(g, s, counter=None, early_stop=True):
    """모든 간선을 차례로 이완하는 라운드를 반복한다.

    early_stop=True 면 한 라운드에서 아무것도 줄지 않을 때 멈춘다. V-1 라운드 뒤에도
    V 번째 라운드에서 줄어들면 NegativeCycle 을 던진다.
    """
    c = counter if counter is not None else Counter()
    dist = [INF] * g.n
    parent = [None] * g.n
    dist[s] = 0
    edges = list(g.edges())
    for r in range(g.n):
        c.rounds += 1
        changed = False
        for u, v, w in edges:
            if dist[u] == INF:
                continue
            c.relax += 1
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                c.improve += 1
                changed = True
        if not changed and early_stop:
            return dist, parent
        if changed and r == g.n - 1:
            raise NegativeCycle()
    return dist, parent


# ---------------------------------------------------------------- 무작위 그래프
def random_digraph(n, m, rng, wmin=1, wmax=20):
    """서로 다른 (u,v) 쌍 m 개(자기 자신 제외)에 [wmin, wmax] 정수 길이."""
    pairs = set()
    while len(pairs) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            pairs.add((u, v))
    pairs = sorted(pairs)
    rng.shuffle(pairs)
    return [(u, v, rng.randint(wmin, wmax)) for u, v in pairs]


def has_negative_cycle(n, edges):
    """모든 정점에서 닿는 음수 순환이 있는지(가상 출발점에서 길이 0 간선)."""
    dist = [0] * n
    for r in range(n):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            return False
    return True


def random_negative_graph(n, m, neg_ratio, rng, wmax=20, neg_min=-5, max_tries=100000):
    """음수 간선이 정확히 round(neg_ratio*m) 개인, 음수 순환 없는 그래프를 거절 표집으로 만든다.

    양수 간선은 [1, wmax], 음수 간선은 [neg_min, -1] 정수. 돌려주는 값: (edges, 거절 횟수).
    """
    k = round(neg_ratio * m)
    for tries in range(max_tries):
        edges = random_digraph(n, m, rng, 1, wmax)
        idx = rng.sample(range(m), k)
        for i in idx:
            u, v, _ = edges[i]
            edges[i] = (u, v, rng.randint(neg_min, -1))
        if not has_negative_cycle(n, edges):
            return edges, tries
    raise RuntimeError("음수 순환 없는 그래프를 못 만듦")


def reachable(g, s):
    seen = [False] * g.n
    seen[s] = True
    st = [s]
    while st:
        u = st.pop()
        for v, _ in g.adj[u]:
            if not seen[v]:
                seen[v] = True
                st.append(v)
    return seen


# ---------------------------------------------------------------- 격자와 A*
def random_grid(rows, cols, p_block, rng, keep=()):
    grid = [[rng.random() < p_block for _ in range(cols)] for _ in range(rows)]
    for r, c in keep:
        grid[r][c] = False
    return grid  # True 가 장애물


def grid_from_strings(lines):
    """'#' 장애물, '.' 빈칸, 'S','T' 출발·목표."""
    grid, s, t = [], None, None
    for r, line in enumerate(lines):
        row = []
        for c, ch in enumerate(line):
            row.append(ch == "#")
            if ch == "S":
                s = (r, c)
            if ch == "T":
                t = (r, c)
        grid.append(row)
    return grid, s, t


def neighbors(grid, cell):
    r, c = cell
    R, C = len(grid), len(grid[0])
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and not grid[nr][nc]:
            yield (nr, nc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def zero(a, b):
    return 0


def scaled(h, k):
    def hk(a, b):
        return k * h(a, b)
    hk.__name__ = f"{k}x{h.__name__}"
    return hk


def a_star(grid, s, t, h, counter=None, trace=None):
    """4방향 격자 A*. 꺼내는 순서 (f, h, 넣은 순서). 닫힌 칸도 g 가 줄면 다시 연다.

    돌려주는 값: (경로 비용 또는 None, 경로 칸 목록, 확장 순서 목록)
    """
    c = counter if counter is not None else Counter()
    g = {s: 0}
    parent = {s: None}
    closed = set()
    seq = 0
    hs = h(s, t)
    heap = [(hs, hs, seq, s)]
    c.push += 1
    order = []
    while heap:
        f, hv, _, u = heapq.heappop(heap)
        c.pop += 1
        if u in closed or f > g[u] + h(u, t):
            continue  # 낡은 항목
        closed.add(u)
        c.expand += 1
        order.append(u)
        if trace is not None:
            trace.append((u, g[u], h(u, t)))
        if u == t:
            path = []
            x = t
            while x is not None:
                path.append(x)
                x = parent[x]
            return g[t], path[::-1], order
        for v in neighbors(grid, u):
            c.relax += 1
            ng = g[u] + 1
            if ng < g.get(v, INF):
                g[v] = ng
                parent[v] = u
                closed.discard(v)  # 닫힌 칸 다시 열기(1968 Step 4)
                seq += 1
                hv2 = h(v, t)
                heapq.heappush(heap, (ng + hv2, hv2, seq, v))
                c.push += 1
                c.improve += 1
    return None, [], order


def grid_dijkstra_cost(grid, s, t):
    """검증용: h=0 A* 와 별도로, BFS 로 격자 최단 거리(칸 비용이 모두 1)."""
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            return dist[u]
        for v in neighbors(grid, u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return None


def grid_to_graph(grid):
    """격자를 Graph 로(칸 번호 r*C+c, 간선 길이 1). Dijkstra 와 A*(h=0) 대조용."""
    R, C = len(grid), len(grid[0])
    g = Graph(R * C)
    for r in range(R):
        for cc in range(C):
            if grid[r][cc]:
                continue
            for v in neighbors(grid, (r, cc)):
                g.add(r * C + cc, v[0] * C + v[1], 1)
    return g


def path_cost(g, path):
    total = 0
    for a, b in zip(path, path[1:]):
        total += min(w for v, w in g.adj[a] if v == b)
    return total


def check_settled_invariants(dist, done, heap, true_dist):
    """위젯·테스트용 불변식 두 개.

    1) 확정된 모든 라벨 <= 큐(프런티어)에 있는 모든 라벨
    2) 확정된 라벨 = 참 거리(Bellman-Ford 로 미리 구한 값)
    """
    settled = [dist[v] for v in range(len(dist)) if done[v]]
    frontier = [d for d, v in heap if not done[v] and d == dist[v]]
    inv1 = (not settled or not frontier) or max(settled) <= min(frontier)
    inv2 = all(dist[v] == true_dist[v] for v in range(len(dist)) if done[v])
    return inv1, inv2


def seeded(seed):
    return random.Random(seed)
