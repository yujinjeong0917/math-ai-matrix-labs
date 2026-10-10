"""알고리즘 6장 최소 구현: 그래프를 빠짐없이 도는 법(BFS, DFS)과 방문 표시가 없을 때의 실패, 오일러 판정.

표준 라이브러리만 쓴다(NumPy 대조는 아래 *_np 함수 두 개, 그리고 experiments.py 와 테스트에서만).

셈 규칙
    visits      : 정점 하나를 "들러서 이웃을 보기 시작"할 때마다 1.
                  방문 표시가 있으면 정점마다 많아야 1번이다. 표시가 없으면 같은 정점을 길마다 다시 센다.
    edge_checks : 들른 정점에서 나가는 간선 하나를 볼 때마다 1(이웃이 이미 표시됐든 아니든).
                  방문 표시가 있으면 방향 그래프는 정확히 E, 무방향 그래프는 정확히 2E(간선을 양쪽 끝에서 한 번씩)다.
    max_frontier: 아직 들르지 않았지만 줄을 선 정점 수(BFS 큐) 또는 DFS 스택 깊이의 최댓값.

색(위젯·불변식에서 쓰는 이름)
    흰색 : 아직 발견하지 못한 정점
    회색 : 발견해서 프런티어(큐·스택)에 들어 있지만, 이웃을 아직 다 보지 않은 정점
    검정 : 이웃을 모두 본 정점
    BFS 는 큐에 넣을 때 회색으로 칠한다(넣을 때 표시). 그래서 같은 정점이 큐에 두 번 들어가지 않는다.
    DFS(반복판)는 (정점, 다음에 볼 이웃 번호)를 스택에 쌓는 방식이라, 재귀판과 같은 순서로 돌고
    스택 깊이는 재귀 깊이와 같다. 정점을 스택에 올릴 때 회색, 이웃을 다 보고 내릴 때 검정이다.

들어 있는 것
    Counter, Graph(인접 리스트, 평행 간선 허용)
    bfs, dfs_iterative, dfs_recursive(RecursionError 재현용), dfs_no_visited(표시 없는 실패 재현용)
    reachable_count, check_bfs_distances, black_has_no_white_neighbor
    degree_parity, euler_check, euler_bruteforce, hierholzer
    diamond_chain, cycle_graph, path_graph, grid_graph, erdos_renyi, star_graph, binary_tree
    konigsberg, euler_15_bridges(Euler E53 §15 의 다리 15개)
    floyd_warshall_np, bfs_levels_np : NumPy 대조
    trace_bfs, trace_dfs, trace_no_visited : 위젯용 단계 기록
"""

import random
from collections import deque


class Counter:
    def __init__(self):
        self.visits = 0
        self.edge_checks = 0
        self.max_frontier = 0


def _c(counter):
    return counter if counter is not None else Counter()


class Graph:
    """정점 0..n-1, 인접 리스트. adj[u] = [(v, 간선 번호), ...]. 평행 간선(같은 두 정점 사이 여러 간선)을 허용한다."""

    def __init__(self, n, directed=False):
        self.n = n
        self.directed = directed
        self.adj = [[] for _ in range(n)]
        self.edges = []  # [(u, v)]

    def add_edge(self, u, v):
        eid = len(self.edges)
        self.edges.append((u, v))
        self.adj[u].append((v, eid))
        if not self.directed:
            self.adj[v].append((u, eid))
        return eid

    @property
    def m(self):
        return len(self.edges)

    def neighbors(self, u):
        return [v for v, _ in self.adj[u]]


# ---------------------------------------------------------------- 탐색 세 가지

def bfs(g, s, counter=None):
    """너비 우선 탐색. 반환 (dist, parent). 닿지 않는 정점의 dist 는 None.

    큐에 넣을 때 표시한다(dist 를 적는다). 그래서 정점마다 많아야 한 번 큐에 들어간다.
    """
    c = _c(counter)
    dist = [None] * g.n
    parent = [None] * g.n
    dist[s] = 0
    q = deque([s])
    c.max_frontier = max(c.max_frontier, 1)
    while q:
        u = q.popleft()
        c.visits += 1
        for v, _ in g.adj[u]:
            c.edge_checks += 1
            if dist[v] is None:  # 흰색이면
                dist[v] = dist[u] + 1  # 회색으로 칠하고
                parent[v] = u
                q.append(v)  # 줄 뒤에 세운다
        c.max_frontier = max(c.max_frontier, len(q))
    return dist, parent


def bfs_mark_on_pop(g, s, counter=None):
    """흔한 실패 재현: 큐에서 꺼낼 때 표시하는 BFS. 답(거리)은 맞지만 같은 정점이 큐에 여러 번 들어간다.
    반환 (dist, pushes). pushes = 큐에 넣은 총 횟수."""
    c = _c(counter)
    dist = [None] * g.n
    q = deque([(s, 0)])
    pushes = 1
    while q:
        u, d = q.popleft()
        if dist[u] is not None:
            continue
        dist[u] = d
        c.visits += 1
        for v, _ in g.adj[u]:
            c.edge_checks += 1
            if dist[v] is None:
                q.append((v, d + 1))
                pushes += 1
        if len(q) > c.max_frontier:
            c.max_frontier = len(q)
    return dist, pushes


def bfs_all(g, counter=None):
    """연결 요소가 여럿이어도 모든 정점을 덮는 BFS. 표시판(dist) 하나를 끝까지 같이 쓴다.
    반환 (dist, comp): comp[v] = 정점 v 가 속한 연결 요소 번호. 전체 일은 O(V + E)."""
    c = _c(counter)
    dist = [None] * g.n
    comp = [None] * g.n
    k = 0
    for s in range(g.n):
        if dist[s] is not None:
            continue
        dist[s] = 0
        comp[s] = k
        q = deque([s])
        while q:
            u = q.popleft()
            c.visits += 1
            for v, _ in g.adj[u]:
                c.edge_checks += 1
                if dist[v] is None:
                    dist[v] = dist[u] + 1
                    comp[v] = k
                    q.append(v)
            if len(q) > c.max_frontier:
                c.max_frontier = len(q)
        k += 1
    return dist, comp


def bfs_all_fresh_marks(g, counter=None):
    """흔한 실패 재현: 연결 요소마다 bfs(g, s)를 불러 표시판을 매번 새로 만든다.
    간선 검사는 bfs_all 과 같지만, 길이 V 인 배열을 연결 요소 수만큼 만들어서 O(V x 요소 수)가 된다."""
    c = _c(counter)
    seen = [False] * g.n
    comps = 0
    for s in range(g.n):
        if not seen[s]:
            comps += 1
            dist, _ = bfs(g, s, c)
            for v, d in enumerate(dist):
                if d is not None:
                    seen[v] = True
    return comps


def dfs_iterative(g, s, counter=None):
    """깊이 우선 탐색(반복판). 반환 (order, parent, finish).

    order  : 처음 발견한(회색이 된) 순서
    finish : 이웃을 모두 보고 검정이 된 순서
    스택에는 (정점, 다음에 볼 이웃 번호)를 쌓는다. 재귀판과 같은 순서이고 스택 깊이 = 재귀 깊이.
    """
    c = _c(counter)
    seen = [False] * g.n
    parent = [None] * g.n
    order, finish = [], []
    seen[s] = True
    order.append(s)
    c.visits += 1
    stack = [[s, 0]]
    c.max_frontier = max(c.max_frontier, 1)
    while stack:
        top = stack[-1]
        u, i = top
        if i < len(g.adj[u]):
            top[1] = i + 1
            v = g.adj[u][i][0]
            c.edge_checks += 1
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                order.append(v)
                c.visits += 1
                stack.append([v, 0])
                c.max_frontier = max(c.max_frontier, len(stack))
        else:
            finish.append(u)
            stack.pop()
    return order, parent, finish


def dfs_recursive(g, s, counter=None):
    """교과서식 재귀 DFS. 깊은 그래프에서 RecursionError 가 난다(2절 실패 3)."""
    c = _c(counter)
    seen = [False] * g.n
    order = []

    def rec(u, depth):
        seen[u] = True
        order.append(u)
        c.visits += 1
        c.max_frontier = max(c.max_frontier, depth)
        for v, _ in g.adj[u]:
            c.edge_checks += 1
            if not seen[v]:
                rec(v, depth + 1)

    rec(s, 1)
    return order


def dfs_no_visited(g, s, limit=10 ** 6, counter=None):
    """방문 표시 없이 '갈 수 있는 길을 모두 걷는' 탐색. 반환 {'visits', 'edge_checks', 'capped', 'max_frontier'}.

    정점에 도착할 때마다 visits 를 1 올리고 나가는 간선을 모두 스택에 쌓는다. 순환이 있으면 끝나지 않으므로
    visits 가 limit 에 닿으면 멈추고 capped=True 를 돌려준다. capped 인 숫자는 '실제 횟수'가 아니라 '여기서 끊었다'는 뜻이다.
    재귀를 쓰지 않으니 깊이 때문에 터지지는 않는다.
    """
    c = _c(counter)
    stack = [s]
    capped = False
    while stack:
        u = stack.pop()
        c.visits += 1
        if c.visits >= limit:
            capped = True
            break
        for v, _ in reversed(g.adj[u]):  # reversed: 표시판 DFS 와 같은 순서로 보게
            c.edge_checks += 1
            stack.append(v)
        if len(stack) > c.max_frontier:
            c.max_frontier = len(stack)
    return {"visits": c.visits, "edge_checks": c.edge_checks, "capped": capped, "max_frontier": c.max_frontier}


def reachable_count(g, s):
    return sum(d is not None for d in bfs(g, s)[0])


# ---------------------------------------------------------------- 불변식 검사

def check_bfs_distances(g, dist):
    """[(이름, 성립 여부, 설명)]. 무방향은 |d(u)-d(v)| <= 1, 방향은 d(v) <= d(u)+1."""
    bad = []
    for u, v in g.edges:
        du, dv = dist[u], dist[v]
        if du is None and dv is None:
            continue
        if g.directed:
            if du is not None and (dv is None or dv > du + 1):
                bad.append((u, v))
        else:
            if du is None or dv is None or abs(du - dv) > 1:
                bad.append((u, v))
    name = "간선 (u,v) 마다 d(v) <= d(u)+1" if g.directed else "간선 (u,v) 마다 |d(u)-d(v)| <= 1"
    return [(name, not bad, f"어긋난 간선 {len(bad)}개")]


def black_has_no_white_neighbor(g, color):
    """color[v] in {'white','gray','black'}. 검정 정점의 (나가는) 이웃 중 흰색이 없으면 True."""
    for u in range(g.n):
        if color[u] == "black":
            for v, _ in g.adj[u]:
                if color[v] == "white":
                    return False
    return True


# ---------------------------------------------------------------- 오일러

def degree_parity(g):
    """무방향 그래프의 차수(평행 간선은 따로, 고리 u-u 는 2로 센다)."""
    deg = [0] * g.n
    for u, v in g.edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _edges_connected(g):
    """간선이 하나라도 닿은 정점들이 서로 이어져 있나(간선 없는 외톨이 정점은 무시)."""
    deg = degree_parity(g)
    live = [v for v in range(g.n) if deg[v] > 0]
    if not live:
        return True
    dist = bfs(g, live[0])[0]
    return all(dist[v] is not None for v in live)


def euler_check(g):
    """무방향 (다중)그래프. 반환 ('circuit' | 'path' | 'none', 홀수 차수 정점 목록).

    Euler E53 §20 의 규칙: 홀수 개의 다리가 닿는 땅이 2개보다 많으면 불가능, 2개면 그중 하나에서 출발, 0개면 아무 데서나.
    Hierholzer(1873)는 그 역(이어져 있고 홀수 정점이 0개나 2개면 한 번에 그릴 수 있다)을 증명했다.
    """
    assert not g.directed
    deg = degree_parity(g)
    odd = [v for v in range(g.n) if deg[v] % 2]
    if not _edges_connected(g):
        return "none", odd
    if len(odd) == 0:
        return "circuit", odd
    if len(odd) == 2:
        return "path", odd
    return "none", odd


def euler_bruteforce(g, counter=None):
    """작은 그래프용 전수 탐색: 모든 출발점에서 안 쓴 간선만 골라 걷는 길을 다 해 본다.
    반환 ('circuit' | 'path' | 'none', 시도한 걸음 수). 간선 8개 안팎까지만 쓴다."""
    c = _c(counter)
    m = g.m
    if m == 0:
        return "circuit", 0
    used = [False] * m
    found = {"circuit": False, "path": False}

    def walk(start, u, k):
        if k == m:
            if u == start:
                found["circuit"] = True
            else:
                found["path"] = True
            return
        for v, e in g.adj[u]:
            if not used[e]:
                c.edge_checks += 1
                used[e] = True
                walk(start, v, k + 1)
                used[e] = False
                if found["circuit"]:
                    return

    deg = degree_parity(g)
    for s in range(g.n):
        if deg[s]:
            walk(s, s, 0)
            if found["circuit"]:
                break
    kind = "circuit" if found["circuit"] else ("path" if found["path"] else "none")
    return kind, c.edge_checks


def hierholzer(g, start=None):
    """오일러 경로/회로를 실제로 만든다(무방향). 반환: 지나는 (정점 목록, 간선 번호 목록) 또는 None.
    간선마다 한 번씩만 보므로 O(V + E)."""
    kind, odd = euler_check(g)
    if kind == "none":
        return None
    deg = degree_parity(g)
    if start is None:
        start = odd[0] if odd else next((v for v in range(g.n) if deg[v]), 0)
    used = [False] * g.m
    ptr = [0] * g.n
    stack = [(start, None)]
    vs, es = [], []
    while stack:
        u, e_in = stack[-1]
        while ptr[u] < len(g.adj[u]) and used[g.adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(g.adj[u]):
            stack.pop()
            vs.append(u)
            if e_in is not None:
                es.append(e_in)
        else:
            v, e = g.adj[u][ptr[u]]
            used[e] = True
            stack.append((v, e))
    vs.reverse()
    es.reverse()
    return vs, es


# ---------------------------------------------------------------- 그래프 만들기

def diamond_chain(k):
    """다이아몬드 k개를 이은 방향 그래프. 정점 3k+1, 간선 4k.

    번호: 마디 i 는 3i (i = 0..k), 다이아몬드 i(1..k)의 위·아래 정점은 3i-2, 3i-1.
    간선: 3(i-1) -> 3i-2, 3(i-1) -> 3i-1, 3i-2 -> 3i, 3i-1 -> 3i.
    시작(0)에서 끝(3k)까지 길은 2^k 개다. 방향이 있어서 되돌아가는 길은 없다.
    """
    g = Graph(3 * k + 1, directed=True)
    for i in range(1, k + 1):
        a, up, dn, b = 3 * (i - 1), 3 * i - 2, 3 * i - 1, 3 * i
        g.add_edge(a, up)
        g.add_edge(a, dn)
        g.add_edge(up, b)
        g.add_edge(dn, b)
    return g


def diamond_no_visited_formula(k):
    """표시 없는 DFS 가 다이아몬드 사슬에서 정점에 도착하는 횟수 = 시작에서 각 정점까지 길 수의 합.
    마디 i: 2^i, 위·아래 정점: 2^(i-1) 씩 -> (2^(k+1)-1) + (2^(k+1)-2) = 2^(k+2) - 3."""
    return 2 ** (k + 2) - 3


def cycle_graph(n, directed=True):
    g = Graph(n, directed=directed)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def path_graph(n, directed=False):
    g = Graph(n, directed=directed)
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def grid_graph(r, c):
    """r x c 격자(무방향). 정점 번호 = 행 * c + 열. 이웃은 오른쪽, 아래쪽 순서로 넣는다."""
    g = Graph(r * c)
    for i in range(r):
        for j in range(c):
            v = i * c + j
            if j + 1 < c:
                g.add_edge(v, v + 1)
            if i + 1 < r:
                g.add_edge(v, v + c)
    return g


def erdos_renyi(n, p, rng):
    """G(n, p): 두 정점마다 확률 p 로 간선을 잇는다(무방향). 큰 n 에서도 빠르게, 간선 사이 간격을 기하분포로 건너뛴다."""
    import math
    g = Graph(n)
    if p <= 0:
        return g
    if p >= 1:
        for u in range(n):
            for v in range(u + 1, n):
                g.add_edge(u, v)
        return g
    lp = math.log(1 - p)
    v, w = 1, -1
    while v < n:
        r = rng.random()
        w = w + 1 + int(math.log(1 - r) / lp)
        while w >= v and v < n:
            w -= v
            v += 1
        if v < n:
            g.add_edge(v, w)
    return g


def star_graph(n):
    g = Graph(n)
    for i in range(1, n):
        g.add_edge(0, i)
    return g


def binary_tree(n):
    g = Graph(n)
    for i in range(1, n):
        g.add_edge((i - 1) // 2, i)
    return g


# Königsberg: Euler E53 그림 1. A = 섬(Kneiphof), B·C = 강 양쪽 기슭, D = 두 강줄기 사이 땅.
# 다리 a, b: A-B / c, d: A-C / e: A-D / f: B-D / g: C-D  (E53 §6, §9 의 짝과 차수 A5, B3, C3, D3 에 맞춤)
KONIG_NAMES = ["A", "B", "C", "D"]
KONIG_BRIDGES = [("a", 0, 1), ("b", 0, 1), ("c", 0, 2), ("d", 0, 2), ("e", 0, 3), ("f", 1, 3), ("g", 2, 3)]


def konigsberg():
    g = Graph(4)
    for _, u, v in KONIG_BRIDGES:
        g.add_edge(u, v)
    return g


# Euler E53 §15 의 두 번째 예(땅 A..F, 다리 15개). 본문이 적은 길 "EaFbBcFdAeFfCgAhCiDkAmEnApBoElD"
# 에서 대문자 사이 소문자가 지난 다리다. 이 길을 그대로 읽어 다리의 양 끝을 정한다.
EULER15_ROUTE = "EaFbBcFdAeFfCgAhCiDkAmEnApBoElD"


def euler_15_bridges():
    """반환 (Graph, 땅 이름 목록, 다리 이름 목록, 길(땅 번호), 길(간선 번호))."""
    names = list("ABCDEF")
    idx = {ch: i for i, ch in enumerate(names)}
    route = EULER15_ROUTE
    lands = [idx[ch] for ch in route[0::2]]
    bridges = list(route[1::2])
    g = Graph(len(names))
    eids = []
    for i, b in enumerate(bridges):
        eids.append(g.add_edge(lands[i], lands[i + 1]))
    return g, names, bridges, lands, eids


# ---------------------------------------------------------------- NumPy 대조

def floyd_warshall_np(g):
    """간선 길이 1 인 Floyd-Warshall. 반환 (n, n) float 배열, 닿지 않으면 inf."""
    import numpy as np
    n = g.n
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0.0)
    for u, v in g.edges:
        if u != v:
            D[u, v] = 1.0
            if not g.directed:
                D[v, u] = 1.0
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def bfs_levels_np(g, s):
    """한 층씩 넓히는 BFS 를 불리언 배열로: 다음 층 = (지금 층의 이웃) 중 아직 거리가 없는 정점."""
    import numpy as np
    n = g.n
    src = np.array([u for u, v in g.edges] + ([v for u, v in g.edges] if not g.directed else []), dtype=np.int64)
    dst = np.array([v for u, v in g.edges] + ([u for u, v in g.edges] if not g.directed else []), dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int64)
    dist[s] = 0
    frontier = np.zeros(n, dtype=bool)
    frontier[s] = True
    d = 0
    while frontier.any():
        nxt = np.zeros(n, dtype=bool)
        nxt[dst[frontier[src]]] = True
        nxt &= dist < 0
        d += 1
        dist[nxt] = d
        frontier = nxt
    return dist


# ---------------------------------------------------------------- 위젯용 단계 기록

def _colors_snapshot(color):
    return "".join({"white": "w", "gray": "g", "black": "b"}[x] for x in color)


def trace_bfs(g, s):
    """단계마다 [설명, 색 문자열(w/g/b), 큐(정점 목록), 큐의 거리 목록, dist, 누적 방문, 누적 간선 검사,
    큐 불변식 성립, 검정-흰색 불변식 성립, 지금 보는 간선(u, v) 또는 None]."""
    color = ["white"] * g.n
    dist = [None] * g.n
    steps = []
    vis = chk = 0

    def snap(msg, q, edge=None):
        qd = [dist[x] for x in q]
        ok_q = all(qd[i] <= qd[i + 1] for i in range(len(qd) - 1)) and (not qd or qd[-1] - qd[0] <= 1)
        steps.append([msg, _colors_snapshot(color), list(q), qd, list(dist), vis, chk, ok_q,
                      black_has_no_white_neighbor(g, color), edge])

    q = deque()
    snap("시작 전: 모든 정점이 흰색", q)
    color[s] = "gray"
    dist[s] = 0
    q.append(s)
    snap(f"출발점 {s}: 회색으로 칠하고 줄에 세움(거리 0)", q)
    while q:
        u = q.popleft()
        vis += 1
        snap(f"줄 맨 앞에서 {u} 꺼냄(거리 {dist[u]})", q)
        for v, _ in g.adj[u]:
            chk += 1
            if color[v] == "white":
                color[v] = "gray"
                dist[v] = dist[u] + 1
                q.append(v)
                snap(f"간선 {u}-{v}: {v}이(가) 흰색이라 회색으로 칠하고 줄 뒤에 세움(거리 {dist[v]})", q, [u, v])
            else:
                snap(f"간선 {u}-{v}: {v}에 이미 표시가 있어 건너뜀", q, [u, v])
        color[u] = "black"
        snap(f"정점 {u}: 이웃을 다 봐서 검정", q)
    return {"start": s, "steps": steps}


def trace_dfs(g, s):
    """trace_bfs 와 같은 칸. 큐 자리에 스택(아래 -> 위), 큐 불변식 자리는 항상 None."""
    color = ["white"] * g.n
    steps = []
    vis = chk = 0

    def snap(msg, stack, edge=None):
        steps.append([msg, _colors_snapshot(color), [x[0] for x in stack], None, None, vis, chk, None,
                      black_has_no_white_neighbor(g, color), edge])

    stack = []
    snap("시작 전: 모든 정점이 흰색", stack)
    color[s] = "gray"
    vis += 1
    stack.append([s, 0])
    snap(f"출발점 {s}: 회색으로 칠하고 스택에 올림", stack)
    while stack:
        top = stack[-1]
        u, i = top
        if i < len(g.adj[u]):
            top[1] = i + 1
            v = g.adj[u][i][0]
            chk += 1
            if color[v] == "white":
                color[v] = "gray"
                vis += 1
                stack.append([v, 0])
                snap(f"간선 {u}-{v}: {v}이(가) 흰색이라 회색으로 칠하고 스택에 올림(더 깊이)", stack, [u, v])
            else:
                snap(f"간선 {u}-{v}: {v}에 이미 표시가 있어 건너뜀", stack, [u, v])
        else:
            color[u] = "black"
            stack.pop()
            snap(f"정점 {u}: 이웃을 다 봐서 검정, 스택에서 내림(되돌아가기)", stack)
    return {"start": s, "steps": steps}


def trace_no_visited(g, s, limit=200):
    """표시 없는 DFS. 단계마다 [설명, 정점별 누적 도착 횟수, 스택, 누적 방문, 누적 간선 검사]."""
    counts = [0] * g.n
    steps = [["시작 전", list(counts), [s], 0, 0]]
    stack = [s]
    vis = chk = 0
    while stack and vis < limit:
        u = stack.pop()
        vis += 1
        counts[u] += 1
        for v, _ in reversed(g.adj[u]):
            chk += 1
            stack.append(v)
        steps.append([f"정점 {u}에 도착({counts[u]}번째)", list(counts), list(stack), vis, chk])
    return {"start": s, "steps": steps, "capped": bool(stack)}
