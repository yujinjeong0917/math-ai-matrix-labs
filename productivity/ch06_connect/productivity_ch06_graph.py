"""6장 등록부를 방향 그래프로 본다: 노드 = 정보가 놓인 위치, 간선 = 원본 -> 참조 방향.

- validate(): 항목마다 원본이 하나인지(들어오는 간선이 0인 노드가 원본 하나뿐), 모든 노드가 값을 한 곳에서만
  받는지(들어오는 간선 1개 이하), 고리(양방향 동기화)가 없는지, 원본에서 모든 위치에 닿는지 본다.
- 한 위치가 바뀌면 어디까지 퍼지나(도달 집합)를 세 방식으로 구한다: 순수 파이썬 BFS, NumPy 불리언 행렬 거듭제곱,
  sqlite3 재귀 CTE. 서로 독립된 구현이라 결과가 같아야 한다(6절 대조).
- 사람 확인 간선 위험: 위치마다 원본에서 그 위치까지 가는 길에 있는 사람 확인 간선 수.
"""

import random
import sqlite3
from collections import defaultdict, deque

import numpy as np


def item_graph(item):
    """item = {"src": ..., "edges": [(a, b, m), ...]} -> (노드 목록, 간선 목록)."""
    nodes = [item["src"]]
    for a, b, _ in item["edges"]:
        for x in (a, b):
            if x not in nodes:
                nodes.append(x)
    return nodes, list(item["edges"])


def validate(items):
    """등록부 규칙 위반 목록. 비어 있으면 통과."""
    problems = []
    src_owner = {}
    for k, it in items.items():
        nodes, edges = item_graph(it)
        indeg = defaultdict(int)
        out = defaultdict(list)
        for a, b, m in edges:
            indeg[b] += 1
            out[a].append(b)
            if m not in ("ref", "human", "auto"):
                problems.append((k, f"모르는 갱신 방식: {m}"))
        roots = [n for n in nodes if indeg[n] == 0]
        if roots != [it["src"]]:
            problems.append((k, f"원본이 하나가 아님: {roots or '없음(고리)'}"))
        for n in nodes:
            if indeg[n] > 1:
                problems.append((k, f"{n}이 값을 {indeg[n]}곳에서 받음"))
        # 고리 찾기(색칠 DFS)
        color = {n: 0 for n in nodes}

        def dfs(u):
            color[u] = 1
            for v in out[u]:
                if color[v] == 1:
                    return True
                if color[v] == 0 and dfs(v):
                    return True
            color[u] = 2
            return False

        if any(color[n] == 0 and dfs(n) for n in nodes):
            problems.append((k, "고리가 있음(양방향 동기화 = 원본이 둘)"))
        seen = reach_bfs(edges, it["src"])
        missing = [n for n in nodes if n not in seen]
        if missing:
            problems.append((k, f"원본에서 닿지 않는 위치: {missing}"))
        if it["src"] in src_owner:
            problems.append((k, f"원본 위치가 {src_owner[it['src']]}와 겹침"))
        src_owner[it["src"]] = k
    return problems


# ------------------------------------------------------------------ 도달 집합 세 방식
def reach_bfs(edges, start):
    """start에서 간선을 따라 닿는 모든 노드(start 포함)."""
    out = defaultdict(list)
    for a, b, *_ in edges:
        out[a].append(b)
    seen, q = {start}, deque([start])
    while q:
        u = q.popleft()
        for v in out[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def reach_numpy(edges, nodes):
    """모든 노드 쌍의 도달 여부 행렬 R. R[i, j] = i에서 j에 닿나(자기 자신 포함).

    R0 = I + A 에서 시작해 R <- (R @ R) > 0 을 바뀌지 않을 때까지 반복한다(제곱할 때마다 길이 두 배).
    """
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    R = np.eye(n, dtype=np.int64)
    for a, b, *_ in edges:
        R[idx[a], idx[b]] = 1
    while True:
        R2 = ((R @ R) > 0).astype(np.int64)
        if np.array_equal(R2, R):
            return R.astype(bool), idx
        R = R2


def reach_sql(edges, start):
    """sqlite3 재귀 CTE로 같은 도달 집합."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE e(a TEXT, b TEXT)")
    con.executemany("INSERT INTO e VALUES (?, ?)", [(a, b) for a, b, *_ in edges])
    rows = con.execute(
        """WITH RECURSIVE r(n) AS (SELECT ? UNION SELECT e.b FROM e JOIN r ON e.a = r.n)
           SELECT n FROM r""", (start,)).fetchall()
    con.close()
    return {x for (x,) in rows}


def human_hops(item):
    """위치마다 원본에서 그 위치까지 가는 길에 있는 사람 확인 간선 수와 자동화 간선 수."""
    nodes, edges = item_graph(item)
    parent = {b: (a, m) for a, b, m in edges}
    out = {}
    for n in nodes:
        h = a_ = 0
        cur = n
        guard = 0
        while cur in parent and guard <= len(nodes):
            a, m = parent[cur]
            h += m == "human"
            a_ += m == "auto"
            cur = a
            guard += 1
        out[n] = {"human": h, "auto": a_}
    return out


def expected_mismatch(item, q, auto_alive=True):
    """원본이 한 번 바뀐 뒤 어긋나 있을 위치 수의 기댓값과, 하나라도 어긋날 확률.

    사람 확인 간선마다 독립으로 확률 q로 옮긴다고 두면, 위치 n이 맞을 확률은 q^(n까지의 사람 확인 간선 수).
    자동화 간선은 살아 있으면 1, 죽었으면 0.
    """
    nodes, edges = item_graph(item)
    out = defaultdict(list)
    for a, b, m in edges:
        out[a].append((b, m))
    exp_wrong = 0.0
    # 사람 확인 간선 하나가 실패하면 그 아래가 통째로 틀린다. 모든 간선 결과의 조합을 다 셀 필요 없이,
    # 위치별로 "그 위치까지의 길이 모두 성공할 확률"만 곱하면 된다.
    hops = human_hops(item)
    for n in nodes:
        if n == item["src"]:
            continue
        p_ok = q ** hops[n]["human"] * (1.0 if (auto_alive or hops[n]["auto"] == 0) else 0.0)
        exp_wrong += 1 - p_ok
    # 하나라도 어긋날 확률 = 1 - (모든 사람 확인 간선 성공 확률). 자동화가 죽어 있고 자동화 간선이 있으면 1.
    n_h = sum(m == "human" for _, _, m in edges)
    n_a = sum(m == "auto" for _, _, m in edges)
    p_any = 1 - q ** n_h * (1.0 if (auto_alive or n_a == 0) else 0.0)
    return exp_wrong, p_any


def random_registry(rng, n_items=None):
    """무작위 등록부(나무 모양 항목 여러 개). 대응 구현 대조용."""
    n_items = n_items or rng.randint(1, 6)
    items = {}
    for k in range(n_items):
        size = rng.randint(1, 8)
        nodes = [f"i{k}:n{j}" for j in range(size)]
        edges = []
        for j in range(1, size):
            edges.append((nodes[rng.randrange(j)], nodes[j], rng.choice(["ref", "human", "auto"])))
        items[f"I{k}"] = {"src": nodes[0], "edges": edges}
    return items


def correspondence(n_cases=500, seed=6):
    """무작위 등록부 n_cases개에서 BFS / NumPy / SQL 세 방식이 모든 시작점에서 같은 도달 집합을 내는지."""
    rng = random.Random(seed)
    diff = 0
    starts = 0
    for _ in range(n_cases):
        items = random_registry(rng)
        edges = [e for it in items.values() for e in it["edges"]]
        nodes = []
        for it in items.values():
            for x in item_graph(it)[0]:
                nodes.append(x)
        R, idx = reach_numpy(edges, nodes)
        for s in nodes:
            a = reach_bfs(edges, s)
            b = {nodes[j] for j in np.flatnonzero(R[idx[s]])}
            c = reach_sql(edges, s)
            starts += 1
            if not (a == b == c):
                diff += 1
    return {"cases": n_cases, "starts": starts, "disagreements": diff}
