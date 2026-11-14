"""7장. 병목은 어느 연결인가: 최대유량·최소절단.

표준 라이브러리만 쓴다. 단위는 건/시간, 용량은 모두 예시용 가상값이다.
"""
from collections import deque
from itertools import combinations

NODES = ("S", "A", "B", "C", "D", "T")
LABEL = {"S": "주문 접수", "A": "결제 검증", "B": "재고 확인", "C": "창고 1 출고", "D": "창고 2 출고", "T": "배송 인계"}
EDGES = {
    ("S", "A"): 7, ("S", "B"): 6,
    ("A", "C"): 4, ("A", "D"): 3,
    ("B", "C"): 2, ("B", "D"): 4,
    ("C", "T"): 5, ("D", "T"): 6,
}


def edmonds_karp(edges, s="S", t="T"):
    """최대유량. 반환: (값, 간선별 흐름 dict, 증가경로 목록, 잔여 용량 dict)."""
    res = {}
    adj = {}
    for (u, v), c in edges.items():
        res[(u, v)] = res.get((u, v), 0) + c
        res.setdefault((v, u), 0)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    total = 0
    paths = []
    while True:
        prev = {s: None}
        q = deque([s])
        while q and t not in prev:
            u = q.popleft()
            for v in sorted(adj.get(u, [])):
                if v not in prev and res[(u, v)] > 0:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            break
        path = [t]
        while path[-1] != s:
            path.append(prev[path[-1]])
        path.reverse()
        push = min(res[(a, b)] for a, b in zip(path, path[1:]))
        for a, b in zip(path, path[1:]):
            res[(a, b)] -= push
            res[(b, a)] += push
        total += push
        paths.append((path, push))
    flow = {e: edges[e] - res[e] for e in edges}
    return total, flow, paths, res


def min_cut_from_residual(edges, res, s="S"):
    """잔여 그래프에서 s로 갈 수 있는 점 집합 = 최소절단의 s쪽."""
    seen = {s}
    q = deque([s])
    while q:
        u = q.popleft()
        for (a, b), c in res.items():
            if a == u and c > 0 and b not in seen:
                seen.add(b)
                q.append(b)
    cut = [e for e in edges if e[0] in seen and e[1] not in seen]
    return seen, cut


def all_cuts(edges, nodes=NODES, s="S", t="T"):
    """s는 포함, t는 제외하는 모든 점 집합의 절단 용량(대조용 전수 열거)."""
    mid = [n for n in nodes if n not in (s, t)]
    out = []
    for k in range(len(mid) + 1):
        for chosen in combinations(mid, k):
            side = {s, *chosen}
            cap = sum(c for (u, v), c in edges.items() if u in side and v not in side)
            out.append((cap, tuple(sorted(side))))
    return sorted(out)


def cut_edges(edges, side):
    return [e for e in edges if e[0] in side and e[1] not in side]


def upgrade_table(edges, amount=2):
    """간선 하나씩 용량을 amount 늘렸을 때 최대유량이 얼마나 느는지."""
    base = edmonds_karp(edges)[0]
    rows = []
    for e in edges:
        e2 = dict(edges)
        e2[e] += amount
        rows.append((e, edges[e], edmonds_karp(e2)[0] - base))
    return base, rows


# 창고 2 출고→배송 인계 용량을 7로 올린 변형: 최소절단이 두 개가 된다.
TIE_EDGES = dict(EDGES)
TIE_EDGES[("D", "T")] = 7


def min_cuts(edges):
    """용량이 최소인 절단을 모두 돌려준다(전수 열거)."""
    cuts = all_cuts(edges)
    return [(cap, cut_edges(edges, side)) for cap, side in cuts if cap == cuts[0][0]]


def main():
    value, flow, paths, res = edmonds_karp(EDGES)
    print("최대유량:", value)
    for p, f in paths:
        print(" 증가경로", "→".join(p), f)
    print("간선별 흐름:", flow)
    side, cut = min_cut_from_residual(EDGES, res)
    print("잔여 그래프의 최소절단 s쪽:", sorted(side), "끊는 간선:", cut, "용량", sum(EDGES[e] for e in cut))
    cuts = all_cuts(EDGES)
    print("모든 절단(용량, s쪽):")
    for c in cuts:
        print(" ", c, cut_edges(EDGES, c[1]))
    print("최소 용량 절단 개수:", sum(1 for c in cuts if c[0] == cuts[0][0]))
    base, rows = upgrade_table(EDGES, 2)
    print("간선 하나를 +2 했을 때 최대유량 증가 (기준", base, ")")
    for e, c, d in rows:
        print(" ", e, c, "+", d)
    both = dict(EDGES)
    both[("C", "T")] += 2
    both[("D", "T")] += 2
    print("두 병목 간선을 함께 +2:", edmonds_karp(both)[0])
    print("--- 변형(D→T=7): 최소절단이 둘 ---")
    print("최대유량:", edmonds_karp(TIE_EDGES)[0])
    for cap, es in min_cuts(TIE_EDGES):
        print(" 최소절단", cap, es)
    base, rows = upgrade_table(TIE_EDGES, 2)
    for e, c, d in rows:
        print(" ", e, c, "+", d)


if __name__ == "__main__":
    main()
