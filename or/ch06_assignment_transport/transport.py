"""6장. 주문을 어느 창고에 보낼까: 수송 문제를 최소비용 흐름으로 풀기.

표준 라이브러리만 쓴다. 단위는 '10개 묶음', 비용은 묶음당 가상 배송비(예시용)다.
금지 조합은 간선을 아예 만들지 않아서(None) 표현한다.
"""
from itertools import product

WAREHOUSES = ("W1", "W2", "W3")
REGIONS = ("A", "B", "C")
SUPPLY = {"W1": 4, "W2": 3, "W3": 3}      # 창고별 보낼 수 있는 묶음 수
DEMAND = {"A": 4, "B": 3, "C": 2}         # 지역별 주문 묶음 수
COST = {                                    # COST[창고][지역], None = 금지 조합
    "W1": {"A": 4, "B": 6, "C": 9},
    "W2": {"A": 2, "B": 3, "C": 4},
    "W3": {"A": 5, "B": 4, "C": None},
}


class MinCostFlow:
    """연속 최단경로(Bellman-Ford) 방식의 최소비용 흐름."""

    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])
        return (u, len(self.graph[u]) - 1)

    def flow_on(self, ref):
        u, i = ref
        v, cap, cost, rev = self.graph[u][i]
        return self.graph[v][rev][1]

    def run(self, s, t, want):
        flow = cost_total = 0
        while flow < want:
            dist = [None] * self.n
            dist[s] = 0
            prev = [None] * self.n
            for _ in range(self.n):
                changed = False
                for u in range(self.n):
                    if dist[u] is None:
                        continue
                    for i, (v, cap, c, _) in enumerate(self.graph[u]):
                        if cap > 0 and (dist[v] is None or dist[u] + c < dist[v]):
                            dist[v] = dist[u] + c
                            prev[v] = (u, i)
                            changed = True
                if not changed:
                    break
            if dist[t] is None:
                break
            push = want - flow
            v = t
            while v != s:
                u, i = prev[v]
                push = min(push, self.graph[u][i][1])
                v = u
            v = t
            while v != s:
                u, i = prev[v]
                self.graph[u][i][1] -= push
                self.graph[v][self.graph[u][i][3]][1] += push
                v = u
            flow += push
            cost_total += push * dist[t]
        return flow, cost_total


def solve_transport(supply, demand, cost):
    """반환: dict(feasible, shipped, flow, cost, plan). plan[(창고,지역)] = 묶음 수."""
    ws, rs = list(supply), list(demand)
    n = 2 + len(ws) + len(rs)
    s, t = 0, n - 1
    g = MinCostFlow(n)
    for i, w in enumerate(ws):
        g.add_edge(s, 1 + i, supply[w], 0)
    refs = {}
    for i, w in enumerate(ws):
        for j, r in enumerate(rs):
            if cost[w][r] is not None:
                refs[(w, r)] = g.add_edge(1 + i, 1 + len(ws) + j, min(supply[w], demand[r]), cost[w][r])
    for j, r in enumerate(rs):
        g.add_edge(1 + len(ws) + j, t, demand[r], 0)
    want = sum(demand.values())
    flow, total = g.run(s, t, want)
    plan = {k: g.flow_on(v) for k, v in refs.items() if g.flow_on(v) > 0}
    return {"feasible": flow == want, "shipped": flow, "cost": total, "plan": plan}


def plan_cost(plan, cost):
    return sum(q * cost[w][r] for (w, r), q in plan.items())


def plan_ok(plan, supply, demand, cost):
    for (w, r), q in plan.items():
        if cost[w][r] is None and q > 0:
            return False
    for w in supply:
        if sum(q for (ww, _), q in plan.items() if ww == w) > supply[w]:
            return False
    for r in demand:
        if sum(q for (_, rr), q in plan.items() if rr == r) != demand[r]:
            return False
    return True


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def brute_force(supply, demand, cost):
    """지역마다 창고별 몫을 모두 나눠 보는 전수 탐색(대조용). 반환: (최소비용, 최적 해 수, 가능한 계획 수)."""
    ws, rs = list(supply), list(demand)
    per_region = [list(_compositions(demand[r], len(ws))) for r in rs]
    best, n_best, n_ok = None, 0, 0
    for combo in product(*per_region):
        plan = {(w, r): combo[j][i] for j, r in enumerate(rs) for i, w in enumerate(ws) if combo[j][i]}
        if not plan_ok(plan, supply, demand, cost):
            continue
        n_ok += 1
        c = plan_cost(plan, cost)
        if best is None or c < best:
            best, n_best = c, 1
        elif c == best:
            n_best += 1
    return best, n_best, n_ok


def nearest_only(supply, demand, cost):
    """각 지역이 용량을 무시하고 가장 싼 창고에 전부 주문한다(깨지는 장면)."""
    plan = {}
    for r in demand:
        w = min((w for w in supply if cost[w][r] is not None), key=lambda w: cost[w][r])
        plan[(w, r)] = demand[r]
    load = {w: sum(q for (ww, _), q in plan.items() if ww == w) for w in supply}
    return plan, load, plan_cost(plan, cost)


def greedy_in_order(supply, demand, cost):
    """지역 A, B, C 순서로 남은 용량 안에서 가장 싼 창고부터 채운다."""
    left = dict(supply)
    plan = {}
    for r in demand:
        need = demand[r]
        for w in sorted((w for w in supply if cost[w][r] is not None), key=lambda w: cost[w][r]):
            q = min(need, left[w])
            if q:
                plan[(w, r)] = plan.get((w, r), 0) + q
                left[w] -= q
                need -= q
        if need:
            return None
    return plan, plan_cost(plan, cost)


def main():
    print("창고 용량", SUPPLY, "지역 주문", DEMAND, "(합 공급 %d, 합 주문 %d)" % (sum(SUPPLY.values()), sum(DEMAND.values())))
    plan, load, c = nearest_only(SUPPLY, DEMAND, COST)
    print("가장 싼 창고에 전부:", plan, "창고 부하", load, "비용", c)
    gp, gc = greedy_in_order(SUPPLY, DEMAND, COST)
    print("순서대로 탐욕:", gp, "비용", gc)
    res = solve_transport(SUPPLY, DEMAND, COST)
    print("최소비용 흐름:", res)
    print("전수 탐색 (최소비용, 최적 해 수, 가능한 계획 수):", brute_force(SUPPLY, DEMAND, COST))
    print("창고 W2 용량을 늘리면:")
    for cap in range(3, 7):
        s = dict(SUPPLY, W2=cap)
        print(" ", cap, solve_transport(s, DEMAND, COST)["cost"])
    print("금지 조합 때문에 불가능한 경우:")
    s = {"W1": 4, "W2": 3, "W3": 1}
    cost2 = {w: dict(COST[w]) for w in COST}
    cost2["W1"]["C"] = None
    cost2["W2"]["C"] = None
    cost2["W3"]["C"] = 6
    r2 = solve_transport(s, DEMAND, cost2)
    print(" ", r2, "전수:", brute_force(s, DEMAND, cost2))


if __name__ == "__main__":
    main()
