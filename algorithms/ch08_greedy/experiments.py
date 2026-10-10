"""알고리즘 8장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch08_greedy/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 동전 {1,3,4} 금액 6, 간선 5개 MST, 최대 덮기 k=2 의 3/4 사례
E1 실패 최소 예제 1(거스름돈): 동전 체계 6종, 금액 1~200 에서 탐욕이 최소가 아닌 금액 비율, 가장 작은 반례,
   Kozen·Zaks(1994) 범위 c3+1 < x < cm+c(m-1) 안에 드는지
E2 실패 최소 예제 2(최대 덮기): (a) 원소 4개, 집합 4개의 모든 사례 전수(65,536가지) k=2,3,
   (b) 무작위 사례 1,000개(원소 12, 집합 8, 포함 확률 0.3) k=2,3,4. 탐욕/최적 비율과 1-((k-1)/k)^k 비교
E3 같은 탐욕, 다른 구조: 점 5개 무작위 그래프 1,000개에서 "무거운 것부터" 탐욕으로 숲과 매칭을 고르고 전수 최적과 비교
E4 비교 측정(MST): 연결 무작위 그래프 V=100, 1,000, 10,000(E=4V), 시드 10개. Kruskal·Prim 의 연산 수와 시간,
   UnionFind 의 경로 압축·랭크를 끈 변형의 포인터 걸음 수
E5 흔한 실패: (i) 최대 덮기에서 새로 덮는 수 대신 집합 크기로 고르기, (ii) Prim 의 키를 다익스트라처럼 d(u)+w 로 두기
E6 위젯 트레이스: 6점 그래프의 Kruskal, 거스름돈 세 사례

결과는 results/ch08.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
연산 수는 입력과 시드가 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

import itertools
import json
import os
import platform
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402  (버전 기록용)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch08_greedy as A  # noqa: E402

OUT = HERE / "results" / "ch08.json"


def r4(x):
    return round(float(x), 4)


def frac(f):
    return {"num": f.numerator, "den": f.denominator, "value": r4(f)}


# ---------------------------------------------------------------- E0
def e0_first_screen():
    coins = [1, 3, 4]
    g = A.greedy_coins(coins, 6)
    _, opt = A.coin_change_dp(coins, 6)
    edges = [(1, 0, 1), (2, 1, 2), (3, 0, 2), (4, 2, 3), (5, 1, 3)]  # A-B 1, B-C 2, A-C 3, C-D 4, B-D 5
    tr = []
    total, chosen = A.kruskal(4, edges, trace=tr)
    sets = [frozenset({1, 3}), frozenset({1, 2}), frozenset({3, 4})]
    gv, gi = A.greedy_max_coverage(sets, 2)
    ov, oi = A.brute_force_coverage(sets, 2)
    kr = [1, 5, 10, 50, 100, 500]
    return {
        "coin": {"coins": coins, "amount": 6, "greedy": g, "optimal": opt},
        "coin_kr_760": {"coins": kr, "amount": 760, "greedy": A.greedy_coins(kr, 760), "optimal": A.coin_change_dp(kr, 760)[1]},
        "mst": {"names": "ABCD", "edges": edges, "steps": [[list(e), ok] for e, ok, _ in tr],
                "total": total, "brute": A.brute_force_mst(4, edges)},
        "coverage": {"sets": [sorted(s) for s in sets], "k": 2, "greedy_idx": gi, "greedy": gv,
                     "opt_idx": oi, "opt": ov, "ratio": frac(Fraction(gv, ov)), "bound_k2": frac(A.coverage_bound(2))},
        "matching_path": _matching_path(),
    }


def _matching_path():
    # a-b 3, b-c 4, c-d 3 인 길: 무거운 것부터 고르면 b-c(4) 하나, 최적은 a-b + c-d = 6
    edges = [(3, 0, 1), (4, 1, 2), (3, 2, 3)]
    gw, gc = A.greedy_max_weight(4, edges, A.is_matching)
    return {"edges": edges, "greedy": gw, "greedy_edges": gc, "opt": A.brute_force_max_weight(4, edges, A.is_matching)}


# ---------------------------------------------------------------- E1
SYSTEMS = [
    ("미국 센트", [1, 5, 10, 25]),
    ("한국 동전(원)", [1, 5, 10, 50, 100, 500]),
    ("유로 센트", [1, 2, 5, 10, 20, 50, 100, 200]),
    ("{1,3,4}", [1, 3, 4]),
    ("{1,7,10}", [1, 7, 10]),
    ("{1,10,25}", [1, 10, 25]),
]


def e1_coins(max_amount=200):
    rows = []
    for name, cs in SYSTEMS:
        fails = A.coin_failures(cs, max_amount)
        cs_sorted = sorted(cs)
        lo, hi = cs_sorted[2] + 1, cs_sorted[-1] + cs_sorted[-2]
        smallest = fails[0][0] if fails else None
        # Kozen-Zaks 범위 끝(cm + c(m-1))까지만 봐도 반례가 있는지 여부가 같은지
        fails_kz = A.coin_failures(cs, hi)
        excess = [g - m for _, g, m in fails]
        rows.append({
            "name": name, "coins": cs, "amounts": max_amount,
            "fail_count": len(fails), "fail_ratio": r4(len(fails) / max_amount),
            "smallest_counterexample": smallest,
            "smallest_detail": (lambda f: {"greedy": A.greedy_coins(cs, f), "optimal": A.coin_change_dp(cs, f)[1]})(smallest) if smallest else None,
            "kz_range": [lo, hi],
            "smallest_in_kz_open_range": (lo < smallest < hi) if smallest else None,
            "has_fail_up_to_kz_hi": bool(fails_kz),
            "max_excess_coins": max(excess) if excess else 0,
            "mean_excess_coins": r4(statistics.mean(excess)) if excess else 0,
            "first_fails": [f[0] for f in fails[:8]],
        })
    big = {}
    for name, cs in SYSTEMS[:3]:
        big[name] = len(A.coin_failures(cs, 1000))
    return {"rows": rows, "canonical_check_1_to_1000_fail_counts": big}


# ---------------------------------------------------------------- E2
def e2_coverage():
    # (a) 전수: 원소 {1,2,3,4}, 집합 4개(빈 집합 포함 16가지씩) -> 16^4 = 65,536 사례
    U = [1, 2, 3, 4]
    subsets = [frozenset(c) for r in range(5) for c in itertools.combinations(U, r)]
    exhaustive = {}
    for k in (2, 3):
        worst = Fraction(1)
        worst_case = None
        n_cases = n_worse = 0
        for combo in itertools.product(subsets, repeat=4):
            sets = list(combo)
            ov, _ = A.brute_force_coverage(sets, k)
            if ov == 0:
                continue
            n_cases += 1
            gv, gi = A.greedy_max_coverage(sets, k)
            if gv < ov:
                n_worse += 1
            r = Fraction(gv, ov)
            if r < worst:
                worst, worst_case = r, {"sets": [sorted(s) for s in sets], "greedy_idx": gi, "greedy": gv, "opt": ov}
        exhaustive[str(k)] = {"cases": n_cases, "greedy_worse": n_worse, "worse_ratio": r4(n_worse / n_cases),
                              "min_ratio": frac(worst), "bound": frac(A.coverage_bound(k)),
                              "min_ratio_ge_bound": worst >= A.coverage_bound(k), "worst_case": worst_case}
    # (b) 무작위 1,000 개
    rand = {}
    for k in (2, 3, 4):
        ratios = []
        worse = 0
        for seed in range(1000):
            rng = A.seeded(80000 + seed)
            sets = A.random_sets(8, 12, 0.3, rng)
            ov, _ = A.brute_force_coverage(sets, k)
            gv, _ = A.greedy_max_coverage(sets, k)
            r = Fraction(gv, ov)
            ratios.append(r)
            worse += gv < ov
        mn = min(ratios)
        fl = [float(r) for r in ratios]
        rand[str(k)] = {"instances": 1000, "greedy_worse": worse, "min_ratio": frac(mn), "mean_ratio": r4(statistics.mean(fl)),
                        "p05_ratio": r4(np.percentile(fl, 5)), "bound": frac(A.coverage_bound(k)),
                        "all_ge_bound": all(r >= A.coverage_bound(k) for r in ratios)}
    strict = [frozenset({1, 2, 3}), frozenset({4, 5, 6}), frozenset({1, 2, 4, 5})]
    sg, si = A.greedy_max_coverage(strict, 2)
    so, soi = A.brute_force_coverage(strict, 2)
    return {"exhaustive_u4_n4": exhaustive, "random_u12_n8_p03": rand,
            "no_tie_example": {"sets": [sorted(s) for s in strict], "greedy_idx": si, "greedy": sg, "opt_idx": soi, "opt": so},
            "one_minus_inv_e": r4(1 - 1 / np.e)}


# ---------------------------------------------------------------- E3
def e3_matroid_contrast(trials=1000):
    res = {}
    stats = {"forest": [0, Fraction(1)], "matching": [0, Fraction(1)]}
    n_used = 0
    for seed in range(trials):
        rng = A.seeded(30000 + seed)
        edges = A.random_small_graph(5, 0.6, rng)
        if not edges:
            continue
        n_used += 1
        for name, ind in (("forest", A.is_forest), ("matching", A.is_matching)):
            gw, _ = A.greedy_max_weight(5, edges, ind)
            ow = A.brute_force_max_weight(5, edges, ind)
            if gw < ow:
                stats[name][0] += 1
            stats[name][1] = min(stats[name][1], Fraction(gw, ow))
    for name, (bad, mn) in stats.items():
        res[name] = {"graphs": n_used, "greedy_worse": bad, "worse_ratio": r4(bad / n_used), "min_ratio": frac(mn)}
    return res


# ---------------------------------------------------------------- E4
def _ms(fn, reps):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return out, statistics.median(ts)


def e4_mst(sizes=(100, 1000, 10000), seeds=10):
    rows = []
    for n in sizes:
        m = 4 * n
        agg = {k: [] for k in ("k_scan", "k_find", "k_steps", "p_scan", "p_push", "p_pop", "k_ms", "p_ms",
                               "naive_steps", "rank_steps", "comp_steps")}
        agree = 0
        reps = 5 if n <= 1000 else 3
        for seed in range(seeds):
            rng = A.seeded(n * 100 + seed)
            edges = A.random_connected_graph(n, m, rng)
            adj = A.adjacency(n, edges)
            ck, cp = A.Counter(), A.Counter()
            kw, kc = A.kruskal(n, edges, counter=ck)
            pw, pc = A.prim(n, edges, counter=cp, adj=adj)
            agree += (kw == pw) and A.is_spanning_tree(n, kc) and A.is_spanning_tree(n, pc)
            _, kms = _ms(lambda: A.kruskal(n, edges), reps)
            _, pms = _ms(lambda: A.prim(n, edges, adj=adj), reps)
            agg["k_scan"].append(ck.edge_scan); agg["k_find"].append(ck.find_calls); agg["k_steps"].append(ck.find_steps)
            agg["p_scan"].append(cp.edge_scan); agg["p_push"].append(cp.push); agg["p_pop"].append(cp.pop)
            agg["k_ms"].append(kms); agg["p_ms"].append(pms)
            for key, comp, rank in (("naive_steps", False, False), ("rank_steps", False, True), ("comp_steps", True, True)):
                cc = A.Counter()
                A.kruskal(n, edges, counter=cc, compress=comp, by_rank=rank)
                agg[key].append(cc.find_steps)
        mean = {k: r4(statistics.mean(v)) for k, v in agg.items() if not k.endswith("ms")}
        rows.append({"n": n, "m": m, "seeds": seeds, "weights_equal": agree,
                     **{"mean_" + k: v for k, v in mean.items()},
                     "kruskal_ms_median": r4(statistics.median(agg["k_ms"])),
                     "prim_ms_median": r4(statistics.median(agg["p_ms"]))})
    return rows


# ---------------------------------------------------------------- E5
def e5_common_failures():
    # (i) 크기 순으로 고르기
    by = {}
    for k in (2, 3, 4):
        worse_gain = worse_size = size_lt_gain = 0
        min_size = Fraction(1)
        for seed in range(1000):
            rng = A.seeded(80000 + seed)
            sets = A.random_sets(8, 12, 0.3, rng)
            ov, _ = A.brute_force_coverage(sets, k)
            gv, _ = A.greedy_max_coverage(sets, k)
            sv, _ = A.greedy_max_coverage(sets, k, by="size")
            worse_gain += gv < ov
            worse_size += sv < ov
            size_lt_gain += sv < gv
            min_size = min(min_size, Fraction(sv, ov))
        by[str(k)] = {"instances": 1000, "gain_worse_than_opt": worse_gain, "size_worse_than_opt": worse_size,
                      "size_worse_than_gain": size_lt_gain, "size_min_ratio": frac(min_size),
                      "size_below_bound": min_size < A.coverage_bound(k)}
    # 크기 순이 무너지는 손 사례: 같은 원소를 거의 공유하는 큰 집합 두 개
    hand = [frozenset({1, 2, 3, 4}), frozenset({1, 2, 3, 5}), frozenset({6, 7, 8})]
    hs, _ = A.greedy_max_coverage(hand, 2, by="size")
    hg, _ = A.greedy_max_coverage(hand, 2)
    # (ii) Prim 에 다익스트라 키
    wrong = 0
    excess = []
    for seed in range(100):
        rng = A.seeded(50000 + seed)
        edges = A.random_connected_graph(50, 200, rng, wmax=20)
        mw, _ = A.prim(50, edges)
        dw, dc = A.prim_with_dijkstra_key(50, edges)
        assert A.is_spanning_tree(50, dc)
        if dw > mw:
            wrong += 1
            excess.append((dw - mw) / mw)
    return {"size_instead_of_gain": by,
            "size_hand": {"sets": [sorted(s) for s in hand], "size_pick": hs, "gain_pick": hg},
            "prim_dijkstra_key": {"graphs": 100, "n": 50, "m": 200, "weights": "1..20", "heavier_than_mst": wrong,
                                  "mean_excess_ratio": r4(statistics.mean(excess)) if excess else 0,
                                  "max_excess_ratio": r4(max(excess)) if excess else 0}}


# ---------------------------------------------------------------- E6
def e6_widget():
    names = "ABCDEF"
    pos = [[70, 70], [250, 40], [430, 70], [90, 230], [290, 220], [520, 220]]
    edges = [(4, 0, 1), (2, 0, 3), (5, 1, 2), (3, 1, 3), (6, 1, 4), (1, 3, 4), (7, 2, 4), (8, 2, 5), (4, 4, 5)]
    tr = []
    total, chosen = A.kruskal(6, edges, trace=tr)
    steps = []
    acc = []
    for (w, u, v), ok, comp in tr:
        side = {i for i in range(6) if comp[i] == comp[u]} if not ok else None
        # 받은 간선이면, 받기 직전 u 의 조각을 절단 한쪽으로 본다
        steps.append({"edge": [w, u, v], "ok": ok, "comp": comp})
        if ok:
            acc.append([w, u, v])
    # 절단 불변식: 받기 직전 조각(u 쪽)을 한쪽으로 두면 받은 간선이 그 절단에서 가장 가벼운가
    prev = list(range(6))
    for st in steps:
        w, u, v = st["edge"]
        side = {i for i in range(6) if prev[i] == prev[u]}
        st["cut"] = sorted(side)
        st["lightest_cut"] = A.lightest_crossing(6, edges, side)
        st["is_lightest"] = (w == st["lightest_cut"]) if st["ok"] else None
        prev = st["comp"]
    coins = []
    for cs, amt in (([1, 3, 4], 6), ([1, 7, 10], 14), ([1, 5, 10, 25], 30)):
        coins.append({"coins": cs, "amount": amt, "greedy": A.greedy_coins(cs, amt), "optimal": A.coin_change_dp(cs, amt)[1]})
    return {"names": names, "pos": pos, "edges": edges, "steps": steps, "total": total,
            "brute": A.brute_force_mst(6, edges), "coins": coins}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                "machine": platform.machine(), "omp_threads": os.environ.get("OMP_NUM_THREADS")},
        "E0_first_screen": e0_first_screen(),
        "E1_coins": e1_coins(),
        "E2_coverage": e2_coverage(),
        "E3_matroid_contrast": e3_matroid_contrast(),
        "E4_mst": e4_mst(),
        "E5_common_failures": e5_common_failures(),
        "E6_widget": e6_widget(),
    }
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("E6_widget",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
