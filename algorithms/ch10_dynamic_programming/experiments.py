"""알고리즘 10장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch10_dynamic_programming/experiments.py` 로 실행한다.

E0 첫 화면 손계산: fib(5) 재귀 호출, RUN -> RAIN 편집거리 표와 고치기 순서
E1 실패 최소 예제 1: 순진한 재귀 fib(n) 의 호출 수(n = 10..35)와 2F(n+1)-1, 메모·표와의 시간
E2 실패 최소 예제 2 + 비교 측정: 무작위 문자열(알파벳 4개, 시드 10개) 편집거리
    재귀(세 갈래 / 같은 글자면 대각선만), 메모, 표, 행 두 개의 호출·칸 수, 시간, 메모리
E3 길이를 키우면: 표와 행 두 개의 시간·메모리(길이 100..2000), 메모 판의 재귀 한도
E4 비터비 vs 전수 탐색: 연산 수(TK^2 꼴 vs K^T 꼴)와 시간, 같은 답인지
E5 비터비 언더플로: 확률을 그대로 곱하면 언제 모두 0.0 이 되나
E6 거스름돈(8장 연결): 메모 없는 재귀 vs 표
E7 위젯 트레이스: 편집거리 표 채우기 순서·역추적, fib(6) 재귀 트리

결과는 results/ch10.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
호출 수·칸 수는 입력이 같으면 항상 같다. 실측 시간과 메모리는 기기·부하·버전에 따라 달라진다.
"""

import json
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch10_dp as dp  # noqa: E402

OUT = HERE / "results" / "ch10.json"
SEEDS = 10
ALPHA = "ACGT"


def rand_str(rng, n):
    return "".join(rng.choice(ALPHA) for _ in range(n))


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    v = fn(*args, **kw)
    return v, (time.perf_counter() - t0) * 1000


def peak_kib(fn, *args, **kw):
    tracemalloc.start()
    tracemalloc.reset_peak()
    v = fn(*args, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return v, peak / 1024


# ------------------------------------------------------------------ E0

def e0_first_screen():
    visits = [0] * 6
    c = dp.Counter()
    v = dp.fib_naive(5, c, visits)
    c30 = dp.Counter()
    dp.fib_naive(30, c30)
    a, b = "RUN", "RAIN"
    D = dp.edit_distance_table(a, b)
    ops = dp.backtrace(D, a, b)
    return {
        "fib5": {"value": v, "calls": c.calls, "distinct": 6, "visits": visits},
        "fib30": {"value": dp.fib_value(30), "calls": c30.calls, "distinct": 31},
        "edit": {"a": a, "b": b, "D": D.tolist(), "distance": int(D[-1, -1]), "ops": ops,
                 "applied": dp.apply_ops(a, b, ops), "cells": int(D.size)},
    }


# ------------------------------------------------------------------ E1

def e1_fib():
    rows = []
    for n in range(10, 36, 5):
        visits = [0] * (n + 1)
        c = dp.Counter()
        _, t_naive = timed(dp.fib_naive, n, c, visits)
        cm = dp.Counter()
        vm, t_memo = timed(dp.fib_memo, n, cm)
        ct = dp.Counter()
        vt, t_table = timed(dp.fib_table, n, ct)
        rows.append({
            "n": n, "F_n": dp.fib_value(n), "naive_calls": c.calls,
            "formula_2F_n1_minus_1": 2 * dp.fib_value(n + 1) - 1,
            "distinct": n + 1, "ratio_calls_per_distinct": c.calls / (n + 1),
            "max_visits": max(visits), "max_visits_arg": visits.index(max(visits)),
            "visits_fib2": visits[2],
            "memo_calls": cm.calls, "table_cells": ct.cells,
            "naive_ms": t_naive, "memo_ms": t_memo, "table_ms": t_table,
            "all_equal": vm == vt == dp.fib_value(n),
        })
    return {"rows": rows}


# ------------------------------------------------------------------ E2

def e2_edit_small():
    rows = []
    for m in (6, 8, 10, 12):
        seeds = SEEDS if m <= 10 else 1
        rec = {"m": m, "seeds_full": seeds, "distinct": (m + 1) ** 2,
               "naive_calls_formula": dp.naive_call_count(m, m),
               "delannoy_m1": dp.delannoy(m - 1, m - 1)}
        full_calls, full_ms, sc_calls, sc_ms = [], [], [], []
        memo_calls, memo_ms, table_ms, two_ms, dists = [], [], [], [], []
        max_visit_ok, max_visit = True, None
        memo_kib, table_kib, two_kib = [], [], []
        for s in range(SEEDS):
            rng = random.Random(1000 * m + s)
            a, b = rand_str(rng, m), rand_str(rng, m)
            if s < seeds:
                c = dp.Counter()
                visits = [[0] * (m + 1) for _ in range(m + 1)] if m <= 10 else None
                v, t = timed(dp.edit_distance_naive, a, b, c, visits)
                full_calls.append(c.calls)
                full_ms.append(t)
                if visits is not None:
                    mv = max(max(r) for r in visits)
                    arg = next((i, j) for i in range(m + 1) for j in range(m + 1) if visits[i][j] == mv)
                    max_visit = {"any_cell": mv, "any_cell_at": list(arg), "interior_1_1": visits[1][1],
                                 "edge_1_0": visits[1][0], "edge_0_1": visits[0][1]}
                    # 안쪽 칸 (i, j >= 1) 의 방문 수 = Delannoy(m-i, m-j). 가장자리 칸은 여러 안쪽 칸에서 와서 더 많다.
                    ok = all(visits[i][j] == dp.delannoy(m - i, m - j)
                             for i in range(1, m + 1) for j in range(1, m + 1))
                    max_visit_ok = max_visit_ok and ok
            c2 = dp.Counter()
            v2, t2 = timed(dp.edit_distance_naive, a, b, c2, None, True)
            sc_calls.append(c2.calls)
            sc_ms.append(t2)
            c3 = dp.Counter()
            v3, t3 = timed(dp.edit_distance_memo, a, b, c3)
            memo_calls.append(c3.calls)
            memo_ms.append(t3)
            D, t4 = timed(dp.edit_distance_table, a, b)
            table_ms.append(t4)
            v5, t5 = timed(dp.edit_distance_two_rows, a, b)
            two_ms.append(t5)
            dists.append(int(D[-1, -1]))
            assert v2 == v3 == int(D[-1, -1]) == v5
            if s < seeds:
                assert v == v2
            memo_kib.append(peak_kib(dp.edit_distance_memo, a, b)[1])
            table_kib.append(peak_kib(dp.edit_distance_table, a, b)[1])
            two_kib.append(peak_kib(dp.edit_distance_two_rows, a, b)[1])
        rec.update({
            "naive_calls": sorted(set(full_calls)), "naive_ms_median": statistics.median(full_ms),
            "naive_ms_all": full_ms,
            "max_visit": max_visit, "interior_visits_equal_delannoy": max_visit_ok if m <= 10 else None,
            "shortcut_calls_min": min(sc_calls), "shortcut_calls_median": statistics.median(sc_calls),
            "shortcut_calls_max": max(sc_calls), "shortcut_ms_median": statistics.median(sc_ms),
            "memo_calls": sorted(set(memo_calls)), "memo_ms_median": statistics.median(memo_ms),
            "table_cells": (m + 1) ** 2, "table_ms_median": statistics.median(table_ms),
            "two_rows_ms_median": statistics.median(two_ms),
            "memo_peak_kib_median": statistics.median(memo_kib),
            "table_peak_kib_median": statistics.median(table_kib),
            "two_rows_peak_kib_median": statistics.median(two_kib),
            "distances": dists,
        })
        rows.append(rec)
    return {"alphabet": ALPHA, "rows": rows}


# ------------------------------------------------------------------ E3

def e3_edit_large():
    rows = []
    for m in (100, 500, 1000, 2000):
        t_tab, t_two, k_tab, k_two, nbytes = [], [], [], [], None
        memo = None
        for s in range(3):
            rng = random.Random(77 + 31 * m + s)
            a, b = rand_str(rng, m), rand_str(rng, m)
            D, t = timed(dp.edit_distance_table, a, b)
            v, t2 = timed(dp.edit_distance_two_rows, a, b)
            assert int(D[-1, -1]) == v
            t_tab.append(t)
            t_two.append(t2)
            nbytes = int(D.nbytes)
            if s == 0:
                k_tab.append(peak_kib(dp.edit_distance_table, a, b)[1])
                k_two.append(peak_kib(dp.edit_distance_two_rows, a, b)[1])
                try:
                    vm, tm = timed(dp.edit_distance_memo, a, b)
                    memo = {"ok": True, "ms": tm, "same": vm == v}
                except RecursionError as e:
                    memo = {"ok": False, "error": type(e).__name__}
        rows.append({"m": m, "cells": (m + 1) ** 2, "table_ms_median": statistics.median(t_tab),
                     "two_rows_ms_median": statistics.median(t_two),
                     "table_nbytes": nbytes, "table_peak_kib": k_tab[0], "two_rows_peak_kib": k_two[0],
                     "memo_default_limit": memo})
    # 메모 판의 재귀 한도: 기본 한도에서 fib_memo(n) 이 되는 가장 큰 n 근처
    lim = sys.getrecursionlimit()
    fib_rows = []
    for n in (500, 900, 950, 990, 1000, 2000):
        try:
            dp.fib_memo(n)
            fib_rows.append({"n": n, "memo": "ok"})
        except RecursionError:
            fib_rows.append({"n": n, "memo": "RecursionError"})
    fib_table_2000 = dp.fib_table(2000) == dp.fib_value(2000)
    return {"rows": rows, "recursion_limit": lim, "fib_memo": fib_rows,
            "fib_table_2000_ok": fib_table_2000, "fib_2000_digits": len(str(dp.fib_value(2000)))}


# ------------------------------------------------------------------ E4

def e4_viterbi():
    rows = []
    for K in (2, 3, 4):
        for T in (4, 6, 8, 10, 12, 20):
            nr = np.random.default_rng(100 * K + T)
            pi, A, B = dp.random_hmm(nr, K, 3)
            obs = nr.integers(0, 3, size=T).tolist()
            lp, lA, lB = np.log(pi), np.log(A), np.log(B)
            cv = dp.Counter()
            (path, score), tv = timed(dp.viterbi, lp, lA, lB, obs, cv)
            rec = {"K": K, "T": T, "viterbi_ops": cv.ops, "viterbi_formula": K + (T - 1) * K * K,
                   "paths": K ** T, "brute_ops_formula": (K ** T) * 2 * T, "viterbi_ms": tv}
            if K ** T <= 70000:
                cb = dp.Counter()
                (bpath, bscore), tb = timed(dp.viterbi_bruteforce, lp, lA, lB, obs, cb)
                rec.update({"brute_ops": cb.ops, "brute_ms": tb, "same_path": bpath == path,
                            "same_score": abs(bscore - score) < 1e-9})
            rows.append(rec)
    return {"rows": rows, "M": 3}


# ------------------------------------------------------------------ E5

def e5_underflow():
    nr = np.random.default_rng(2026)
    pi, A, B = dp.random_hmm(nr, 3, 4)
    obs = nr.integers(0, 4, size=3000).tolist()
    lp, lA, lB = np.log(pi), np.log(A), np.log(B)
    out = {"K": 3, "M": 4, "T_total": 3000}
    p_path, p_val, first_zero = dp.viterbi_prob(pi, A, B, obs)
    l_path, l_val = dp.viterbi(lp, lA, lB, obs)
    out.update({"first_all_zero_T": first_zero, "prob_final_value": p_val, "log_final": l_val,
                "log_final_log10": l_val / np.log(10),
                "paths_agree_states": int(sum(x == y for x, y in zip(p_path, l_path))),
                "prob_path_logprob": dp.path_logprob(lp, lA, lB, obs, p_path)})
    # 언더플로 전 길이에서는 두 판이 같은 경로를 낸다
    Ts = [50, 200, first_zero - 1 if first_zero else 500]
    agree = []
    for T in Ts:
        pp, _, _ = dp.viterbi_prob(pi, A, B, obs[:T])
        ll, _ = dp.viterbi(lp, lA, lB, obs[:T])
        agree.append({"T": T, "same_path": pp == ll})
    out["before_underflow"] = agree
    out["float64_min_subnormal_log10"] = float(np.log10(np.nextafter(0, 1)))
    return out


# ------------------------------------------------------------------ E6

def e6_coins():
    coins = [1, 3, 4]
    best, combo = dp.coin_change_dp(coins, 6)
    rows = []
    for amount in (6, 10, 20, 30):
        c = dp.Counter()
        v, t = timed(dp.coin_change_bruteforce, coins, amount, c)
        b2, _ = dp.coin_change_dp(coins, amount)
        rows.append({"amount": amount, "brute_calls": c.calls, "brute_ms": t,
                     "table_cells": amount + 1, "value": v, "same": v == b2[amount]})
    return {"coins": coins, "table_0_6": best[:7], "combo_6": combo,
            "greedy_6": dp.greedy_coins(coins, 6), "rows": rows}


# ------------------------------------------------------------------ E7

def _widget_case(a, b):
    order = []
    D = dp.edit_distance_table(a, b, None, order)
    ops = dp.backtrace(D, a, b)
    # 짧은 접두사 쌍마다 재귀로 다시 구해 표와 대조
    check = [[dp.edit_distance_naive(a[:i], b[:j], None, None, True) for j in range(len(b) + 1)]
             for i in range(len(a) + 1)]
    pos = {cell: k for k, cell in enumerate(order)}
    dep_ok = all(pos[(i - 1, j)] < pos[(i, j)] and pos[(i, j - 1)] < pos[(i, j)] and pos[(i - 1, j - 1)] < pos[(i, j)]
                 for (i, j) in order if i > 0 and j > 0)
    return {"a": a, "b": b, "D": D.tolist(), "naive": check,
            "all_match_naive": check == D.tolist(), "order_ok": dep_ok,
            "path": [[op, i, j] for op, i, j in ops], "distance": int(D[-1, -1]),
            "applied": dp.apply_ops(a, b, ops)}


def _fib_tree(n):
    nodes = []

    def visit(k, depth, parent):
        idx = len(nodes)
        nodes.append([k, depth, parent])
        if k >= 2:
            visit(k - 1, depth + 1, idx)
            visit(k - 2, depth + 1, idx)

    visit(n, 0, -1)
    return nodes


def e7_widget():
    return {"cases": [_widget_case("RUN", "RAIN"), _widget_case("PYTHON", "TYPHOON")],
            "fib_tree": {"n": 6, "nodes": _fib_tree(6)}}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. NumPy 스레드 2개",
        },
        "E0_first_screen": e0_first_screen(),
        "E1_fib": e1_fib(),
        "E2_edit_small": e2_edit_small(),
        "E3_edit_large": e3_edit_large(),
        "E4_viterbi": e4_viterbi(),
        "E5_underflow": e5_underflow(),
        "E6_coins": e6_coins(),
        "E7_widget": e7_widget(),
    }
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", OUT, res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
