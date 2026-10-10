"""알고리즘 5장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch05_balanced_tree/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 키 1..7 을 차례로 넣은 BST 와 AVL, 그리고 같은 키를 4, 2, 6, 1, 3, 5, 7 순서로 넣은 BST
E1 실패 최소 예제와 비교 측정: n = 100, 1000, 5000, 넣는 순서 3종(정렬, 무작위, 지그재그) x 트리 2종 x 시드 10
E2 재귀판 넣기는 정렬된 입력에서 RecursionError 가 난다: 몇 번째 키에서 나는가
E3 AVL 높이의 상한: N(h) = N(h-1) + N(h-2) + 1, 피보나치 트리, h <= log_phi(n+1)(이 장의 유도)과
   논문 Lemma 1 을 이 장의 높이 단위로 옮긴 h - 1 < log_phi(n+1) < (3/2) log2(n+1)
E4 무작위 순서 BST 의 평균 깊이: 작은 n 은 모든 순서를 다 넣어 보고, 큰 n 은 시드 평균.
   비교 식 2(1+1/n)H_n - 3 은 '키 i 가 키 j 의 조상일 확률 = 1/(|i-j|+1)' 에서 유도한다(웹 챕터 3절)
E5 힙: heapify 와 push n 번의 비교 수, heapq 와 같은 답, 빔 서치처럼 상위 k 개만 남기기
E6 위젯 트레이스(키 10개)

결과는 results/ch05.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
비교 수·높이·회전 수는 입력과 시드가 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

import heapq
import itertools
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch05_trees as T  # noqa: E402

OUT = HERE / "results" / "ch05.json"
NS = (100, 1000, 5000)
ORDERS = ("sorted", "random", "zigzag")
SEEDS = 10
PHI = (1 + 5 ** 0.5) / 2
C_PHI = 1 / math.log2(PHI)  # 1.4404...


def e0_first_screen():
    keys = list(range(1, 8))
    out = {"keys": keys}
    for kind in ("bst", "avl"):
        t, c = T.build(kind, keys)
        d = T.depths(t.root)
        out[kind] = {"height": T.height(t.root), "insert_comparisons": c.comparisons, "rotations": c.rotations,
                     "depths": [d[k] for k in keys], "search_all": sum(d.values()),
                     "search_mean": sum(d.values()) / len(keys), "find_7": d[7]}
    good = [4, 2, 6, 1, 3, 5, 7]
    t, c = T.build("bst", good)
    d = T.depths(t.root)
    out["bst_good_order"] = {"order": good, "height": T.height(t.root), "insert_comparisons": c.comparisons,
                             "depths": [d[k] for k in keys], "search_all": sum(d.values())}
    # 1000 개면
    out["n1000"] = {"bst_sorted_height": 1000, "log2_n_plus_1": math.log2(1001),
                    "bound_1_44": C_PHI * math.log2(1001)}
    return out


def _timed_build(kind, keys):
    t0 = time.perf_counter()
    t, c = T.build(kind, keys)
    return t, c, (time.perf_counter() - t0) * 1000


def e1_grid():
    rows = []
    for n in NS:
        row = {"n": n, "min_height": math.ceil(math.log2(n + 1)), "bound_phi": C_PHI * math.log2(n + 1),
               "paper_lemma_h": 1 + C_PHI * math.log2(n + 1), "paper_lemma_h_coarse": 1 + 1.5 * math.log2(n + 1), "n_n_minus_1_over_2": n * (n - 1) // 2,
               "mean_depth_chain": (n + 1) / 2}
        for order in ORDERS:
            runs = SEEDS if order == "random" else 1
            for kind in ("bst", "avl"):
                vals = []
                for s in range(runs):
                    keys = T.orders(order, n, random.Random(s))
                    t, c, ms = _timed_build(kind, keys)
                    vals.append({"height": T.height(t.root), "search_mean": T.mean_search_comparisons(t.root, keys),
                                 "insert_comparisons": c.comparisons, "rotations": c.rotations, "single": c.single,
                                 "double": c.double, "ms": ms})
                agg = {"runs": runs}
                for f in ("height", "search_mean", "insert_comparisons", "rotations", "single", "double"):
                    agg[f + "_mean"] = statistics.mean(v[f] for v in vals)
                agg["height_max"] = max(v["height"] for v in vals)
                agg["height_min"] = min(v["height"] for v in vals)
                agg["rotations_per_insert"] = agg["rotations_mean"] / n
                agg["ms_median"] = statistics.median(v["ms"] for v in vals)
                row[f"{order}_{kind}"] = agg
        row["ratio_sorted_bst_to_avl_ms"] = row["sorted_bst"]["ms_median"] / row["sorted_avl"]["ms_median"]
        row["ratio_sorted_bst_to_avl_cmp"] = row["sorted_bst"]["insert_comparisons_mean"] / row["sorted_avl"]["insert_comparisons_mean"]
        rows.append(row)
    return {"seeds": SEEDS, "rows": rows}


def e2_recursion():
    limit = sys.getrecursionlimit()
    t = T.BST()
    fail_at = None
    for k in range(1, 5001):
        try:
            t.insert_recursive(k)
        except RecursionError:
            fail_at = k
            break
    out = {"recursion_limit": limit, "sorted_fail_at_key": fail_at, "nodes_before_fail": t.n,
           "height_before_fail": T.height(t.root)}
    # 같은 재귀판이 무작위 순서 10만 개에서는 문제없다
    keys = T.orders("random", 100_000, random.Random(0))
    t2 = T.BST()
    for k in keys:
        t2.insert_recursive(k)
    out["random_100k_ok"] = t2.n == 100_000
    out["random_100k_height"] = T.height(t2.root)
    # 반복판은 정렬된 입력 5000 개도 넣는다
    t3, c3 = T.build("bst", list(range(1, 5001)))
    out["iterative_sorted_5000_height"] = T.height(t3.root)
    return out


def e3_avl_bound():
    table = []
    fib = [0, 1]
    while len(fib) < 40:
        fib.append(fib[-1] + fib[-2])
    for h in range(1, 26):
        N = T.min_avl_nodes(h)
        table.append({"h": h, "N": N, "fib_h_plus_2_minus_1": fib[h + 2] - 1, "phi_pow_h": PHI ** h,
                      "h_over_log2_N_plus_1": h / math.log2(N + 1),
                      "bound_phi": C_PHI * math.log2(N + 1), "paper_lemma_h": 1 + C_PHI * math.log2(N + 1), "paper_lemma_h_coarse": 1 + 1.5 * math.log2(N + 1)})
    # 피보나치 트리를 실제로 만들어 AVL 인지, 높이와 노드 수가 맞는지
    fibs = []
    for h in (5, 10, 15):
        order, root = T.fibonacci_tree_keys(h)
        bst, _ = T.build("bst", order)
        fibs.append({"h": h, "nodes": len(order), "height": T.height(bst.root),
                     "is_avl": all(ok for _, ok, _ in T.check_avl(root)),
                     "same_shape_as_bst": T.snapshot(bst.root)[1] == T.snapshot(root)[1]})
    # 무작위 넣기·지우기 열 뒤에도 높이가 상한 아래인가
    worst_ratio, checks = 0.0, 0
    for s in range(20):
        rng = random.Random(100 + s)
        t = T.AVL()
        present = set()
        for _ in range(4000):
            k = rng.randrange(3000)
            if rng.random() < 0.6:
                t.insert(k)
                present.add(k)
            else:
                t.delete(k)
                present.discard(k)
            if t.n:
                worst_ratio = max(worst_ratio, T.height(t.root) / (C_PHI * math.log2(t.n + 1)))
        checks += 1
        assert T.inorder(t.root) == sorted(present)
    # Binet 꼴 근사를 NumPy 로: N(h) + 1 = F(h+2), F(k) = round(phi^k / sqrt5)
    hs = np.arange(1, 26)
    binet = np.rint(PHI ** (hs + 2) / np.sqrt(5)).astype(np.int64) - 1
    return {"phi": PHI, "c_phi": C_PHI, "table": table, "fibonacci_trees": fibs,
            "random_ops": {"sequences": checks, "ops_each": 4000, "worst_height_over_bound": worst_ratio},
            "binet_matches": bool((binet == np.array([r["N"] for r in table])).all())}


def _harmonic(n):
    return sum(1 / i for i in range(1, n + 1))


def e4_random_bst():
    small = []
    for n in range(1, 9):
        tot, cnt, hmax_tot = 0, 0, 0
        for perm in itertools.permutations(range(1, n + 1)):
            t, _ = T.build("bst", perm)
            d = T.depths(t.root)
            tot += sum(d.values()) / n
            hmax_tot += T.height(t.root)
            cnt += 1
        small.append({"n": n, "orders": cnt, "mean_depth": tot / cnt,
                      "formula": 2 * (1 + 1 / n) * _harmonic(n) - 3, "mean_height": hmax_tot / cnt})
    big = []
    for n in (1000, 5000):
        ms = []
        for s in range(SEEDS):
            keys = T.orders("random", n, random.Random(s))
            t, _ = T.build("bst", keys)
            ms.append(T.mean_search_comparisons(t.root, keys))
        big.append({"n": n, "mean_depth": statistics.mean(ms), "formula": 2 * (1 + 1 / n) * _harmonic(n) - 3,
                    "two_ln_n": 2 * math.log(n), "log2_n": math.log2(n)})
    return {"small_exhaustive": small, "big": big}


class _Cnt:
    """sorted() 의 비교를 세려고 감싼 값."""
    calls = 0

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        _Cnt.calls += 1
        return self.v < other.v


def e5_heap():
    rows = []
    for n in (1000, 10_000, 100_000):
        for name in ("random", "ascending", "descending"):
            xs = list(range(n))
            if name == "random":
                random.Random(0).shuffle(xs)
            elif name == "descending":
                xs.reverse()
            c1 = T.Counter()
            h1 = T.BinaryHeap(c1)
            h1.heapify(xs)
            c2 = T.Counter()
            h2 = T.BinaryHeap(c2)
            for x in xs:
                h2.push(x)
            ok = T.check_heap(h1.a)[1] and T.check_heap(h2.a)[1]
            arr = np.array(h1.a)
            idx = np.arange(1, n)
            np_ok = bool((arr[(idx - 1) // 2] <= arr[idx]).all())
            rows.append({"n": n, "input": name, "heapify": c1.sift, "pushes": c2.sift, "ratio": c2.sift / c1.sift,
                         "two_n": 2 * n, "n_log2_n": n * math.log2(n), "heap_ok": ok and np_ok})
    # 꺼내는 순서가 heapq 와 같은가
    xs = [random.Random(1).randrange(10 ** 6) for _ in range(20_000)]
    h = T.BinaryHeap()
    h.heapify(xs)
    mine = [h.pop() for _ in range(len(xs))]
    ref = list(xs)
    heapq.heapify(ref)
    theirs = [heapq.heappop(ref) for _ in range(len(xs))]
    same = mine == theirs == sorted(xs)
    # 상위 k 개만 남기기(빔 서치의 후보 고르기와 같은 일): 크기 k 힙 vs 전체 정렬
    N, topk = 100_000, []
    rng = random.Random(2)
    scores = [rng.random() for _ in range(N)]
    for k in (5, 50):
        c = T.Counter()
        hk = T.BinaryHeap(c)
        for s in scores:
            if len(hk) < k:
                hk.push(s)
            else:
                c.sift += 1
                if s > hk.a[0]:
                    hk.a[0] = s
                    hk._sift_down(0)
        best = sorted(hk.a, reverse=True)
        _Cnt.calls = 0
        full = sorted((_Cnt(s) for s in scores), reverse=True)
        topk.append({"N": N, "k": k, "heap_comparisons": c.sift, "sort_comparisons": _Cnt.calls,
                     "same_answer": best == [x.v for x in full[:k]] == heapq.nlargest(k, scores)})
    return {"rows": rows, "pop_order_matches_heapq": same, "topk": topk}


def e6_traces():
    n = 10
    sets = {"sorted": T.orders("sorted", n), "zigzag": T.orders("zigzag", n),
            "random": T.orders("random", n, random.Random(3))}
    tr = {}
    for name, keys in sets.items():
        for kind in ("bst", "avl"):
            tr[f"{name}_{kind}"] = T.trace_inserts(keys, kind)
    return {"n": n, "orders": sets, "traces": tr}


def main():
    t0 = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                   "platform": platform.platform(), "recursion_limit": sys.getrecursionlimit(),
                   "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. NumPy 스레드 2개"}}
    for key, fn in (("E0_first_screen", e0_first_screen), ("E2_recursion", e2_recursion), ("E3_avl_bound", e3_avl_bound),
                    ("E4_random_bst", e4_random_bst), ("E5_heap", e5_heap), ("E6_widget", e6_traces),
                    ("E1_grid", e1_grid)):
        res[key] = fn()
        print(key, round(time.perf_counter() - t0, 1), flush=True)
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    show = {k: v for k, v in res.items() if k != "E6_widget"}
    print(json.dumps(show, ensure_ascii=False, indent=1)[:12000])


if __name__ == "__main__":
    main()
