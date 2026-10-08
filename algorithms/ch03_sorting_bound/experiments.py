"""알고리즘 3장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch03_sorting_bound/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 카드 3장·4장·8장의 순서 가짓수와 필요한 질문 수, 정렬된 8장에 첫 원소 피벗 퀵정렬
E1 하한 전수 검사: n = 1..8 의 모든 순열에 정렬 7종을 돌려 최대·평균 비교 수와 ceil(log2 n!) 비교
E2 실패 최소 예제: 정렬된 입력 n = 500..8000 에 첫 원소 피벗 퀵정렬, 비교 수 = n(n-1)/2 인지, 깊이, 시간
E3 재귀판이 파이썬 재귀 한도에 걸리는 첫 n
E4 무작위 피벗의 기대 비교 수: 정확한 기대값(Lomuto) 2(n+1)H_n - 4n, Hoare 의 2N ln N, 하한과의 비
E5 비교 측정: 입력 5종 x 알고리즘 8종 x 시드 10, n = 2000
E6 median-of-3 killer: Musser 의 K_n 에서 median-of-3 퀵정렬과 introsort
E7 AI 연결: top-k 를 전체 정렬 대신 부분 선택으로(np.argpartition, torch.topk)

결과는 results/ch03.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
비교 횟수와 전수 검사는 입력이 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

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
import torch  # noqa: E402

torch.set_num_threads(2)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch03_sorts as s  # noqa: E402

OUT = HERE / "results" / "ch03.json"
CARDS8 = [5, 2, 7, 1, 8, 3, 6, 4]


def e0_first_screen():
    rows = []
    for n in (3, 4, 8):
        f = math.factorial(n)
        rows.append({"n": n, "orders": f, "bound": s.ceil_log2_factorial(n),
                     "two_pow_bound_minus1": 2 ** (s.ceil_log2_factorial(n) - 1), "two_pow_bound": 2 ** s.ceil_log2_factorial(n)})
    _, c_sorted = s.run_sort("lomuto_first", list(range(1, 9)))
    _, c_mixed = s.run_sort("lomuto_first", CARDS8)
    _, c_merge = s.run_sort("merge", list(range(1, 9)))
    return {"rows": rows, "sorted8_lomuto_first": c_sorted.comparisons, "sorted8_formula": 8 * 7 // 2,
            "mixed8": CARDS8, "mixed8_lomuto_first": c_mixed.comparisons, "sorted8_merge": c_merge.comparisons}


E1_SORTS = ["insertion", "merge", "heap", "lomuto_first", "hoare_random", "median3", "introsort"]


def e1_exhaustive(max_n=8):
    out = {}
    for n in range(1, max_n + 1):
        bound = s.ceil_log2_factorial(n)
        row = {"orders": math.factorial(n), "bound": bound, "log2_fact": s.log2_factorial(n), "algos": {}}
        for name in E1_SORTS:
            worst, total, below = 0, 0, 0
            for k, p in enumerate(itertools.permutations(range(n))):
                a, c = s.run_sort(name, list(p), seed=k)
                assert a == list(range(n))
                worst = max(worst, c.comparisons)
                total += c.comparisons
                below += c.comparisons < bound
            row["algos"][name] = {"max": worst, "mean": total / math.factorial(n), "inputs_below_bound": below,
                                  "max_ge_bound": worst >= bound, "mean_ge_log2_fact": total / math.factorial(n) >= s.log2_factorial(n) - 1e-12}
        out[str(n)] = row
    return out


def _timeit(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def e2_sorted_first_pivot():
    rows = []
    for n in (500, 1000, 2000, 4000, 8000):
        xs = list(range(n))
        t0 = time.perf_counter()
        a, c = s.run_sort("lomuto_first", xs)
        t_first = time.perf_counter() - t0
        hr = []
        for seed in range(10):
            a2, c2 = s.run_sort("hoare_random", xs, seed)
            hr.append(c2.comparisons)
        t_hr = _timeit(lambda: s.run_sort("hoare_random", xs, 0), 3)
        a3, c3 = s.run_sort("introsort_first", xs)
        t_if = _timeit(lambda: s.run_sort("introsort_first", xs), 3)
        rows.append({
            "n": n, "first_comparisons": c.comparisons, "formula": n * (n - 1) // 2, "first_max_depth": c.max_depth,
            "first_seconds": t_first, "hoare_random_mean": statistics.mean(hr), "hoare_random_max": max(hr),
            "hoare_random_seconds": t_hr, "introsort_first_comparisons": c3.comparisons, "introsort_first_seconds": t_if,
            "introsort_first_heapsort_sizes": c3.heapsort_sizes, "bound": s.ceil_log2_factorial(n),
        })
    return {"rows": rows}


def e3_recursion_limit():
    limit = sys.getrecursionlimit()

    def fails(n):
        try:
            s.lomuto_first_recursive(list(range(n)))
            return False
        except RecursionError:
            return True

    lo, hi = 1, 4000
    assert fails(hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if fails(mid):
            hi = mid
        else:
            lo = mid + 1
    c = s.Counter()
    s.lomuto_first_recursive(list(range(500)), c)
    return {"recursion_limit": limit, "first_failing_n": lo, "works_500": not fails(500), "fails_2000": fails(2000),
            "random_2000_ok": _random_rec_ok(2000), "depth_at_500": c.max_depth}


def _random_rec_ok(n):
    xs = list(range(n))
    random.Random(0).shuffle(xs)
    c = s.Counter()
    s.lomuto_first_recursive(xs, c)
    return {"ok": xs == list(range(n)), "max_depth": c.max_depth}


def e4_random_pivot_expectation():
    rows = []
    for n in (100, 1000, 10000):
        lr, hr = [], []
        for seed in range(10):
            xs = s.make_input("random", n, seed)
            lr.append(s.run_sort("lomuto_random", xs, seed)[1].comparisons)
            hr.append(s.run_sort("hoare_random", xs, seed)[1].comparisons)
        exact = s.lomuto_expected(n)
        rows.append({"n": n, "lomuto_random_mean": statistics.mean(lr), "lomuto_random_sd": statistics.pstdev(lr),
                     "lomuto_exact_expected": exact, "hoare_2nlnn": 2 * n * math.log(n),
                     "hoare_random_mean": statistics.mean(hr), "bound": s.ceil_log2_factorial(n),
                     "ratio_exact_to_bound": exact / s.ceil_log2_factorial(n), "n_log2_n": n * math.log2(n)})
    return {"rows": rows, "two_ln2": 2 * math.log(2)}


ALGOS = ["lomuto_first", "lomuto_random", "hoare_random", "median3", "introsort", "merge", "heap"]
INPUTS = ["random", "sorted", "reversed", "few_unique", "killer"]


def e5_grid(n=2000, seeds=10):
    out = {}
    bound = s.ceil_log2_factorial(n)
    for kind in INPUTS:
        out[kind] = {}
        for name in ALGOS:
            deterministic = kind in ("sorted", "reversed", "killer") and name not in s.RANDOMIZED
            comps, depth, times = [], [], []
            for seed in range(1 if deterministic else seeds):
                xs = s.make_input(kind, n, seed)
                t0 = time.perf_counter()
                a, c = s.run_sort(name, xs, seed)
                times.append(time.perf_counter() - t0)
                assert a == sorted(xs)
                comps.append(c.comparisons)
                depth.append(c.max_depth)
            out[kind][name] = {"mean": statistics.mean(comps), "max": max(comps), "max_depth": max(depth),
                               "ms_median": statistics.median(times) * 1000, "runs": len(comps),
                               "ratio_to_bound": statistics.mean(comps) / bound}
    return {"n": n, "seeds": seeds, "bound": bound, "log2_fact": s.log2_factorial(n), "rows": out}


def e6_killer():
    rows = []
    for n in (1000, 2000, 4000, 8000):
        xs = s.median3_killer(n)
        a, cm = s.run_sort("median3", xs)
        tm = _timeit(lambda: s.run_sort("median3", xs), 1)
        b, ci = s.run_sort("introsort", xs)
        ti = _timeit(lambda: s.run_sort("introsort", xs), 3)
        h, ch = s.run_sort("heap", xs)
        th = _timeit(lambda: s.run_sort("heap", xs), 3)
        rx = s.make_input("random", n, 0)
        _, cmr = s.run_sort("median3", rx)
        _, cir = s.run_sort("introsort", rx)
        _, chr_ = s.run_sort("heap", rx)
        k = n // 4
        theorem = cm.partition_sizes[:k] == [(2, n - 2 * j) for j in range(1, k + 1)]
        rows.append({"n": n, "theorem1_first_n_over_4_partitions": theorem,
                     "median3_comparisons": cm.comparisons, "median3_max_depth": cm.max_depth, "median3_seconds": tm,
                     "introsort_comparisons": ci.comparisons, "introsort_partitions": ci.partitions,
                     "introsort_depth_limit": 2 * int(math.floor(math.log2(n))), "introsort_heapsort_sizes": ci.heapsort_sizes,
                     "introsort_seconds": ti, "heap_comparisons": ch.comparisons, "heap_seconds": th,
                     "random_median3": cmr.comparisons, "random_introsort": cir.comparisons, "random_heap": chr_.comparisons,
                     "bound": s.ceil_log2_factorial(n)})
    # 무작위 입력에서 introsort 가 힙정렬로 넘긴 적이 있는지(시드 10개, n = 2000)
    calls = 0
    for seed in range(10):
        c = s.Counter()
        s.introsort(s.make_input("random", 2000, seed), c)
        calls += len(c.heapsort_sizes)
    return {"rows": rows, "k8_example": s.median3_killer(8), "k16_example": s.median3_killer(16),
            "random2000_introsort_heapsort_calls_over_10_seeds": calls}


def e7_topk(n=1_000_000, k=10, repeats=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    full = np.argsort(-x, kind="stable")[:k]
    part = np.argpartition(-x, k - 1)[:k]
    part = part[np.argsort(-x[part], kind="stable")]
    tv, ti = torch.topk(torch.from_numpy(x), k)
    t_sort = _timeit(lambda: np.argsort(-x)[:k], repeats)
    t_part = _timeit(lambda: np.argpartition(-x, k - 1)[:k], repeats)
    t_topk = _timeit(lambda: torch.topk(torch.from_numpy(x), k), repeats)
    return {"n": n, "k": k, "same_indices_sort_vs_partition": bool(np.array_equal(full, part)),
            "same_indices_sort_vs_torch": bool(np.array_equal(full, ti.numpy())),
            "ms_argsort": t_sort * 1000, "ms_argpartition": t_part * 1000, "ms_torch_topk": t_topk * 1000,
            "bound_full_sort_log2_fact": s.log2_factorial(n), "torch": torch.__version__}


def e8_derived():
    """본문에 쓰는 결정적 보조 숫자(시간 측정 없음)."""
    n = 2000
    _, ce = s.run_sort("lomuto_random", [5] * 1024, 0)
    _, ch = s.run_sort("hoare_random", [5] * 1024, 0)
    return {"lomuto_expected_2000": s.lomuto_expected(n), "ratio_expected_2000_to_bound": s.lomuto_expected(n) / s.ceil_log2_factorial(n),
            "n_log2_n_2000": n * math.log2(n), "stirling_log2_fact_2000": s.stirling_log2_factorial(n), "log2_fact_2000": s.log2_factorial(n),
            "all_equal_1024_lomuto_random": ce.comparisons, "all_equal_1024_hoare_random": ch.comparisons,
            "ratio_2nlnn_to_exact": {str(m): 2 * m * math.log(m) / s.lomuto_expected(m) for m in (100, 1000, 10000)},
            "first_pivot_8000_over_bound": (8000 * 7999 // 2) / s.ceil_log2_factorial(8000)}


def traces_for_widget():
    return {"mixed": s.trace_lomuto_first(CARDS8), "sorted": s.trace_lomuto_first(list(range(1, 9))),
            "equal": s.trace_lomuto_first([4] * 8)}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "platform": platform.platform(),
                "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. 스레드 2개"},
        "E0_first_screen": e0_first_screen(),
        "E1_exhaustive": e1_exhaustive(),
        "E3_recursion": e3_recursion_limit(),
        "E4_expectation": e4_random_pivot_expectation(),
        "E6_killer": e6_killer(),
        "E7_topk": e7_topk(),
        "traces": traces_for_widget(),
        "E8_derived": e8_derived(),
    }
    print("E0-E7(일부)", time.perf_counter() - t0, flush=True)
    res["E2_sorted_first_pivot"] = e2_sorted_first_pivot()
    print("E2", time.perf_counter() - t0, flush=True)
    res["E5_grid"] = e5_grid()
    print("E5", time.perf_counter() - t0, flush=True)
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("traces", "E1_exhaustive", "E5_grid")}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
