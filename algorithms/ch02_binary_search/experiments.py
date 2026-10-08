"""알고리즘 2장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch02_binary_search/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 카드 8장 [3,5,8,12,15,19,23,30]에서 16 이상인 첫 카드 찾기(올바른 판, lo = mid 판)
E1 전수 검사: 길이 0~5 정렬 배열(값 0,2,4,6,8 중복 허용) x 목표값 -1~9, 변형 a~d 결과 수와 가장 짧은 반례
E2 오버플로 흉내: np.int32 로 (low+high)//2 와 low+(high-low)//2
E3 무작위 성질 검사: 시드 0, 무작위 정렬 배열 3000개에서 bisect_left 와 일치, 반복 수 상한, 매 단계 불변식
E4 반복 횟수 상한이 꽉 차는가: n 마다 모든 x 에 대한 최대 반복 수 = ceil(log2(n+1))
E5 선형 vs 이분: n = 10^3..10^6 비교 횟수(결정적)와 실측 시간(10회 중앙값), bisect 모듈과 비교
E6 CPython bpo-13496: range(sys.maxsize) 에서 bisect 가 지금은 끝나는지, 옛 식을 64비트로 흉내 내면 어디서 넘치는지
E7 AI 연결: 누적확률에서 표본 위치 찾기, 칸을 다 세는 방식(LLM 11장)과 이분 탐색이 같은 답인지

결과는 results/ch02.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
실측 시간은 기기와 그 순간의 부하에 따라 달라진다. 비교 횟수와 전수 검사 결과는 같아야 한다.
"""

import bisect
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
import algorithms_ch02_bsearch as bs  # noqa: E402

OUT = HERE / "results" / "ch02.json"
CARDS = [3, 5, 8, 12, 15, 19, 23, 30]


def e0_first_screen():
    good = bs.run_variant("d_correct", CARDS, 16, trace=True)
    stuck = bs.run_variant("b_lo_mid", CARDS, 16, trace=True)
    return {
        "cards": CARDS, "x": 16,
        "correct": {"answer": good["answer"], "card": CARDS[good["answer"]], "steps": good["steps"],
                    "states": [r["state"] for r in good["trace"]]},
        "lo_eq_mid": {"outcome": stuck["outcome"], "states_first6": [r["state"] for r in stuck["trace"][:6]]},
        "bound_8": bs.max_steps_bound(8),
        "bound_1e6": bs.max_steps_bound(10 ** 6),
        "log2_1e6_plus_1": math.log2(10 ** 6 + 1),
    }


def e1_exhaustive():
    r = bs.exhaustive(5)
    return {"max_len": 5, "values": [0, 2, 4, 6, 8], "targets": bs.targets(), "max_steps_cap": bs.MAX_STEPS,
            "n_arrays": len(list(bs.sorted_arrays(5))), "n_targets": len(bs.targets()),
            "code": bs.VARIANT_CODE, "results": r}


def e3_random_props(n_arrays=3000, seed=0):
    rng = random.Random(seed)
    agree = 0
    step_ok = 0
    inv_ok = 0
    checked_states = 0
    variant_fail = {k: 0 for k in bs.VARIANTS}
    variant_shortest = {k: None for k in bs.VARIANTS}
    for _ in range(n_arrays):
        n = rng.randint(0, 40)
        a = sorted(rng.randint(0, 20) for _ in range(n))
        x = rng.randint(-1, 21)
        c = bs.Counter()
        got = bs.lower_bound(a, x, c)
        agree += got == bisect.bisect_left(a, x)
        step_ok += c.steps <= bs.max_steps_bound(n)
        t = bs.run_variant("d_correct", a, x, trace=True)
        ok = all(i["holds"] for row in t["trace"] for i in row["invariants"])
        ok = ok and all(row["state"].get("shrank", True) for row in t["trace"])
        checked_states += len(t["trace"])
        inv_ok += ok
        for k in bs.VARIANTS:
            r = bs.run_variant(k, a, x)
            if r["outcome"] != "ok":
                variant_fail[k] += 1
                cur = variant_shortest[k]
                if cur is None or len(a) < len(cur["a"]):
                    variant_shortest[k] = {"a": a, "x": x, "outcome": r["outcome"]}
    return {"seed": seed, "n_arrays": n_arrays, "len_range": [0, 40], "value_range": [0, 20],
            "agree_with_bisect_left": agree, "steps_within_bound": step_ok, "invariant_and_shrink_every_step": inv_ok,
            "states_checked": checked_states, "variant_fail": variant_fail, "variant_shortest_found": variant_shortest}


def e4_tight_bound():
    rows = []
    for n in list(range(1, 33)) + [100, 1000, 10 ** 6]:
        a = list(range(n))
        worst = 0
        for x in range(-1, n + 1):
            c = bs.Counter()
            bs.lower_bound(a, x, c)
            worst = max(worst, c.steps)
        rows.append({"n": n, "max_steps": worst, "bound": bs.max_steps_bound(n)})
    return {"rows": rows, "all_tight": all(r["max_steps"] == r["bound"] for r in rows)}


def _median_time(fn, repeats=10):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def e5_linear_vs_binary(repeats=10, n_queries=20, seed=0):
    out = {}
    for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        rng = random.Random(seed)
        a = sorted(rng.sample(range(10 * n), n))
        arr = np.asarray(a)
        qs = [rng.randrange(-1, 10 * n + 1) for _ in range(n_queries)]
        # 비교 횟수(결정적): 같은 질의 묶음의 평균, 그리고 최악(x 가 모두보다 큰 경우)
        lin_c = []
        bin_c = []
        for x in qs:
            c1, c2 = bs.Counter(), bs.Counter()
            assert bs.linear_lower_bound(a, x, c1) == bs.lower_bound(a, x, c2) == bisect.bisect_left(a, x)
            lin_c.append(c1.comparisons)
            bin_c.append(c2.comparisons)
        w1, w2 = bs.Counter(), bs.Counter()
        bs.linear_lower_bound(a, 10 * n + 1, w1)
        bs.lower_bound(a, 10 * n + 1, w2)

        def run(f):
            return lambda: [f(a, x) for x in qs]

        t_lin = _median_time(run(bs.linear_lower_bound), repeats)
        t_bin = _median_time(run(bs.lower_bound), repeats)
        t_bis = _median_time(run(bisect.bisect_left), repeats)
        t_np = _median_time(lambda: [np.searchsorted(arr, x) for x in qs], repeats)
        per = 1e6 / n_queries  # 질의 1개당 마이크로초
        out[str(n)] = {
            "comparisons_mean": {"linear": statistics.mean(lin_c), "binary": statistics.mean(bin_c)},
            "comparisons_worst": {"linear": w1.comparisons, "binary": w2.comparisons, "bound": bs.max_steps_bound(n)},
            "us_per_query_median": {"linear": t_lin[0] * per, "binary_py": t_bin[0] * per,
                                    "bisect_c": t_bis[0] * per, "np_searchsorted_scalar": t_np[0] * per},
            "us_per_query_min_max": {"linear": [t_lin[1] * per, t_lin[2] * per], "binary_py": [t_bin[1] * per, t_bin[2] * per],
                                     "bisect_c": [t_bis[1] * per, t_bis[2] * per]},
        }
    return {"repeats": repeats, "n_queries": n_queries, "seed": seed, "rows": out}


def e6_cpython_issue():
    """bpo-13496(2011): C 모듈 _bisect 가 Py_ssize_t 로 (lo + hi) / 2 를 계산해, len 이 아주 큰
    가짜 시퀀스(range)에서 넘쳤다. 2012-04-15 에 (size_t) 변환으로 고쳤다."""
    big = sys.maxsize
    t0 = time.perf_counter()
    got = bisect.bisect(range(big), big - 3)
    dt = time.perf_counter() - t0
    # 옛 식을 64비트 부호 있는 정수로 흉내 낸다(bisect_right, 목표 big-3)
    import warnings

    lo, hi = np.int64(0), np.int64(big)
    x = big - 3
    first_neg = None
    steps = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while lo < hi and steps < 200:
            mid = (lo + hi) // np.int64(2)
            steps += 1
            if mid < 0:
                first_neg = {"step": steps, "lo": int(lo), "hi": int(hi), "mid": int(mid)}
                break
            if x < int(mid):
                hi = mid
            else:
                lo = mid + np.int64(1)
    return {"python": platform.python_version(), "sys_maxsize": big, "bisect_result": got, "expected": big - 2,
            "seconds": dt, "old_formula_int64_first_negative_mid": first_neg}


def e7_sampling(seed=0, n_u=10000):
    rng = np.random.default_rng(seed)
    out = {}
    for V in (48, 32768):
        p = rng.random(V)
        p /= p.sum()
        cdf = np.cumsum(p)
        u = rng.random(n_u) * cdf[-1]
        lin = np.array([bs.sample_index_linear(cdf, ui) for ui in u[:2000]])
        bis = bs.sample_index_bisect(cdf, u[:2000])
        out[str(V)] = {"agree": int((lin == bis).sum()), "checked": int(len(lin)),
                       "comparisons_linear": V, "comparisons_bisect_max": bs.max_steps_bound(V)}
    return out


def traces_for_widget():
    cases = {
        "d_correct": (CARDS, 16),
        "b_lo_mid": (CARDS, 16),
        "c_closed_hi": (CARDS, 31),
        "a_le_with_hi_mid": (CARDS, 16),
    }
    out = {}
    for k, (a, x) in cases.items():
        r = bs.run_variant(k, a, x, trace=True)
        out[k] = {"a": a, "x": x, "outcome": r["outcome"], "answer": r["answer"], "truth": r["truth"],
                  "trace": r["trace"][:8]}
    return out


def main():
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                "platform": platform.platform(), "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정"},
        "E0_first_screen": e0_first_screen(),
        "E1_exhaustive": e1_exhaustive(),
        "E2_overflow": bs.mid_overflow_demo(),
        "E3_random_props": e3_random_props(),
        "E4_tight_bound": e4_tight_bound(),
        "E5_linear_vs_binary": e5_linear_vs_binary(),
        "E6_cpython_bpo13496": e6_cpython_issue(),
        "E7_sampling": e7_sampling(),
        "traces": traces_for_widget(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    e1 = res["E1_exhaustive"]["results"]
    print({k: (v["counts"], v["first_failure"]) for k, v in e1.items()})
    print(json.dumps(res["E5_linear_vs_binary"]["rows"], indent=1))
    print(res["E6_cpython_bpo13496"], res["E7_sampling"], res["E4_tight_bound"]["all_tight"])
    print({k: v for k, v in res["E3_random_props"].items() if k != "variant_shortest_found"})


if __name__ == "__main__":
    main()
