"""알고리즘 4장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch04_hashing/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 사물함 8칸, 섞인 키와 8의 배수 키, 범용 해시 한 번(p = 43, a = 6, b = 1)
E1 실패 최소 예제와 비교 측정: m = 1024, n = 1000..8000, 키 2종 x 해시 2종 x 시드 10
E2 부하율 alpha = 0.5, 1, 2, 4 에서 찾기 비교 수와 정확한 식 1 + (n-1)/(2m), 없는 키 찾기와 alpha
E3 크기 재조정은 고정 해시를 구하지 못한다: 2^20 의 배수 키, 1024 의 배수 키
E4 시드가 새면: a, b 를 아는 공격자가 같은 칸 키를 무차별 대입으로 찾기
E5 Carter-Wegman 보장의 전수 검사(p = 97, m = 8)와 대규모 표본 검사, NumPy 벡터화 해시와 체인 길이 대조
E6 Python dict: str 은 실행마다 소금을 치지만 int 는 hash(x) = x mod (2^61 - 1) 이라 같은 해시를 만들 수 있다
E7 위젯 트레이스(m = 8)

결과는 results/ch04.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
비교 수는 입력과 시드가 같으면 항상 같다. 실측 시간은 기기와 그 순간의 부하에 따라 달라진다.
"""

import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch04_hash as hh  # noqa: E402

OUT = HERE / "results" / "ch04.json"
M = 1024
NS = (1000, 2000, 4000, 8000)
SEEDS = 10


def e0_first_screen():
    m = 8
    mixed = [3, 12, 25, 30, 41]
    adv = [8, 16, 24, 32, 40]
    out = {"m": m}
    for name, keys in (("mixed", mixed), ("adversarial", adv)):
        t, c = hh.build(keys, m, hh.mod_hash(m))
        out[name] = {"keys": keys, "buckets": [k % m for k in keys], "comparisons": c.comparisons,
                     "max_chain": t.max_chain()}
    u = hh.UniversalHash(43, m, 6, 1)
    t, c = hh.build(adv, m, u)
    out["universal"] = {"p": 43, "a": 6, "b": 1, "keys": adv, "ax_plus_b": [6 * x + 1 for x in adv],
                        "mod_p": [(6 * x + 1) % 43 for x in adv], "buckets": [u(x) for x in adv],
                        "comparisons": c.comparisons, "max_chain": t.max_chain()}
    # 같은 키 5개를 모든 (a, b) 에 넣어 보면 몇 번이나 한 칸에 둘 이상 몰리나
    worst, any_collision, total = 0, 0, 0
    for a in range(1, 43):
        for b in range(43):
            h = hh.UniversalHash(43, m, a, b)
            tt, _ = hh.build(adv, m, h)
            worst = max(worst, tt.max_chain())
            any_collision += tt.max_chain() > 1
            total += 1
    out["universal_all_seeds"] = {"functions": total, "with_some_collision": any_collision, "worst_max_chain": worst}
    # 공격자가 a = 6, b = 1 을 알면: 6번 칸(키 8 이 가는 칸)으로 가는 작은 키 5개
    ks, tries = hh.keys_for_known_seed(u, 5, target=u(8), start=0)
    t2, c2 = hh.build(ks, m, u)
    out["known_seed"] = {"target": u(8), "keys": ks, "tries": tries, "comparisons": c2.comparisons, "max_chain": t2.max_chain()}
    return out


def _time_build(keys, m, h):
    t0 = time.perf_counter()
    t, c = hh.build(keys, m, h)
    return t, c, (time.perf_counter() - t0) * 1000


def e1_grid():
    rows = []
    for n in NS:
        row = {"n": n, "alpha": n / M, "n_n_minus_1_over_2": n * (n - 1) // 2,
               "uni_expected_bound_n_n_minus_1_over_2m": n * (n - 1) / (2 * M)}
        # 적대적 키 + mod: 입력도 해시도 무작위가 없어 한 번만
        adv = hh.adversarial_keys(M, n)
        t, c, ms = _time_build(adv, M, hh.mod_hash(M))
        row["adv_mod"] = {"comparisons": c.comparisons, "max_chain": t.max_chain(), "ms": ms, "runs": 1}
        cells = {"adv_uni": [], "rand_mod": [], "rand_uni": []}
        for seed in range(SEEDS):
            rng = random.Random(seed)
            rk = hh.random_keys(n, rng)
            hu = hh.universal_hash(hh.P31, M, random.Random(1000 + seed))
            for name, keys, h in (("adv_uni", adv, hu), ("rand_mod", rk, hh.mod_hash(M)), ("rand_uni", rk, hu)):
                t, c, ms = _time_build(keys, M, h)
                cells[name].append((c.comparisons, t.max_chain(), ms))
        for name, vals in cells.items():
            row[name] = {"comparisons_mean": statistics.mean(v[0] for v in vals), "comparisons_max": max(v[0] for v in vals),
                         "max_chain_mean": statistics.mean(v[1] for v in vals), "max_chain_max": max(v[1] for v in vals),
                         "ms_median": statistics.median(v[2] for v in vals), "runs": len(vals)}
        row["adv_uni"]["per_seed_comparisons"] = [v[0] for v in cells["adv_uni"]]
        row["ratio_adv_mod_to_adv_uni_comparisons"] = row["adv_mod"]["comparisons"] / row["adv_uni"]["comparisons_mean"]
        row["ratio_adv_mod_to_adv_uni_ms"] = row["adv_mod"]["ms"] / row["adv_uni"]["ms_median"]
        rows.append(row)
    return {"m": M, "seeds": SEEDS, "p": hh.P31, "rows": rows}


def e1b_many_seeds(n=8000, seeds=200):
    """적대적 키(1024 의 배수) 8000개를 시드 200개의 범용 해시에 넣어, 기대값 보장과 운 나쁜 시드를 함께 본다."""
    adv = hh.adversarial_keys(M, n)
    rand = hh.random_keys(n, random.Random(0))
    cs, chains, rc = [], [], []
    for s in range(seeds):
        h = hh.universal_hash(hh.P31, M, random.Random(1000 + s))
        t, c = hh.build(adv, M, h)
        cs.append(c.comparisons)
        chains.append(t.max_chain())
        t2, c2 = hh.build(rand, M, h)
        rc.append(c2.comparisons)
    bound = n * (n - 1) / (2 * M)
    return {"n": n, "m": M, "seeds": seeds, "bound": bound, "adv_mean": statistics.mean(cs), "adv_median": statistics.median(cs),
            "adv_max": max(cs), "adv_over_2x_bound": sum(c > 2 * bound for c in cs), "adv_max_chain_max": max(chains),
            "adv_max_chain_median": statistics.median(chains), "rand_mean": statistics.mean(rc), "rand_max": max(rc),
            "rand_over_2x_bound": sum(c > 2 * bound for c in rc), "mod_adv": n * (n - 1) // 2}


def e2_load():
    rows = []
    for alpha in (0.5, 1, 2, 4):
        n = int(alpha * M)
        succ, unsucc = [], []
        for seed in range(SEEDS):
            rng = random.Random(seed)
            allk = hh.random_keys(2 * n, rng)
            keys, absent = allk[:n], allk[n:]
            h = hh.universal_hash(hh.P31, M, random.Random(2000 + seed))
            t, _ = hh.build(keys, M, h)
            succ.append(hh.successful_search_cost(t, keys))
            unsucc.append(hh.unsuccessful_search_cost(t, absent))
        rows.append({"alpha": alpha, "n": n, "success_mean": statistics.mean(succ), "success_sd": statistics.pstdev(succ),
                     "exact_1_plus_n_minus_1_over_2m": 1 + (n - 1) / (2 * M), "approx_1_plus_alpha_over_2": 1 + alpha / 2,
                     "unsuccess_mean": statistics.mean(unsucc), "alpha_value": n / M})
    return {"m": M, "seeds": SEEDS, "rows": rows}


def e3_resize(n=4000):
    out = {"n": n, "start_m": 8, "max_load": 1.0}
    cases = {"mult_2_20": [i * 2 ** 20 for i in range(n)], "mult_1024": hh.adversarial_keys(1024, n),
             "random": hh.random_keys(n, random.Random(0), hi=2 ** 40)}
    for name, keys in cases.items():
        t, c = hh.build(keys, 8, hh.mod_hash(8), max_load=1.0)
        out[name] = {"final_m": t.m, "resizes": c.resizes, "max_chain": t.max_chain(),
                     "nonempty_buckets": sum(1 for x in t.chain_lengths() if x), "comparisons": c.comparisons}
    return out


def e4_known_seed(n=2000):
    h = hh.universal_hash(hh.P31, M, random.Random(4))
    t0 = time.perf_counter()
    keys, tries = hh.keys_for_known_seed(h, n, start=0)
    search_s = time.perf_counter() - t0
    t, c, ms = _time_build(keys, M, h)
    redraw = []
    for seed in range(SEEDS):
        h2 = hh.universal_hash(hh.P31, M, random.Random(5000 + seed))
        t2, c2, ms2 = _time_build(keys, M, h2)
        redraw.append((c2.comparisons, t2.max_chain(), ms2))
    return {"n": n, "m": M, "a": h.a, "b": h.b, "tries": tries, "tries_per_key": tries / n, "search_seconds": search_s,
            "largest_key": max(keys), "same_seed": {"comparisons": c.comparisons, "max_chain": t.max_chain(), "ms": ms},
            "formula": n * (n - 1) // 2,
            "redrawn_seed": {"comparisons_mean": statistics.mean(r[0] for r in redraw), "max_chain_max": max(r[1] for r in redraw),
                             "ms_median": statistics.median(r[2] for r in redraw), "runs": SEEDS}}


def e5_universality():
    # (1) 전수 검사: p = 97, m = 8, 모든 (a, b) 와 모든 키 쌍
    p, m = 97, 8
    xs = np.arange(p, dtype=np.int64)
    coll = np.zeros((p, p), dtype=np.int64)
    funcs = 0
    for a in range(1, p):
        hv = ((a * xs[None, :] + np.arange(p, dtype=np.int64)[:, None]) % p) % m  # 행: b
        for row in hv:
            coll += row[:, None] == row[None, :]
            funcs += 1
    np.fill_diagonal(coll, 0)
    off = coll[~np.eye(p, dtype=bool)]
    sizes = [len(range(r, p, m)) for r in range(m)]
    lemma6 = sum(s * (s - 1) for s in sizes)
    exhaustive = {"p": p, "m": m, "functions": funcs, "pairs": p * (p - 1) // 2, "min_collisions": int(off.min()),
                  "max_collisions": int(off.max()), "bound_p_p_minus_1_over_m": p * (p - 1) / m, "lemma6_count": lemma6,
                  "prob": int(off.max()) / funcs, "one_over_m": 1 / m}
    # (2) 표본 검사: p = 2^31 - 1, m = 1024, 적대적 키 쌍(1024 의 배수) 1000개 x 시드 1000개
    rng = random.Random(7)
    pairs = []
    while len(pairs) < 1000:
        i, j = rng.sample(range(1, 2 ** 20), 2)
        pairs.append((i * 1024, j * 1024))
    P = hh.P31
    X = np.array([q[0] for q in pairs], dtype=np.int64)
    Y = np.array([q[1] for q in pairs], dtype=np.int64)
    hits = 0
    srng = random.Random(8)
    for _ in range(1000):
        a, b = srng.randrange(1, P), srng.randrange(P)
        hits += int(np.count_nonzero(((a * X + b) % P) % M == ((a * Y + b) % P) % M))
    trials = 1000 * 1000
    phat = hits / trials
    sd = (1 / M * (1 - 1 / M) / trials) ** 0.5
    sampled = {"pairs": 1000, "seeds": 1000, "collisions": hits, "rate": phat, "one_over_m": 1 / M,
               "three_sd": 3 * sd, "within": phat <= 1 / M + 3 * sd, "mod_hash_rate": 1.0}
    # (3) NumPy 벡터화 해시와 테이블의 체인 길이가 같은가
    h = hh.universal_hash(P, M, random.Random(9))
    keys = hh.random_keys(8000, random.Random(9))
    t, _ = hh.build(keys, M, h)
    vec = ((h.a * np.array(keys, dtype=np.int64) + h.b) % P) % M
    same = np.bincount(vec, minlength=M).tolist() == t.chain_lengths()
    return {"exhaustive": exhaustive, "sampled": sampled, "numpy_bincount_matches_table": bool(same)}


DICT_SNIPPET = r"""
import json, random, sys, time
P = sys.hash_info.modulus
out = {"modulus": P, "hash_of_multiples": [hash(k * P) for k in range(1, 6)], "rows": []}
for n in (1000, 2000, 4000, 8000):
    bad = [k * P for k in range(1, n + 1)]
    good = random.Random(0).sample(range(1, 2 ** 60), n)
    row = {"n": n}
    for name, keys in (("same_hash", bad), ("random", good)):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); d = {}
            for k in keys:
                d[k] = 1
            ts.append(time.perf_counter() - t0)
        row[name + "_ms"] = sorted(ts)[1] * 1000
    row["ratio"] = row["same_hash_ms"] / row["random_ms"]
    out["rows"].append(row)
print(json.dumps(out))
"""


def e6_python_dict():
    out = json.loads(subprocess.run([sys.executable, "-c", DICT_SNIPPET], capture_output=True, text=True, check=True).stdout)
    # str 은 실행마다 바뀌고 PYTHONHASHSEED 를 고정하면 같아진다. int 는 늘 같다
    def h(seed, expr):
        env = dict(os.environ)
        if seed is None:
            env.pop("PYTHONHASHSEED", None)
        else:
            env["PYTHONHASHSEED"] = str(seed)
        return int(subprocess.run([sys.executable, "-c", f"print(hash({expr}))"], capture_output=True, text=True, env=env).stdout)
    out["str_hash_two_runs_differ"] = h(None, "'token'") != h(None, "'token'")
    out["str_hash_fixed_seed_same"] = h(0, "'token'") == h(0, "'token'")
    out["int_hash_two_runs"] = [h(None, "12345"), h(None, "12345")]
    hi = sys.hash_info
    out["hash_info"] = {"algorithm": hi.algorithm, "hash_bits": hi.hash_bits, "seed_bits": hi.seed_bits, "width": hi.width,
                        "modulus": hi.modulus}
    return out


def e7_traces():
    m = 8
    adv = [8, 16, 24, 32, 40, 48, 56, 64]
    mixed = [3, 12, 25, 30, 41, 52, 66, 79]
    tr = {"adv_mod": hh.trace_inserts(adv, hh.mod_hash(m)), "mixed_mod": hh.trace_inserts(mixed, hh.mod_hash(m))}
    seeds = []
    for s in range(3):
        u = hh.universal_hash(97, m, random.Random(30 + s))
        seeds.append({"a": u.a, "b": u.b})
        tr[f"adv_uni{s}"] = hh.trace_inserts(adv, u)
        tr[f"mixed_uni{s}"] = hh.trace_inserts(mixed, u)
    return {"m": m, "p": 97, "seeds": seeds, "traces": tr}


def main():
    t0 = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                   "platform": platform.platform(),
                   "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. NumPy 스레드 2개"}}
    for key, fn in (("E0_first_screen", e0_first_screen), ("E2_load", e2_load), ("E3_resize", e3_resize),
                    ("E4_known_seed", e4_known_seed), ("E5_universality", e5_universality), ("E6_python_dict", e6_python_dict),
                    ("E7_widget", e7_traces), ("E1_grid", e1_grid), ("E1b_many_seeds", e1b_many_seeds)):
        res[key] = fn()
        print(key, round(time.perf_counter() - t0, 1), flush=True)
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("E7_widget",)}, ensure_ascii=False, indent=1)[:9000])


if __name__ == "__main__":
    main()
