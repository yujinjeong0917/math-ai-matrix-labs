"""알고리즘 9장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch09_divide_conquer/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 12 x 34, 1213 x 2121, 자릿수 2^k 에서 4^k 와 3^k
E1 실패 최소 예제: 학교식의 한 자리 곱이 정확히 n^2 (n = 2^4..2^12, 시드 10개)
E2 비교 측정: 학교식, Karatsuba(자리올림 판, 계수 판, 기저 1/8/32), FFT 곱의 곱 횟수·더하기 횟수·실측 시간
E3 재귀 트리: 층별 노드 수와 작업량(곱 3번 vs 4번)
E4 FFT 반올림 한계: 2^bits 진법 계수로 쪼갠 NumPy FFT 곱이 틀리기 시작하는 크기
E5 CPython int 곱셈: 실측 시간과 Karatsuba 전환 크기(소스에서 확인한 값)
E6 AI 연결: 긴 1차원 합성곱을 직접 계산할 때와 FFT 로 계산할 때
E7 위젯 트레이스: 8자리 두 수의 계수 판 Karatsuba(곱 3번)와 같은 쪼개기의 곱 4번 판

결과는 results/ch09.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
곱 횟수는 입력이 같으면 항상 같다(계수 판과 학교식은 입력과도 무관하다). 실측 시간은 기기와 부하에 따라 달라진다.
"""

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
import algorithms_ch09_mul as mm  # noqa: E402

OUT = HERE / "results" / "ch09.json"
# 자릿수 n = 2^k, k = 4..12 (16 ~ 4096 자리).
KS = range(4, 13)
# 크기마다 무작위 수 쌍 10개.
SEEDS = 10


def rand_n_digits(rng, n):
    # 정확히 n 자리(첫 자리가 0 이 아님)인 정수.
    return rng.randrange(10 ** (n - 1), 10 ** n)


def e0_first_screen():
    # 12 = 1*10 + 2, 34 = 3*10 + 4 의 조각(a1, a0, b1, b0).
    a1, a0, b1, b0 = 1, 2, 3, 4
    z2, z0 = a1 * b1, a0 * b0
    s = (a1 + a0) * (b1 + b0)
    z1 = s - z2 - z0
    c_s, c_k = mm.Counter(), mm.Counter()
    v_s = mm.schoolbook_mul(12, 34, c_s)
    v_k = mm.karatsuba_poly_mul(12, 34, 1, c_k)
    # 네 자리 예: 쪼갠 조각을 더해도 모든 자리가 9 이하로 남는 1213 x 2121 (손으로 따라 할 때 한 자리 곱만 나온다)
    c4s, c4k = mm.Counter(), mm.Counter()
    mm.schoolbook_mul(1213, 2121, c4s)
    v4 = mm.karatsuba_poly_mul(1213, 2121, 1, c4k)
    tr4 = []
    mm.karatsuba_poly(mm.to_digits(1213), mm.to_digits(2121), 1, None, tr4)
    leaves4 = [(p[0], q[0]) for d, p, q, r, _ in tr4 if len(p) == 1]
    table = [{"k": k, "digits": 2 ** k, "school": 4 ** k, "karatsuba": 3 ** k, "ratio": 4 ** k / 3 ** k}
             for k in range(1, 13)]
    return {
        "x": 12, "y": 34,
        "school_parts": {"a1b1": a1 * b1, "a1b0": a1 * b0, "a0b1": a0 * b1, "a0b0": a0 * b0},
        "karatsuba_parts": {"z2": z2, "z0": z0, "sum_a": a1 + a0, "sum_b": b1 + b0, "s": s, "z1": z1},
        "result": z2 * 100 + z1 * 10 + z0, "school_value": v_s, "karatsuba_value": v_k,
        "school_mults": c_s.mults, "karatsuba_mults": c_k.mults,
        "four_digit": {"x": 1213, "y": 2121, "value": v4, "school_mults": c4s.mults, "karatsuba_mults": c4k.mults,
                       "leaf_products": leaves4, "all_leaves_single_digit": all(0 <= a <= 9 and 0 <= b <= 9 for a, b in leaves4)},
        "table": table,
    }


def timed(fn):
    t0 = time.perf_counter()
    v = fn()
    return v, (time.perf_counter() - t0) * 1000


METHODS = {
    "school": lambda x, y, c: mm.schoolbook_mul(x, y, c),
    "kara_carry_b1": lambda x, y, c: mm.karatsuba_mul(x, y, 1, c),
    "kara_carry_b8": lambda x, y, c: mm.karatsuba_mul(x, y, 8, c),
    "kara_carry_b32": lambda x, y, c: mm.karatsuba_mul(x, y, 32, c),
    "kara_poly_b1": lambda x, y, c: mm.karatsuba_poly_mul(x, y, 1, c),
    "kara_poly_b8": lambda x, y, c: mm.karatsuba_poly_mul(x, y, 8, c),
    "kara_poly_b32": lambda x, y, c: mm.karatsuba_poly_mul(x, y, 32, c),
    # (fft_counted 는 10진 자릿수에 순수 Python 재귀 FFT 를 쓰는 판이다. 곱 횟수 비교가 목적이고 NumPy FFT 의 속도를 대표하지 않는다.)
    "fft_counted": lambda x, y, c: mm.fft_mul(x, y, c),
}


def e1_e2_grid():
    rows = []
    for k in KS:
        n = 2 ** k
        per = {name: {"mults": [], "adds": [], "ms": [], "correct": True} for name in METHODS}
        py_ms = []
        for seed in range(SEEDS):
            # k 와 seed 가 겹치지 않게 시드를 만든다.
            rng = random.Random(1000 * k + seed)
            x, y = rand_n_digits(rng, n), rand_n_digits(rng, n)
            # 정답은 파이썬 정수 곱. 모든 방법이 같은 값을 내는지(correct) 확인.
            exact = x * y
            for name, f in METHODS.items():
                c = mm.Counter()
                # 시간 측정은 1회(중앙값은 시드 10개에 대한 값). 곱 횟수는 결정적이지만 ms 는 기기 부하에 흔들린다.
                v, ms = timed(lambda: f(x, y, c))
                per[name]["mults"].append(c.mults)
                per[name]["adds"].append(c.adds)
                per[name]["ms"].append(ms)
                per[name]["correct"] &= (v == exact)
        row = {"k": k, "n": n, "n_squared": n * n, "three_k": 3 ** k}
        for name, d in per.items():
            row[name] = {
                "mults_min": min(d["mults"]), "mults_max": max(d["mults"]),
                "mults_mean": statistics.mean(d["mults"]),
                "adds_mean": statistics.mean(d["adds"]),
                "ops_mean": statistics.mean(m + a for m, a in zip(d["mults"], d["adds"])),
                "ms_median": statistics.median(d["ms"]),
                "ms_min": min(d["ms"]), "ms_max": max(d["ms"]),
                "correct_all": d["correct"],
            }
        rows.append(row)
        print("E2", n, {nm: round(row[nm]["ms_median"], 2) for nm in METHODS}, flush=True)
    # 공식 확인: 계수 판, n = 2^k, 기저 2^j -> 3^(k-j) 4^j
    formula = []
    for row in rows:
        k = row["k"]
        for j, name in ((0, "kara_poly_b1"), (3, "kara_poly_b8"), (5, "kara_poly_b32")):
            # 기저 크기가 자릿수보다 크면 쪼개지 않고 학교식 한 번이라 4^k.
            want = 3 ** (k - j) * 4 ** j if k >= j else 4 ** k
            formula.append({"k": k, "base": 2 ** j, "formula": want, "measured": row[name]["mults_min"],
                            "same_all_seeds": row[name]["mults_min"] == row[name]["mults_max"] == want})
    # FFT 복소수 곱 공식 3 (N/2) log2 N + N, N = 2n (곱의 길이 2n-1 이상인 2의 거듭제곱)
    fft_formula = []
    for row in rows:
        # FFT 크기 N = 2n (곱의 길이 2n-1 이상인 가장 작은 2 의 거듭제곱).
        N = 2 * row["n"]
        lg = N.bit_length() - 1
        want = 3 * (N // 2) * lg + N
        fft_formula.append({"n": row["n"], "N": N, "formula": want, "measured": row["fft_counted"]["mults_min"]})
    return {"seeds": SEEDS, "rows": rows, "poly_formula": formula, "fft_formula": fft_formula}


def e3_tree(n=4096):
    out = {}
    for b in (3, 4):
        lv = mm.recursion_tree(n, b, 1)
        out[str(b)] = lv
    return {"n": n, "levels": out,
            # 잎 층이 전체 작업에서 차지하는 비율: 3 갈래는 잎이 지배(마스터 정리 식으로 잎 우세), 4 갈래는 더 심하다.
            "leaf_share_3": out["3"][-1]["work"] / sum(l["work"] for l in out["3"]),
            "leaf_share_4": out["4"][-1]["work"] / sum(l["work"] for l in out["4"])}


def e4_fft_precision():
    rows = []
    # 계수 개수 256 .. 2^20.
    sizes = [2 ** e for e in range(8, 21)]
    for bits in (4, 8, 12, 16, 20, 24):
        B = 1 << bits
        for n in sizes:
            # 최악 입력(모든 자리 최댓값)이라 정답을 직접 계산할 수 있고 오차가 가장 크다. 위치 i 에서 겹치는 항 수가 min(i+1, 2n-1-i).
            # 모든 계수가 B-1 인 입력: 정확한 계수는 min(i+1, 2n-1-i) (B-1)^2 로 바로 안다
            p = np.full(n, B - 1, dtype=np.float64)
            f = mm.fft_conv_float(p, p)
            i = np.arange(2 * n - 1)
            exact = np.minimum(i + 1, 2 * n - 1 - i).astype(np.float64) * float((B - 1) ** 2)
            # exact_int 는 파이썬 정수(무한 정밀도)라 정확하다. 위의 float exact 는 오차 크기 측정용.
            # exact 가 2^53 을 넘으면 float64 로 정확히 담기지 않으니 정수로 비교
            exact_int = [min(t + 1, 2 * n - 1 - t) * (B - 1) ** 2 for t in range(2 * n - 1)]
            rounded = np.rint(f)
            wrong_max = sum(1 for r, e in zip(rounded.tolist(), exact_int) if int(r) != e)
            err_max = float(np.max(np.abs(f - exact)))
            # 가능한 최대 계수. log2 가 53 근처를 넘으면 반올림이 틀리기 시작할 것으로 예상(실제 경계는 측정 결과 참고).
            max_coeff = n * (B - 1) ** 2
            row = {"bits": bits, "n_coeffs": n, "total_bits": n * bits, "decimal_digits": int(n * bits * 0.30103),
                   "max_coeff_log2": round(math.log2(max_coeff), 2),
                   "allmax_err": err_max, "allmax_wrong_coeffs": wrong_max, "allmax_ok": wrong_max == 0}
            # (오라클 = 정답 기준기. 2^22 비트 이하일 때만 비교한다.)
            # 무작위 입력: 정확한 곱(Python int)과 비교. 큰 크기는 오라클이 느려서 총 4M비트까지만
            if n * bits <= 2 ** 22:
                rng = random.Random(bits * 100 + n)
                # 최상위 비트를 켜서 길이를 정확히 n * bits 비트로 고정.
                x = rng.getrandbits(n * bits) | (1 << (n * bits - 1))
                y = rng.getrandbits(n * bits) | (1 << (n * bits - 1))
                v, dist = mm.fft_mul_bits(x, y, bits)
                row["random_ok"] = (v == x * y)
                row["random_dist"] = dist
            rows.append(row)
        print("E4", bits, [(r["n_coeffs"], r["allmax_ok"]) for r in rows if r["bits"] == bits], flush=True)
    first_fail = {}
    for bits in (4, 8, 12, 16, 20, 24):
        fails = [r for r in rows if r["bits"] == bits and not r["allmax_ok"]]
        oks = [r for r in rows if r["bits"] == bits and r["allmax_ok"]]
        first_fail[str(bits)] = {
            "last_ok_n": max((r["n_coeffs"] for r in oks), default=None),
            "first_fail_n": min((r["n_coeffs"] for r in fails), default=None),
        }
    return {"sizes": sizes, "rows": rows, "first_fail": first_fail}


def e5_cpython():
    rows = []
    for k in KS:
        n = 2 ** k
        rng = random.Random(7 + k)
        x, y = rand_n_digits(rng, n), rand_n_digits(rng, n)
        # 작은 곱은 너무 빨라 여러 번 곱해 평균 시간을 낸다(단위 ms, 1회당).
        reps = 2000 if n <= 1024 else 300
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                x * y
            ts.append((time.perf_counter() - t0) / reps * 1000)
        rows.append({"n": n, "bits": x.bit_length(), "py_ms": statistics.median(ts),
                     # CPython 내부 자리 하나 = 30 비트. 올림 나눗셈 (비트 + 29)//30 이 자리 수. 70 자리를 넘으면 CPython 도 Karatsuba 를 쓴다(소스에서 확인한 값, 버전에 따라 다를 수 있음).
                     "above_cutoff": (x.bit_length() + 29) // 30 > 70 and (y.bit_length() + 29) // 30 > 70})
    return {
        "source": "CPython v3.12.13 Objects/longobject.c: #define KARATSUBA_CUTOFF 70, KARATSUBA_SQUARE_CUTOFF (2 * KARATSUBA_CUTOFF)",
        "bits_per_digit": sys.int_info.bits_per_digit,
        "cutoff_digits": 70, "cutoff_bits": 70 * sys.int_info.bits_per_digit,
        # 비트를 10진 자릿수로: 비트 x log10(2). 0.30103 = log10(2).
        "cutoff_decimal_digits": round(70 * sys.int_info.bits_per_digit * 0.30102999566, 1),
        "rows": rows,
    }


def e6_conv():
    rng = np.random.default_rng(0)
    L = 2 ** 14
    # AI 연결: 길이 L 신호와 길이 K 커널의 합성곱. 직접 계산은 L x K 곱, FFT 는 O((L+K) log(L+K)).
    sig = rng.standard_normal(L)
    rows = []
    for e in range(2, 15, 2):
        K = 2 ** e
        ker = rng.standard_normal(K)
        d_ts, f_ts = [], []
        for _ in range(7):
            t0 = time.perf_counter()
            d = np.convolve(sig, ker)
            d_ts.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            f = mm.fft_conv_float(sig, ker)
            f_ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"signal": L, "kernel": K, "direct_mults": L * K,
                     "direct_ms": statistics.median(d_ts), "fft_ms": statistics.median(f_ts),
                     # FFT 결과와 직접 합성곱의 최대 절댓값 차(부동소수 오차 크기).
                     "max_abs_diff": float(np.max(np.abs(d - f)))})
    return {"rows": rows}


def _fmt(coeffs):
    """낮은 자리부터의 계수 리스트를 높은 자리부터 읽는 문자열로. 계수가 9를 넘으면 쉼표로 띄운다."""
    # 낮은 자리부터 저장된 리스트를 사람이 읽는 순서(높은 자리 먼저)로 뒤집는다.
    hi = list(reversed(coeffs))
    if all(0 <= v <= 9 for v in hi):
        return "".join(str(v) for v in hi)
    return "[" + ",".join(str(v) for v in hi) + "]"


def e7_widget(x=31415926, y=27182818):
    a, b = mm.to_digits(x), mm.to_digits(y)
    out = {"x": x, "y": y, "product": x * y}
    for name, fn in (("kara", mm.karatsuba_poly), ("school", mm.schoolbook_split_poly)):
        tr = []
        c = mm.Counter()
        res = fn(a, b, 1, c, tr)
        steps, level_counts = [], {}
        for depth, p, q, r, cum in tr:
            level_counts[depth] = level_counts.get(depth, 0) + 1
            # 트레이스 각 단계의 부분 곱이 맞는지 위젯에 표시하려는 검산.
            ok = mm.from_digits(p) * mm.from_digits(q) == mm.from_digits(r)
            steps.append([depth, _fmt(p), _fmt(q), mm.from_digits(r), cum, ok])
        out[name] = {"mults": c.mults, "result": mm.from_digits(res), "steps": steps,
                     "level_counts": [level_counts[d] for d in sorted(level_counts)],
                     "all_ok": all(s[5] for s in steps)}
    return out


def main():
    t0 = time.perf_counter()
    res = {
        "env": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            # (note 문자열은 결과 json 에 그대로 저장되는 값)
            "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정. NumPy 스레드 2개",
        },
        "E0_first_screen": e0_first_screen(),
    }
    res["E1_E2_grid"] = e1_e2_grid()
    res["E3_tree"] = e3_tree()
    res["E4_fft_precision"] = e4_fft_precision()
    res["E5_cpython"] = e5_cpython()
    res["E6_conv"] = e6_conv()
    res["E7_widget"] = e7_widget()
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", OUT, res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
