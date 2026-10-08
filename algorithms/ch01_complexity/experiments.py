"""알고리즘 1장 실험. `OMP_NUM_THREADS=2 uv run python algorithms/ch01_complexity/experiments.py` 로 실행한다.

입력은 모두 중복이 없는 정수 리스트(시드 고정)다. 중복이 없으면 세 방법 모두 끝까지 돌아야 해서
모든 쌍 비교에는 가장 나쁜 입력이 된다.

E0 첫 화면 손계산: 모든 쌍 비교 횟수 n(n-1)/2 (n=4, 8, 16)
E1 연산 수는 결정적인가: 같은 입력 10번, 시드 10개에서 비교 횟수
E2 실측 시간은 흔들리나: n=1000, 같은 입력 10번 잰 1회 호출 시간의 최소·사분위·최대(묶음 측정)
E2b 묶지 않고 한 번씩 잰 시간: n=10, 1000
E3 길이별 비교: n=10..10^4 에서 네 방법의 연산 수와 시간 중앙값
E4 순위가 뒤집히는 곳: 연산 수로는 언제 정렬이 이기고, 시간으로는 언제 이기나
E5 증가율 띠: f(n)/n^2, f(n)/(n log2 n) 이 n 이 커질 때 어느 범위에 머무나
E6 다른 인터프리터: 같은 코드를 Python 3.9, 3.14 에서 다시 재서 시간 비율과 연산 수 비교

결과는 results/ch01.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
실측 시간은 기기와 그 순간의 부하에 따라 달라진다. 연산 수는 같은 Python 버전이면 같아야 한다.
"""

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import algorithms_ch01_dup as dup  # noqa: E402
import algorithms_ch01_timing as tm  # noqa: E402

OUT = HERE / "results" / "ch01.json"
METHODS = ("pairs", "sorted", "msort", "set")


def e0_first_screen():
    return {str(n): dup.count_ops("pairs", list(range(n))) for n in (4, 8, 16)}


def e1_determinism():
    xs = dup.distinct_list(1000, 0)
    same_input = {m: sorted({dup.count_ops(m, xs) for _ in range(10)}) for m in METHODS}
    by_seed = {}
    for n in (10, 100, 1000):
        by_seed[str(n)] = {m: [dup.count_ops(m, dup.distinct_list(n, s)) for s in range(10)] for m in METHODS}
    return {"same_input_n1000_distinct_values": same_input, "by_seed": by_seed}


def e2_jitter():
    r = tm.run([1000], repeats=10, seed=0, methods=METHODS)
    return {row["method"]: {k: row[k] for k in ("min", "q1", "median", "q3", "max", "max_over_min", "ops", "number", "times_s")}
            for row in r["rows"]}


def e2b_single_call():
    """묶지 않고 한 번씩 잰다. 호출 하나가 짧으면 타이머·운영체제 잡음이 그대로 드러난다."""
    import time

    out = {}
    for n in (10, 1000):
        xs = dup.distinct_list(n, 0)
        row = {}
        for m in ("pairs", "sorted"):
            f = dup.METHODS[m]
            f(xs)  # 첫 호출(데우기)은 버린다
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                f(xs)
                ts.append(time.perf_counter() - t0)
            row[m] = tm.summarize(ts)
            row[m]["times_s"] = ts
        out[str(n)] = row
    return out


def e3_table():
    ns = (10, 100, 1000, 10000)
    r = tm.run(list(ns), repeats=10, seed=0, methods=METHODS)
    out = {}
    for row in r["rows"]:
        out.setdefault(str(row["n"]), {})[row["method"]] = {k: row[k] for k in ("ops", "min", "q1", "median", "q3", "max", "number")}
    return out


def _first_win(grid, a, b):
    """grid 에서 b 가 a 보다 작아진 뒤 끝까지 그대로인 첫 n."""
    first = None
    for n in grid:
        if b[n] < a[n]:
            if first is None:
                first = n
        else:
            first = None
    return first


def e4_crossover():
    grid = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256]
    r = tm.run(grid, repeats=10, seed=0, methods=("pairs", "sorted", "msort"))
    med = {m: {} for m in ("pairs", "sorted", "msort")}
    ops = {m: {} for m in ("pairs", "sorted", "msort")}
    for row in r["rows"]:
        med[row["method"]][row["n"]] = row["median"]
        ops[row["method"]][row["n"]] = row["ops"]
    return {
        "grid": grid,
        "median_s": {m: {str(n): v for n, v in d.items()} for m, d in med.items()},
        "ops": {m: {str(n): v for n, v in d.items()} for m, d in ops.items()},
        "ops_msort_beats_pairs_from_n": _first_win(grid, ops["pairs"], ops["msort"]),
        "ops_sorted_beats_pairs_from_n": _first_win(grid, ops["pairs"], ops["sorted"]),
        "time_msort_beats_pairs_from_n": _first_win(grid, med["pairs"], med["msort"]),
        "time_sorted_beats_pairs_from_n": _first_win(grid, med["pairs"], med["sorted"]),
    }


def e5_bands():
    ns = [2 ** k for k in range(4, 15)]  # 16 .. 16384
    rows = []
    for n in ns:
        xs = dup.distinct_list(n, 0)
        f_pairs = n * (n - 1) // 2  # test 에서 세어서 확인한 값
        f_ms = dup.count_ops("msort", xs)
        f_so = dup.count_ops("sorted", xs)
        nl = n * math.log2(n)
        rows.append({"n": n, "pairs_over_n2": f_pairs / n ** 2, "msort_over_nlogn": f_ms / nl,
                     "sorted_over_nlogn": f_so / nl, "msort_ops": f_ms, "sorted_ops": f_so})
    return {"rows": rows,
            "msort_band": [min(r["msort_over_nlogn"] for r in rows), max(r["msort_over_nlogn"] for r in rows)],
            "sorted_band": [min(r["sorted_over_nlogn"] for r in rows), max(r["sorted_over_nlogn"] for r in rows)]}


def e6_interpreters():
    cfg = {"ns": [100, 1000], "repeats": 10, "seed": 0, "methods": ["pairs", "sorted", "msort", "set"]}
    cands = [("this", sys.executable), ("system", "/usr/bin/python3"), ("py314", shutil.which("python3.14"))]
    out = {}
    for name, exe in cands:
        if not exe or not Path(exe).exists():
            out[name] = {"skipped": "interpreter not found"}
            continue
        env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([exe, str(HERE / "algorithms_ch01_timing.py"), json.dumps(cfg)],
                           capture_output=True, text=True, env=env, timeout=600)
        if p.returncode != 0:
            out[name] = {"skipped": p.stderr[-300:]}
            continue
        r = json.loads(p.stdout)
        rows = {f'{row["n"]}/{row["method"]}': {"median": row["median"], "ops": row["ops"], "checksum": row["checksum"]} for row in r["rows"]}
        ratio = {str(n): rows[f"{n}/pairs"]["median"] / rows[f"{n}/sorted"]["median"] for n in cfg["ns"]}
        out[name] = {"python": r["python"], "rows": rows, "pairs_over_sorted_time": ratio}
    return out


def main():
    import numpy as np  # 환경 기록용

    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                "platform": platform.platform(), "note": "다른 에이전트 작업이 같은 기기에서 동시에 돌던 상태에서 측정"},
        "E0_first_screen_pair_comparisons": e0_first_screen(),
        "E1_determinism": e1_determinism(),
        "E2_jitter_n1000": e2_jitter(),
        "E2b_single_call": e2b_single_call(),
        "E3_table": e3_table(),
        "E4_crossover": e4_crossover(),
        "E5_bands": e5_bands(),
        "E6_interpreters": e6_interpreters(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E0_first_screen_pair_comparisons",)}, ensure_ascii=False))
    e4 = res["E4_crossover"]
    print({k: v for k, v in e4.items() if k.endswith("_from_n")})
    print(json.dumps(res["E6_interpreters"], indent=1)[:2000])


if __name__ == "__main__":
    main()
