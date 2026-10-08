"""알고리즘 1장 시간 측정 핵심. 표준 라이브러리만 쓴다(Python 3.9 이상).

다른 인터프리터에서도 그대로 돌리려고 스크립트로도 실행된다.
    python3 algorithms_ch01_timing.py '{"ns":[10,100],"repeats":10,"seed":0}'
결과 JSON 을 표준 출력으로 낸다.

한 번 호출이 타이머 해상도보다 짧을 수 있어서, 묶음 하나가 target 초 이상이 되도록
호출 횟수(number)를 먼저 정하고, 그 묶음을 repeats 번 잰 뒤 1회 호출 시간으로 나눈다.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import algorithms_ch01_dup as dup  # noqa: E402


def calibrate(fn, target):
    """묶음 하나가 대략 target 초가 되는 호출 횟수를 찾는다."""
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = time.perf_counter() - t0
        if dt >= target:
            return number
        if dt >= target / 10:
            return int(number * target / dt) + 1
        number *= 2


def time_call(fn, repeats=10, target=0.02):
    """1회 호출 시간(초) repeats 개를 돌려준다."""
    number = calibrate(fn, target)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        out.append((time.perf_counter() - t0) / number)
    return out, number


def summarize(ts):
    s = sorted(ts)
    q = statistics.quantiles(s, n=4)
    return {"min": s[0], "q1": q[0], "median": statistics.median(s), "q3": q[2], "max": s[-1],
            "max_over_min": s[-1] / s[0]}


def run(ns, repeats=10, seed=0, target=0.02, methods=("pairs", "sorted", "set")):
    rows = []
    for n in ns:
        xs = dup.distinct_list(n, seed)
        checksum = sum(i * x for i, x in enumerate(xs))  # 인터프리터가 달라도 같은 입력인지 확인
        for m in methods:
            f = dup.METHODS[m]
            ts, number = time_call(lambda: f(xs), repeats=repeats, target=target)
            row = {"n": n, "method": m, "checksum": checksum, "number": number, "ops": dup.count_ops(m, xs), "times_s": ts}
            row.update(summarize(ts))
            rows.append(row)
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(), "rows": rows}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(**cfg)))
