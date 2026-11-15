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
    # 호출 횟수를 2배씩 늘려 가며 묶음 시간이 target 에 닿는지 본다.
    # 시간 O(target 의 몇 배): number 가 2배씩 커지므로 측정 횟수는 로그 단위다.
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = time.perf_counter() - t0
        if dt >= target:
            return number
        if dt >= target / 10:
            # 측정값이 충분히 커졌으면 비례식으로 한 번에 보정한다(+1 은 0이 되는 것 방지).
            return int(number * target / dt) + 1
        number *= 2


def time_call(fn, repeats=10, target=0.02):
    """1회 호출 시간(초) repeats 개를 돌려준다."""
    number = calibrate(fn, target)
    # 묶음 시간을 number 로 나눠 1회 시간(초)으로 환산한다. 이 값이 이 장의 "시간" 축이다.
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        out.append((time.perf_counter() - t0) / number)
    return out, number


def summarize(ts):
    # 평균 대신 중앙값과 사분위수를 쓴다: 한 번의 튀는 값(다른 프로세스 간섭 등)에 덜 흔들린다.
    # statistics.quantiles(n=4)는 표본이 2개 미만이면 오류이므로 repeats >= 2 가 필요하다.
    s = sorted(ts)
    q = statistics.quantiles(s, n=4)
    return {"min": s[0], "q1": q[0], "median": statistics.median(s), "q3": q[2], "max": s[-1],
            "max_over_min": s[-1] / s[0]}


def run(ns, repeats=10, seed=0, target=0.02, methods=("pairs", "sorted", "set")):
    rows = []
    for n in ns:
        xs = dup.distinct_list(n, seed)  # 중복 없는 입력 = 모든 방법이 끝까지 도는 최악 경우
        checksum = sum(i * x for i, x in enumerate(xs))  # 인터프리터가 달라도 같은 입력인지 확인
        for m in methods:
            f = dup.METHODS[m]
            # ops 는 시간 측정과 별도 호출로 센 기본 연산 수(계수기 경로)이고, 시간은 계수기 없는 경로로 잰다.
            ts, number = time_call(lambda: f(xs), repeats=repeats, target=target)
            row = {"n": n, "method": m, "checksum": checksum, "number": number, "ops": dup.count_ops(m, xs), "times_s": ts}
            row.update(summarize(ts))
            rows.append(row)
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(), "rows": rows}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(run(**cfg)))
