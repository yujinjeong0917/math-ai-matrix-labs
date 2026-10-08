"""알고리즘 2장 최소 구현: 정렬된 리스트에서 x 이상인 첫 칸(lower bound)을 찾는다.

표준 라이브러리만 쓴다. 오버플로 흉내(mid_overflow_demo)에만 NumPy를 쓴다.

반열린 구간 [lo, hi) 판의 불변식
    I1: 모든 i < lo 에서 a[i] < x       (왼쪽은 확인이 끝난 '작은 쪽')
    I2: 모든 i >= hi 에서 a[i] >= x     (오른쪽은 확인이 끝난 '크거나 같은 쪽')
    척도: hi - lo 가 매 반복 엄격히 줄어든다.
처음(lo=0, hi=n)에는 두 영역이 비어 있어 I1, I2가 저절로 참이고, 끝나면 lo == hi 라서
lo 가 'x 이상인 첫 칸'이다. 이 결론은 bisect.bisect_left 의 문서 설명과 같다.

버그 변형(설계도 a~c)은 모두 같은 질문(lower bound)에 답하려다 한 줄씩 틀린 코드다.
    a_le_with_hi_mid : 루프 조건만 닫힌 구간식(lo <= hi)으로 썼다. lo == hi 에서 mid == lo 라
                       hi = mid 가 구간을 줄이지 못하고, mid == n 이면 배열 밖을 읽는다.
    b_lo_mid         : 오른쪽으로 갈 때 lo = mid + 1 대신 lo = mid. 후보가 한 칸 남은 구간(hi == lo + 1, a[lo] < x)에서 멈추지 않는다.
    c_closed_hi      : 시작을 닫힌 구간식(hi = n - 1)으로, 루프와 갱신은 반열린 식으로 썼다.
                       마지막 칸 뒤(답이 n)인 경우를 영영 고려하지 못한다.
    d_correct        : 올바른 반열린 구간판.
"""

import math

MAX_STEPS = 64  # 길이 5 이하 입력에서 올바른 판은 3번이면 끝난다. 64번을 넘기면 '멈추지 않음'으로 센다.


class Counter:
    """비교 계수기. comparisons 는 a[mid] 와 x 를 비교한 횟수, steps 는 루프를 돈 횟수."""

    def __init__(self):
        self.comparisons = 0
        self.steps = 0


class NonTermination(Exception):
    pass


class OutOfRange(Exception):
    """mid 가 0 <= mid < len(a) 를 벗어났다. 파이썬은 음수 인덱스를 뒤에서부터 읽어 버리므로
    조용히 '동작하는' 것처럼 보이지 않게 여기서 막는다."""


def lower_bound(a, x, counter=None):
    """x 이상인 첫 칸의 위치. 모두 x 보다 작으면 len(a)."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2  # 고정 폭 정수에서도 넘치지 않는 꼴. 파이썬 int 에서는 (lo+hi)//2 와 같다
        if counter is not None:
            counter.comparisons += 1
            counter.steps += 1
        if a[mid] < x:
            lo = mid + 1  # I1 유지: a[mid] < x 이고 a 가 정렬돼 있으니 mid 이하가 모두 x 보다 작다
        else:
            hi = mid  # I2 유지: a[mid] >= x 이니 mid 이상이 모두 x 이상이다
    return lo


def linear_lower_bound(a, x, counter=None):
    """앞에서부터 하나씩 본다. 비교 횟수는 답의 위치 + 1 (끝까지 가면 n)."""
    for i, v in enumerate(a):
        if counter is not None:
            counter.comparisons += 1
        if v >= x:
            return i
    return len(a)


def check_invariant(a, x, lo, hi):
    """반열린 구간 불변식 I1, I2 를 그대로 검사한다. (이름, 성립 여부, 설명) 목록을 돌려준다."""
    i1 = all(a[i] < x for i in range(0, max(0, min(lo, len(a)))))
    i2 = all(a[i] >= x for i in range(max(0, hi), len(a)))
    return [
        ("왼쪽은 모두 x보다 작다", i1, f"a[0:{lo}] < {x}"),
        ("오른쪽은 모두 x 이상이다", i2, f"a[{hi}:{len(a)}] >= {x}"),
    ]


# ---------------------------------------------------------------- 단계 기록(트레이스)


def _read(a, mid):
    if not 0 <= mid < len(a):
        raise OutOfRange(mid)
    return a[mid]


def _variant_step(name, a, x, lo, hi):
    """변형별로 한 반복을 실행한다. (새 lo, 새 hi, mid, 비교 결과) 를 돌려준다."""
    mid = (lo + hi) // 2
    v = _read(a, mid)
    less = v < x
    if name == "b_lo_mid":
        return (mid, hi, mid, less) if less else (lo, mid, mid, less)
    return (mid + 1, hi, mid, less) if less else (lo, mid, mid, less)


VARIANTS = {
    # 이름: (시작 lo, 시작 hi 를 만드는 함수, 루프 조건)
    "a_le_with_hi_mid": (lambda n: n, lambda lo, hi: lo <= hi),
    "b_lo_mid": (lambda n: n, lambda lo, hi: lo < hi),
    "c_closed_hi": (lambda n: n - 1, lambda lo, hi: lo < hi),
    "d_correct": (lambda n: n, lambda lo, hi: lo < hi),
}

VARIANT_CODE = {
    "a_le_with_hi_mid": "lo, hi = 0, len(a)\nwhile lo <= hi:\n    mid = (lo + hi) // 2\n    if a[mid] < x: lo = mid + 1\n    else:          hi = mid",
    "b_lo_mid": "lo, hi = 0, len(a)\nwhile lo < hi:\n    mid = (lo + hi) // 2\n    if a[mid] < x: lo = mid\n    else:          hi = mid",
    "c_closed_hi": "lo, hi = 0, len(a) - 1\nwhile lo < hi:\n    mid = (lo + hi) // 2\n    if a[mid] < x: lo = mid + 1\n    else:          hi = mid",
    "d_correct": "lo, hi = 0, len(a)\nwhile lo < hi:\n    mid = (lo + hi) // 2\n    if a[mid] < x: lo = mid + 1\n    else:          hi = mid",
}


def run_variant(name, a, x, trace=False):
    """변형 하나를 실행한다.

    반환: {"outcome": "ok"|"wrong"|"no_stop"|"out_of_range", "answer", "steps", "trace"}
    trace=True 이면 단계마다 상태와 불변식 판정을 남긴다(공통 스키마 step/op/state/invariants).
    """
    start_hi, cond = VARIANTS[name]
    lo, hi = 0, start_hi(len(a))
    truth = lower_bound(a, x)
    rows = []

    def snap(op, mid=None):
        if trace:
            inv = check_invariant(a, x, lo, hi)
            rows.append({"step": len(rows), "op": op,
                         "state": {"lo": lo, "hi": hi, "mid": mid, "size": hi - lo},
                         "invariants": [{"name": n, "holds": h, "detail": d} for n, h, d in inv]})

    snap("start")
    steps = 0
    try:
        while cond(lo, hi):
            if steps >= MAX_STEPS:
                raise NonTermination()
            before = hi - lo
            lo, hi, mid, less = _variant_step(name, a, x, lo, hi)
            steps += 1
            snap(("a[mid] < x 이므로 오른쪽으로" if less else "a[mid] >= x 이므로 왼쪽으로"), mid)
            if trace:
                rows[-1]["state"]["shrank"] = (hi - lo) < before
    except NonTermination:
        return {"outcome": "no_stop", "answer": None, "steps": steps, "truth": truth, "trace": rows}
    except OutOfRange as e:
        if trace:
            rows.append({"step": len(rows), "op": f"mid = {e.args[0]} 는 배열 밖", "state": {"lo": lo, "hi": hi, "mid": e.args[0], "size": hi - lo}, "invariants": []})
        return {"outcome": "out_of_range", "answer": None, "steps": steps, "truth": truth, "trace": rows}
    outcome = "ok" if lo == truth else "wrong"
    return {"outcome": outcome, "answer": lo, "steps": steps, "truth": truth, "trace": rows}


# ---------------------------------------------------------------- 전수 검사


def sorted_arrays(max_len, values=(0, 2, 4, 6, 8)):
    """길이 0..max_len 의 모든 정렬(같은 값 허용) 배열을 길이 순, 같은 길이 안에서는 사전 순으로."""
    from itertools import combinations_with_replacement

    for n in range(max_len + 1):
        for combo in combinations_with_replacement(values, n):
            yield list(combo)


def targets(values=(0, 2, 4, 6, 8)):
    """배열 값 그 자체, 값 사이, 양쪽 바깥을 모두 포함한다: -1, 0, 1, ..., 9."""
    return list(range(min(values) - 1, max(values) + 2))


def exhaustive(max_len=5):
    """변형별 결과를 센다. 처음 만난 실패가 곧 가장 짧은 반례가 되도록 짧은 배열부터 돈다."""
    out = {}
    for name in VARIANTS:
        counts = {"ok": 0, "wrong": 0, "no_stop": 0, "out_of_range": 0}
        first = None
        by_len = {}
        for a in sorted_arrays(max_len):
            for x in targets():
                r = run_variant(name, a, x)
                counts[r["outcome"]] += 1
                bl = by_len.setdefault(str(len(a)), {"cases": 0, "fail": 0})
                bl["cases"] += 1
                if r["outcome"] != "ok":
                    bl["fail"] += 1
                    if first is None:
                        first = {"a": a, "x": x, "outcome": r["outcome"], "answer": r["answer"], "truth": r["truth"]}
        total = sum(counts.values())
        out[name] = {"counts": counts, "total": total, "fail": total - counts["ok"], "first_failure": first, "by_len": by_len}
    return out


def max_steps_bound(n):
    """올바른 판의 반복 횟수 상한 ceil(log2(n+1))."""
    return math.ceil(math.log2(n + 1)) if n > 0 else 0


# ---------------------------------------------------------------- 오버플로 흉내


def mid_overflow_demo(low=1_500_000_000, high=2_000_000_000):
    """32비트 부호 있는 정수(Java int 와 같은 폭)로 인덱스 계산만 흉내 낸다. 큰 배열은 만들지 않는다.

    NumPy 2.3.3 에서 스칼라 덧셈은 RuntimeWarning 을 내고, 배열 덧셈은 경고 없이 값이 감긴다.
    """
    import warnings

    import numpy as np

    lo, hi = np.int32(low), np.int32(high)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = lo + hi
        naive = s // np.int32(2)
    scalar_warn = [str(m.message) for m in w]
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        arr_sum = (np.array([low], dtype=np.int32) + np.array([high], dtype=np.int32))[0]
    array_warn = [str(m.message) for m in w2]
    safe = lo + (hi - lo) // np.int32(2)
    return {
        "low": low, "high": high, "int32_max": int(np.iinfo(np.int32).max),
        "true_sum": low + high, "wrapped_sum": int(s), "wrapped_sum_by_rule": low + high - 2 ** 32,
        "naive_mid": int(naive), "safe_mid": int(safe), "python_int_mid": (low + high) // 2,
        "scalar_add_warnings": scalar_warn, "array_add_warnings": array_warn, "array_wrapped_sum": int(arr_sum),
    }


# ---------------------------------------------------------------- AI 연결: 누적확률에서 표본 뽑기


def sample_index_linear(cdf, u):
    """LLM 트랙 11장(ch11_decoding/decoding_np.py, sample_from_logits)의 한 줄을 복사했다.
    누적확률 cdf 에서 u 이하인 칸 수를 센다. 칸을 모두 본다(V 번 비교)."""
    import numpy as np

    return (np.asarray(cdf) <= u).sum(-1)


def sample_index_bisect(cdf, u):
    """같은 답을 이분 탐색으로 얻는다. 'u 이하인 칸 수' = u 보다 큰 첫 칸의 위치 = bisect_right."""
    import numpy as np

    return np.searchsorted(cdf, u, side="right")
