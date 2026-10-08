"""알고리즘 3장 최소 구현: 비교 정렬 여러 개와 비교 횟수 계수기, 그리고 비교 정렬의 하한.

표준 라이브러리만 쓴다. 모든 정렬 함수는 리스트를 제자리에서 정렬하고, 같은 Counter 를 받아
"키끼리 비교한 횟수"를 센다. 셈 규칙은 하나다: 두 키에 대해 `<` 를 한 번 평가할 때마다 1.

들어 있는 것
    lomuto_first        : 첫 원소를 피벗으로 하는 Lomuto 분할 퀵정렬(명시적 스택판). 크기 m 구간마다
                          정확히 m-1 번 비교한다. 정렬된 입력에서 n(n-1)/2 번이 된다(설계도의 실패 예).
    lomuto_first_recursive : 같은 알고리즘의 재귀판. 정렬된 입력에서 재귀 깊이가 n-1 이라
                          파이썬 재귀 한도에 먼저 걸린다.
    lomuto_random       : Lomuto 분할 + 무작위 피벗. 같은 키가 많으면 피벗과 상관없이 느려진다(분할 방식 탓).
    hoare_random        : Hoare(1962)식 두 포인터 분할 + 무작위 피벗. 같은 키에서 양쪽 포인터가 멈춰
                          구간을 고르게 나눈다. 대기 구간은 "큰 쪽을 미뤄 두고 작은 쪽부터" 규칙으로 쌓는다.
    median3_quicksort   : Musser(1997)가 적은 HP STL 판 퀵정렬. 첫·가운데·끝의 중앙값을 피벗으로,
                          unguarded 분할, 작은 구간은 마지막에 삽입 정렬.
    introsort           : Musser(1997)의 introsort. 위와 같은데 분할 깊이가 2*floor(log2 n) 에 닿으면
                          그 구간을 힙정렬로 넘긴다.
    introsort_first     : 같은 깊이 제한을 Lomuto 첫 원소 피벗에 붙인 판(Musser 가 끝에서 가능하다고 적은 조합).
    merge_sort, heapsort, insertion_sort : 비교 대상.
    median3_killer(n)   : Musser(1997)가 C. Stewart 에게서 받았다고 적은 순열 K_n.
    ceil_log2_factorial : ceil(log2 n!), 정수로 계산.
"""

import math
import random


class Counter:
    """comparisons: 키 비교 횟수. max_depth: 분할 트리의 최대 깊이(루트 구간 = 0).
    max_pending: 대기 중인 구간(스택) 길이의 최대값. partitions: 분할을 한 횟수.
    heapsort_calls: introsort 가 힙정렬로 넘긴 횟수와 그 구간 길이."""

    def __init__(self):
        self.comparisons = 0
        self.max_depth = 0
        self.max_pending = 0
        self.partitions = 0
        self.heapsort_sizes = []
        self.partition_sizes = []  # (왼쪽 크기, 오른쪽 크기). median3 계열만 기록


def _c(counter):
    return counter if counter is not None else Counter()


# ---------------------------------------------------------------- 하한

def ceil_log2_factorial(n):
    """ceil(log2 n!) 를 정수로. 2^k >= n! 인 가장 작은 k = (n! - 1).bit_length() (n! >= 1)."""
    f = math.factorial(n)
    return (f - 1).bit_length()


def log2_factorial(n):
    """log2 n! (실수). lgamma 로 계산한다. 올림이 필요하면 ceil_log2_factorial 을 쓴다."""
    return math.lgamma(n + 1) / math.log(2)


def stirling_log2_factorial(n):
    """n log2 n - n log2 e + (1/2) log2(2 pi n). 스털링 근사."""
    if n == 0:
        return 0.0
    return n * math.log2(n) - n * math.log2(math.e) + 0.5 * math.log2(2 * math.pi * n)


def harmonic(n):
    return sum(1.0 / k for k in range(1, n + 1))


def lomuto_expected(n):
    """서로 다른 키, 피벗이 구간에서 고르게 뽑힐 때 Lomuto(구간 m 마다 m-1 번) 퀵정렬의 기대 비교 수.
    C_n = (n-1) + (2/n) sum_{k<n} C_k 의 닫힌 꼴 2(n+1)H_n - 4n."""
    return 2 * (n + 1) * harmonic(n) - 4 * n


# ---------------------------------------------------------------- Lomuto 분할

def lomuto_partition(a, lo, hi, counter):
    """a[lo..hi] (양끝 포함), 피벗은 a[lo]. 불변식(j 를 돌 때):
        a[lo+1 .. i]   는 모두 피벗보다 작다
        a[i+1 .. j-1]  는 모두 피벗 이상이다
    끝나면 피벗을 i 자리로 옮기고 i 를 돌려준다. 비교는 정확히 hi - lo 번."""
    p = a[lo]
    i = lo
    for j in range(lo + 1, hi + 1):
        if a[j] < p:
            i += 1
            a[i], a[j] = a[j], a[i]
    counter.comparisons += hi - lo
    a[lo], a[i] = a[i], a[lo]
    return i


def lomuto_first(a, counter=None):
    """첫 원소 피벗 퀵정렬, 명시적 스택. 스택에 (lo, hi, 깊이)를 쌓는다."""
    c = _c(counter)
    stack = [(0, len(a) - 1, 0)]
    while stack:
        c.max_pending = max(c.max_pending, len(stack))
        lo, hi, d = stack.pop()
        if lo >= hi:
            continue
        c.max_depth = max(c.max_depth, d)
        c.partitions += 1
        m = lomuto_partition(a, lo, hi, c)
        stack.append((m + 1, hi, d + 1))
        stack.append((lo, m - 1, d + 1))
    return a


def lomuto_first_recursive(a, counter=None):
    """같은 알고리즘의 재귀판. 정렬된 입력에서 재귀가 n-1 단계로 깊어진다."""
    c = _c(counter)

    def rec(lo, hi, d):
        if lo >= hi:
            return
        c.max_depth = max(c.max_depth, d)
        c.partitions += 1
        m = lomuto_partition(a, lo, hi, c)
        rec(lo, m - 1, d + 1)
        rec(m + 1, hi, d + 1)

    rec(0, len(a) - 1, 0)
    return a


def lomuto_random(a, rng, counter=None):
    """Lomuto 분할 + 무작위 피벗(뽑은 원소를 맨 앞으로 옮긴 뒤 분할)."""
    c = _c(counter)
    stack = [(0, len(a) - 1, 0)]
    while stack:
        c.max_pending = max(c.max_pending, len(stack))
        lo, hi, d = stack.pop()
        if lo >= hi:
            continue
        c.max_depth = max(c.max_depth, d)
        c.partitions += 1
        r = rng.randint(lo, hi)
        a[lo], a[r] = a[r], a[lo]
        m = lomuto_partition(a, lo, hi, c)
        stack.append((m + 1, hi, d + 1))
        stack.append((lo, m - 1, d + 1))
    return a


# ---------------------------------------------------------------- Hoare식 두 포인터 분할

def hoare_partition(a, lo, hi, counter):
    """피벗 a[lo]. 아래 포인터는 피벗보다 작은 동안 오르고, 위 포인터는 피벗보다 큰 동안 내려온다.
    둘 다 피벗과 같은 키에서 멈추므로, 키가 모두 같아도 구간이 반씩 나뉜다. 두 포인터의 큰 틀은 Hoare(1962)지만
    Hoare 원문의 포인터는 피벗 '이하'/'이상'인 동안 계속 가서 같은 키를 지나치고, 그래서 Hoare는 '모든 키가 같을 때'를
    곤란한 경우(awkward situation)로 따로 짚었다. 같은 키에서 멈추는 규칙은 Musser(1997) 각주 3의 권고 쪽이다.
    끝에 피벗을 j 로 옮기고 j 를 돌려준다."""
    p = a[lo]
    i, j = lo, hi + 1
    cmp = 0
    while True:
        i += 1
        while i <= hi:
            cmp += 1
            if a[i] < p:
                i += 1
            else:
                break
        j -= 1
        while True:
            cmp += 1
            if p < a[j]:
                j -= 1
            else:
                break
        if i >= j:
            break
        a[i], a[j] = a[j], a[i]
    counter.comparisons += cmp
    a[lo], a[j] = a[j], a[lo]
    return j


def hoare_random(a, rng, counter=None):
    """무작위 피벗 + 두 포인터 분할. 대기 구간은 큰 쪽을 스택에 미루고 작은 쪽을 먼저 처리한다.
    Hoare(1962): 미뤄 둔 구간 수가 log2 N 을 넘지 않게 하려면 '항상 큰 쪽을 미루면' 충분하다."""
    c = _c(counter)
    stack = [(0, len(a) - 1, 0)]
    while stack:
        lo, hi, d = stack.pop()
        while lo < hi:
            c.max_depth = max(c.max_depth, d)
            c.partitions += 1
            r = rng.randint(lo, hi)
            a[lo], a[r] = a[r], a[lo]
            m = hoare_partition(a, lo, hi, c)
            d += 1
            if m - lo < hi - m:  # 왼쪽이 작다: 오른쪽(큰 쪽)을 미루고 왼쪽을 계속
                stack.append((m + 1, hi, d))
                hi = m - 1
            else:
                stack.append((lo, m - 1, d))
                lo = m + 1
            c.max_pending = max(c.max_pending, len(stack))
    return a


# ---------------------------------------------------------------- 비교 대상

def insertion_sort(a, counter=None, lo=0, hi=None):
    c = _c(counter)
    if hi is None:
        hi = len(a)
    cmp = 0
    for k in range(lo + 1, hi):
        x = a[k]
        j = k - 1
        while j >= lo:
            cmp += 1
            if x < a[j]:
                a[j + 1] = a[j]
                j -= 1
            else:
                break
        a[j + 1] = x
    c.comparisons += cmp
    return a


def merge_sort(a, counter=None):
    """위에서 아래로 반씩 나누는 합병 정렬. 합칠 때 두 줄의 맨 앞을 비교한다(같으면 왼쪽 먼저, 안정)."""
    c = _c(counter)
    buf = a[:]

    def rec(lo, hi):  # [lo, hi)
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        rec(lo, mid)
        rec(mid, hi)
        i, j, k = lo, mid, lo
        cmp = 0
        while i < mid and j < hi:
            cmp += 1
            if a[j] < a[i]:
                buf[k] = a[j]
                j += 1
            else:
                buf[k] = a[i]
                i += 1
            k += 1
        while i < mid:
            buf[k] = a[i]
            i += 1
            k += 1
        while j < hi:
            buf[k] = a[j]
            j += 1
            k += 1
        a[lo:hi] = buf[lo:hi]
        c.comparisons += cmp

    rec(0, len(a))
    return a


def heapsort(a, counter=None, lo=0, hi=None):
    """a[lo:hi] 를 힙정렬(최대 힙)."""
    c = _c(counter)
    if hi is None:
        hi = len(a)
    n = hi - lo
    cmp = 0

    def sift(root, end):  # 힙 인덱스 기준 [0, end)
        nonlocal cmp
        while True:
            child = 2 * root + 1
            if child >= end:
                return
            if child + 1 < end:
                cmp += 1
                if a[lo + child] < a[lo + child + 1]:
                    child += 1
            cmp += 1
            if a[lo + root] < a[lo + child]:
                a[lo + root], a[lo + child] = a[lo + child], a[lo + root]
                root = child
            else:
                return

    for start in range(n // 2 - 1, -1, -1):
        sift(start, n)
    for end in range(n - 1, 0, -1):
        a[lo], a[lo + end] = a[lo + end], a[lo]
        sift(0, end)
    c.comparisons += cmp
    return a


# ---------------------------------------------------------------- Musser(1997)의 STL 판

SIZE_THRESHOLD = 16  # Musser 논문의 __stl_threshold 자리. 논문은 값을 적지 않아 16으로 둔다. 이 이하의 구간은 마지막 삽입 정렬로 넘긴다


def _median3(x, y, z, c):
    """세 값의 중앙값. 비교 2~3번."""
    c.comparisons += 1
    if x < y:
        c.comparisons += 1
        if y < z:
            return y
        c.comparisons += 1
        return z if x < z else x
    c.comparisons += 1
    if x < z:
        return x
    c.comparisons += 1
    return z if y < z else y


def unguarded_partition(a, first, last, pivot, c):
    """Musser(1997)에 실린 STL __unguarded_partition 을 그대로 옮겼다. [first, last) 에서 피벗 값보다
    작은 것은 왼쪽, 큰 것은 오른쪽으로 모으고 경계(cut)를 돌려준다. 피벗이 구간 안의 값이라 경계 검사가 없다."""
    cmp = 0
    while True:
        while True:
            cmp += 1
            if a[first] < pivot:
                first += 1
            else:
                break
        last -= 1
        while True:
            cmp += 1
            if pivot < a[last]:
                last -= 1
            else:
                break
        if not first < last:
            c.comparisons += cmp
            return first
        a[first], a[last] = a[last], a[first]
        first += 1


def _pivot_m3(a, f, b, c):
    return _median3(a[f], a[f + (b - f) // 2], a[b - 1], c)


def median3_quicksort(a, counter=None):
    """Musser(1997)의 Quicksort / Quicksort_loop 의사코드. 작은 쪽으로 재귀하고 큰 쪽은 반복으로 돈다."""
    c = _c(counter)

    def loop(f, b, d):
        while b - f > SIZE_THRESHOLD:
            c.max_depth = max(c.max_depth, d)
            c.partitions += 1
            p = unguarded_partition(a, f, b, _pivot_m3(a, f, b, c), c)
            c.partition_sizes.append((p - f, b - p))
            d += 1
            if p - f >= b - p:
                loop(p, b, d)
                b = p
            else:
                loop(f, p, d)
                f = p

    loop(0, len(a), 0)
    insertion_sort(a, c)
    return a


def introsort(a, counter=None):
    """Musser(1997)의 Introsort / Introsort_loop. 깊이 제한 2*floor(log2 n), 닿으면 그 구간을 힙정렬."""
    c = _c(counter)
    n = len(a)

    def loop(f, b, depth_limit, d):
        while b - f > SIZE_THRESHOLD:
            if depth_limit == 0:
                c.heapsort_sizes.append(b - f)
                heapsort(a, c, f, b)
                return
            depth_limit -= 1
            c.max_depth = max(c.max_depth, d)
            c.partitions += 1
            p = unguarded_partition(a, f, b, _pivot_m3(a, f, b, c), c)
            c.partition_sizes.append((p - f, b - p))
            d += 1
            loop(p, b, depth_limit, d)
            b = p

    if n > 1:
        loop(0, n, 2 * int(math.floor(math.log2(n))), 0)
    insertion_sort(a, c)
    return a


def introsort_first(a, counter=None):
    """lomuto_first 와 같은 분할(첫 원소 피벗)에 깊이 제한만 붙였다. 제한은 2*floor(log2 n)."""
    c = _c(counter)
    n = len(a)
    if n < 2:
        return a
    limit = 2 * int(math.floor(math.log2(n)))
    stack = [(0, n - 1, 0)]
    while stack:
        c.max_pending = max(c.max_pending, len(stack))
        lo, hi, d = stack.pop()
        if lo >= hi:
            continue
        if d >= limit:
            c.heapsort_sizes.append(hi - lo + 1)
            heapsort(a, c, lo, hi + 1)
            continue
        c.max_depth = max(c.max_depth, d)
        c.partitions += 1
        m = lomuto_partition(a, lo, hi, c)
        stack.append((m + 1, hi, d + 1))
        stack.append((lo, m - 1, d + 1))
    return a


def median3_killer(n):
    """Musser(1997)의 K_n (n = 2k, k 짝수). 값 1..n 의 순열.
    앞 절반: 1, k+1, 3, k+3, 5, ..., k-1, 2k-1  (홀수 자리 i 는 i, 짝수 자리 i 는 k+i-1)
    뒤 절반: 2, 4, 6, ..., 2k                    """
    assert n % 4 == 0, "Theorem 1 은 n 이 4의 배수일 때"
    k = n // 2
    first = []
    for i in range(1, k + 1):
        first.append(i if i % 2 == 1 else k + i - 1)
    second = [2 * i for i in range(1, k + 1)]
    return first + second


# ---------------------------------------------------------------- 입력과 이름표

def make_input(kind, n, seed):
    rng = random.Random(seed)
    if kind == "random":
        xs = list(range(n))
        rng.shuffle(xs)
        return xs
    if kind == "sorted":
        return list(range(n))
    if kind == "reversed":
        return list(range(n, 0, -1))
    if kind == "few_unique":  # 같은 값 다수: 0..9 열 가지 값
        return [rng.randrange(10) for _ in range(n)]
    if kind == "killer":
        return median3_killer(n)
    raise ValueError(kind)


SORTS = {
    "lomuto_first": lambda a, rng, c: lomuto_first(a, c),
    "lomuto_random": lambda a, rng, c: lomuto_random(a, rng, c),
    "hoare_random": lambda a, rng, c: hoare_random(a, rng, c),
    "median3": lambda a, rng, c: median3_quicksort(a, c),
    "introsort": lambda a, rng, c: introsort(a, c),
    "introsort_first": lambda a, rng, c: introsort_first(a, c),
    "merge": lambda a, rng, c: merge_sort(a, c),
    "heap": lambda a, rng, c: heapsort(a, c),
    "insertion": lambda a, rng, c: insertion_sort(a, c),
}
RANDOMIZED = {"lomuto_random", "hoare_random"}


def run_sort(name, xs, seed=0):
    a = list(xs)
    c = Counter()
    SORTS[name](a, random.Random(1000 + seed), c)
    return a, c


# ---------------------------------------------------------------- 위젯용 트레이스

def trace_lomuto_first(xs):
    """lomuto_first 를 비교 한 번마다 한 단계로 기록한다. 단계: [설명, 배열, lo, hi, i, j, 비교 누적, 깊이, [불변식1, 불변식2]]
    불변식1: a[lo+1..i] 가 모두 피벗보다 작다. 불변식2: a[i+1..j-1] 가 모두 피벗 이상이다."""
    a = list(xs)
    steps = [["start", a[:], None, None, None, None, 0, 0, [True, True]]]
    comps = 0
    stack = [(0, len(a) - 1, 0)]

    def inv(lo, i, j):
        p = a[lo]
        return [all(a[t] < p for t in range(lo + 1, i + 1)), all(a[t] >= p for t in range(i + 1, j))]

    while stack:
        lo, hi, d = stack.pop()
        if lo >= hi:
            continue
        p = a[lo]
        i = lo
        for j in range(lo + 1, hi + 1):
            comps += 1
            smaller = a[j] < p
            if smaller:
                i += 1
                a[i], a[j] = a[j], a[i]
            steps.append([f"a[{j}]={a[i] if smaller else a[j]} {'<' if smaller else '>='} 피벗 {p}", a[:], lo, hi, i, j + 1, comps, d, inv(lo, i, j + 1)])
        a[lo], a[i] = a[i], a[lo]
        steps.append([f"피벗을 {i}번 칸으로 옮김(값 {p})", a[:], lo, hi, i, hi + 1, comps, d, [True, True]])
        stack.append((i + 1, hi, d + 1))
        stack.append((lo, i - 1, d + 1))
    return {"input": list(xs), "comparisons": comps, "bound": ceil_log2_factorial(len(xs)), "steps": steps}
