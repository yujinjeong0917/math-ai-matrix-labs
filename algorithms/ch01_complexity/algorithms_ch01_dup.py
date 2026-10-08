"""알고리즘 1장 최소 구현: 리스트에 같은 값이 두 번 이상 있는지 세 가지 방법으로 찾는다.

표준 라이브러리만 쓴다(다른 Python 버전에서도 같은 코드로 시간을 재기 위해서다. Python 3.9 이상).

- has_dup_pairs  : 모든 쌍을 비교한다. 중복이 없으면 비교 n(n-1)/2번.
- has_dup_sorted : 정렬한 뒤 이웃끼리만 비교한다. 정렬 비교 + 이웃 비교 n-1번.
- has_dup_set    : 집합에 넣으며 이미 있는지 확인한다. 확인 n번.

counter 를 넘기면 기본 연산 수를 센다. 시간을 잴 때는 counter=None 으로 부른다.
세는 코드가 시간을 부풀리지 않게 두 경로를 따로 둔다.
"""


class Counter:
    """기본 연산 계수기. comparisons 는 원소 두 개를 비교한 횟수, lookups 는 집합 확인 횟수."""

    def __init__(self):
        self.comparisons = 0
        self.lookups = 0

    @property
    def total(self):
        return self.comparisons + self.lookups


class _Counted:
    """sorted() 가 원소를 비교할 때마다 숫자를 센다. CPython 의 sorted 는 < 만 쓴다."""

    __slots__ = ("v", "c")

    def __init__(self, v, c):
        self.v = v
        self.c = c

    def __lt__(self, other):
        self.c.comparisons += 1
        return self.v < other.v


def has_dup_pairs(xs, counter=None):
    n = len(xs)
    if counter is None:
        for i in range(n):
            xi = xs[i]
            for j in range(i + 1, n):
                if xi == xs[j]:
                    return True
        return False
    for i in range(n):
        xi = xs[i]
        for j in range(i + 1, n):
            counter.comparisons += 1
            if xi == xs[j]:
                return True
    return False


def has_dup_sorted(xs, counter=None):
    if counter is None:
        s = sorted(xs)
        for i in range(1, len(s)):
            if s[i - 1] == s[i]:
                return True
        return False
    s = [w.v for w in sorted(_Counted(v, counter) for v in xs)]
    for i in range(1, len(s)):
        counter.comparisons += 1
        if s[i - 1] == s[i]:
            return True
    return False


def _merge_sort(a, counter):
    """파이썬으로 짠 합병 정렬. 내장 sorted(C 구현)와 달리 모든 줄이 파이썬이라
    모든 쌍 비교와 '한 줄의 값'이 비슷하다. counter 가 있으면 원소 비교를 센다."""
    n = len(a)
    if n <= 1:
        return a
    mid = n // 2
    left, right = _merge_sort(a[:mid], counter), _merge_sort(a[mid:], counter)
    out = []
    i = j = 0
    nl, nr = len(left), len(right)
    while i < nl and j < nr:
        if counter is not None:
            counter.comparisons += 1
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def has_dup_msort(xs, counter=None):
    s = _merge_sort(list(xs), counter)
    for i in range(1, len(s)):
        if counter is not None:
            counter.comparisons += 1
        if s[i - 1] == s[i]:
            return True
    return False


def has_dup_set(xs, counter=None):
    seen = set()
    if counter is None:
        for x in xs:
            if x in seen:
                return True
            seen.add(x)
        return False
    for x in xs:
        counter.lookups += 1
        if x in seen:
            return True
        seen.add(x)
    return False


METHODS = {"pairs": has_dup_pairs, "sorted": has_dup_sorted, "msort": has_dup_msort, "set": has_dup_set}


def count_ops(method, xs):
    c = Counter()
    METHODS[method](xs, c)
    return c.total


def distinct_list(n, seed):
    """중복이 없는 정수 n개(모든 쌍 비교의 최악 입력). 0..10n-1 에서 겹치지 않게 뽑는다."""
    import random

    return random.Random(seed).sample(range(10 * n), n)
