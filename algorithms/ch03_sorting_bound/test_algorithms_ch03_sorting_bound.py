"""알고리즘 3장 검증. `uv run pytest -q algorithms/ch03_sorting_bound` 로 실행한다.

설계도는 Hypothesis 성질 테스트를 적었지만 의존성을 늘리지 않으려고, 작은 n 의 모든 순열 전수 검사와
시드를 고정한 무작위 입력 반복문으로 같은 성질을 확인한다.
"""

import collections
import itertools
import math
import random
from fractions import Fraction

import numpy as np
import torch

import algorithms_ch03_sorts as s

torch.set_num_threads(2)


def test_first_screen_numbers():
    # 카드 3장: 6가지 순서, 질문 2번이면 답 패턴 4가지뿐이라 3번이 필요
    assert math.factorial(3) == 6 and 2 ** 2 < 6 <= 2 ** 3 and s.ceil_log2_factorial(3) == 3
    # 카드 4장: 24가지, 2^4 = 16 < 24 <= 32 = 2^5
    assert math.factorial(4) == 24 and 2 ** 4 < 24 <= 2 ** 5 and s.ceil_log2_factorial(4) == 5
    # 카드 8장: 40,320가지, 2^15 = 32,768 < 40,320 <= 65,536 = 2^16
    assert math.factorial(8) == 40320 and 2 ** 15 < 40320 <= 2 ** 16 and s.ceil_log2_factorial(8) == 16
    # 정렬된 8장에 첫 원소 피벗: 7 + 6 + ... + 1 = 28
    _, c = s.run_sort("lomuto_first", list(range(1, 9)))
    assert c.comparisons == 28 == sum(range(8))
    _, c = s.run_sort("lomuto_first", [5, 2, 7, 1, 8, 3, 6, 4])
    assert c.comparisons == 15  # 운 좋은 한 입력은 16보다 적을 수 있다. 하한은 최악과 평균에 대한 말이다


def test_ceil_log2_factorial_is_exact():
    for n in range(0, 300):
        k = s.ceil_log2_factorial(n)
        f = math.factorial(n)
        assert 2 ** k >= f and (k == 0 or 2 ** (k - 1) < f)
        assert abs(s.log2_factorial(n) - math.log2(f) if f > 0 else 0) < 1e-6
    for n in (100, 1000, 2000):
        assert abs(s.stirling_log2_factorial(n) - s.log2_factorial(n)) < 0.01


def test_all_sorts_match_sorted_numpy_torch():
    rng = random.Random(0)
    for t in range(400):
        n = rng.randint(0, 80)
        xs = [rng.randint(0, rng.choice([2, 9, 1000])) for _ in range(n)]
        ref = sorted(xs)
        for name in s.SORTS:
            a, _ = s.run_sort(name, xs, t)
            assert a == ref, (name, xs)
            assert collections.Counter(a) == collections.Counter(xs)
        if n:
            assert np.sort(np.array(xs)).tolist() == ref
            assert torch.sort(torch.tensor(xs)).values.tolist() == ref


def test_exhaustive_lower_bound_small_n():
    for n in range(1, 8):
        bound = s.ceil_log2_factorial(n)
        lf = s.log2_factorial(n)
        for name in ["insertion", "merge", "heap", "lomuto_first", "median3", "introsort"]:
            worst, total = 0, 0
            for p in itertools.permutations(range(n)):
                a, c = s.run_sort(name, list(p))
                assert a == list(range(n))
                worst = max(worst, c.comparisons)
                total += c.comparisons
            assert worst >= bound, (name, n)
            assert total / math.factorial(n) >= lf - 1e-12, (name, n)
    # 합병 정렬의 최악: n = 4 에서 하한 5와 같고, n = 5 에서는 8로 하한 7보다 하나 많다
    worst = {n: max(s.run_sort("merge", list(p))[1].comparisons for p in itertools.permutations(range(n))) for n in (4, 5)}
    assert worst == {4: 5, 5: 8}


def test_first_pivot_sorted_is_quadratic():
    for n in (1, 2, 10, 100, 777):
        _, c = s.run_sort("lomuto_first", list(range(n)))
        assert c.comparisons == n * (n - 1) // 2
        assert c.max_depth == max(0, n - 2)  # 크기 2 구간까지 분할하고 끝(깊이 0부터 셈)
        _, c = s.run_sort("lomuto_first", list(range(n, 0, -1)))
        assert c.comparisons == n * (n - 1) // 2
        _, c = s.run_sort("lomuto_first", [7] * n)
        assert c.comparisons == n * (n - 1) // 2


def test_lomuto_average_equals_closed_form():
    # 서로 다른 키의 모든 순열에서 첫 원소 피벗의 평균 = 피벗을 고르게 뽑을 때의 기대값 = 2(n+1)H_n - 4n
    for n in range(1, 8):
        tot = sum(s.run_sort("lomuto_first", list(p))[1].comparisons for p in itertools.permutations(range(n)))
        avg = Fraction(tot, math.factorial(n))
        exact = 2 * (n + 1) * sum(Fraction(1, k) for k in range(1, n + 1)) - 4 * n
        assert avg == exact
    # 점화식 C_n = (n-1) + (2/n) sum C_k 도 같은 닫힌 꼴
    C = [Fraction(0)]
    for n in range(1, 40):
        C.append((n - 1) + Fraction(2, n) * sum(C[:n]))
        assert abs(float(C[n]) - s.lomuto_expected(n)) < 1e-9


def test_recursion_limit_hits_recursive_version():
    import sys
    assert sys.getrecursionlimit() >= 500
    s.lomuto_first_recursive(list(range(300)))
    try:
        s.lomuto_first_recursive(list(range(sys.getrecursionlimit() + 100)))
        raise AssertionError("재귀 한도에 걸려야 한다")
    except RecursionError:
        pass
    a = list(range(5000))
    s.lomuto_first(a)  # 명시적 스택판은 끝난다


def test_hoare_random_pending_and_equal_keys():
    for kind in ("random", "sorted", "reversed", "few_unique"):
        for n in (100, 1000):
            xs = s.make_input(kind, n, 3)
            a, c = s.run_sort("hoare_random", xs, 3)
            assert a == sorted(xs)
            assert c.max_pending <= math.floor(math.log2(n)) + 1  # Hoare(1962): 큰 쪽을 미루면 log2 N 이하
    eq = [5] * 1024
    _, ch = s.run_sort("hoare_random", eq, 0)
    _, cl = s.run_sort("lomuto_random", eq, 0)
    assert cl.comparisons == 1024 * 1023 // 2  # Lomuto 는 같은 키에서 피벗과 상관없이 n(n-1)/2
    assert ch.comparisons < 2 * 1024 * 11  # 두 포인터 분할은 반씩 나뉘어 n log n 수준
    assert ch.max_depth <= 11


def test_median3_killer_theorem_and_introsort():
    assert s.median3_killer(8) == [1, 5, 3, 7, 2, 4, 6, 8]
    for n in (64, 1000, 2000):
        xs = s.median3_killer(n)
        assert sorted(xs) == list(range(1, n + 1))
        _, c = s.run_sort("median3", xs)
        k = n // 4
        assert c.partition_sizes[:k] == [(2, n - 2 * j) for j in range(1, k + 1)]  # Musser Theorem 1
        _, ci = s.run_sort("introsort", xs)
        limit = 2 * int(math.floor(math.log2(n)))
        assert ci.partitions == limit and ci.heapsort_sizes == [n - 2 * limit]
    c1 = s.run_sort("median3", s.median3_killer(1000))[1].comparisons
    c2 = s.run_sort("median3", s.median3_killer(2000))[1].comparisons
    assert 3.5 < c2 / c1 < 4.5  # n 이 두 배면 약 네 배: 제곱으로 는다
    # 무작위 입력에서 introsort 는 median3 퀵정렬과 비교 수까지 같다(힙정렬로 넘기지 않음)
    for seed in range(5):
        xs = s.make_input("random", 2000, seed)
        cm = s.run_sort("median3", xs)[1]
        ci = s.run_sort("introsort", xs)[1]
        assert cm.comparisons == ci.comparisons and ci.heapsort_sizes == []


def test_widget_traces_keep_invariants():
    for xs in ([5, 2, 7, 1, 8, 3, 6, 4], list(range(1, 9)), [4] * 8):
        t = s.trace_lomuto_first(xs)
        assert t["steps"][-1][1] == sorted(xs)
        assert all(all(step[8]) for step in t["steps"])
        assert t["comparisons"] == s.run_sort("lomuto_first", xs)[1].comparisons
        assert t["bound"] == 16


def test_topk_partition_matches_full_sort():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000).astype(np.float32)
    k = 7
    full = np.argsort(-x, kind="stable")[:k]
    part = np.argpartition(-x, k - 1)[:k]
    part = part[np.argsort(-x[part], kind="stable")]
    assert np.array_equal(full, part)
    assert np.array_equal(full, torch.topk(torch.from_numpy(x), k).indices.numpy())
