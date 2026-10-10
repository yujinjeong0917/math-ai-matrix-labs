"""알고리즘 9장 검증. `uv run pytest -q algorithms/ch09_divide_conquer` 로 실행한다.

설계도는 Hypothesis 의 integers 전략을 적었지만 의존성을 늘리지 않으려고, 시드를 고정한 무작위 정수
(음수, 0, 자릿수가 크게 다른 쌍 포함)로 같은 성질을 확인한다. 정답은 Python 의 정수 곱 x * y 다.
"""

import math
import random

import numpy as np

import algorithms_ch09_mul as mm


def _cases(seed=0, count=300, max_digits=80):
    rng = random.Random(seed)
    out = [(0, 0), (0, 123), (-7, 0), (1, -1), (9, 9), (99, 99), (10 ** 40, 10 ** 39 + 7)]
    for _ in range(count):
        x = rng.randint(-10 ** rng.randint(0, max_digits), 10 ** rng.randint(0, max_digits))
        y = rng.randint(-10 ** rng.randint(0, max_digits), 10 ** rng.randint(0, max_digits))
        out.append((x, y))
    return out


def test_first_screen_numbers():
    # 12 x 34: 학교식 곱 4번, Karatsuba 곱 3번
    assert (1 * 3, 1 * 4, 2 * 3, 2 * 4) == (3, 4, 6, 8)
    assert 3 * 100 + (4 + 6) * 10 + 8 == 408
    z2, z0, s = 1 * 3, 2 * 4, (1 + 2) * (3 + 4)
    assert (z2, z0, s, s - z2 - z0) == (3, 8, 21, 10)
    assert z2 * 100 + (s - z2 - z0) * 10 + z0 == 408 == 12 * 34
    c = mm.Counter()
    assert mm.karatsuba_poly_mul(12, 34, 1, c) == 408 and c.mults == 3
    c = mm.Counter()
    assert mm.schoolbook_mul(12, 34, c) == 408 and c.mults == 4
    c = mm.Counter()
    assert mm.karatsuba_poly_mul(1213, 2121, 1, c) == 2572773 == 1213 * 2121 and c.mults == 9
    c = mm.Counter()
    assert mm.schoolbook_mul(1213, 2121, c) == 2572773 and c.mults == 16
    # 첫 화면의 손계산: 조각을 더해도(12+13=25, 21+21=42) 모든 잎이 한 자리 곱이다
    tr = []
    mm.karatsuba_poly(mm.to_digits(1213), mm.to_digits(2121), 1, None, tr)
    leaves = [(p[0], q[0]) for d, p, q, r, _ in tr if len(p) == 1]
    assert len(leaves) == 9 and all(0 <= a <= 9 and 0 <= b <= 9 for a, b in leaves)
    # 표: 2^12 = 4096자리에서 4^12 대 3^12
    assert 4 ** 12 == 16777216 and 3 ** 12 == 531441
    assert round(4 ** 12 / 3 ** 12, 1) == 31.6


def test_all_multipliers_match_python_int():
    for x, y in _cases():
        want = x * y
        assert mm.schoolbook_mul(x, y) == want
        for b in (1, 2, 8, 32):
            assert mm.karatsuba_mul(x, y, b) == want, (x, y, b)
            assert mm.karatsuba_poly_mul(x, y, b) == want, (x, y, b)
        assert mm.fft_mul(x, y) == want


def test_exact_counts_powers_of_two():
    """n = 2^k 자리, 기저 크기 2^j: 계수 판 Karatsuba 는 정확히 3^(k-j) 4^j, 학교식은 4^k."""
    rng = random.Random(1)
    for k in range(0, 9):
        n = 2 ** k
        for _ in range(3):
            x = rng.randrange(10 ** (n - 1), 10 ** n)
            y = rng.randrange(10 ** (n - 1), 10 ** n)
            c = mm.Counter()
            mm.schoolbook_mul(x, y, c)
            assert c.mults == 4 ** k == n * n
            for j in range(0, k + 1):
                c = mm.Counter()
                assert mm.karatsuba_poly_mul(x, y, 2 ** j, c) == x * y
                assert c.mults == 3 ** (k - j) * 4 ** j
            c = mm.Counter()
            mm.schoolbook_split_poly(mm.to_digits(x), mm.to_digits(y), 1, c)
            assert c.mults == 4 ** k


def test_carry_version_is_near_but_not_exactly_3k():
    """자리올림 판은 (a1 + a0) 가 n/2 + 1 자리가 될 수 있어 3^k 보다 많다. 그래도 n^2 보다는 훨씬 적다."""
    rng = random.Random(2)
    n, k = 256, 8
    x = rng.randrange(10 ** (n - 1), 10 ** n)
    y = rng.randrange(10 ** (n - 1), 10 ** n)
    c = mm.Counter()
    mm.karatsuba_mul(x, y, 1, c)
    assert 3 ** k < c.mults < n * n // 4


def test_recursion_tree_levels():
    for b in (3, 4):
        lv = mm.recursion_tree(4096, b, 1)
        assert len(lv) == 13
        for row in lv:
            assert row["nodes"] == b ** row["level"]
            assert row["size"] == 4096 // 2 ** row["level"]
        assert lv[-1]["nodes"] == b ** 12


def test_widget_trace_invariants():
    a, b = mm.to_digits(31415926), mm.to_digits(27182818)
    for fn, branches, mults in ((mm.karatsuba_poly, 3, 27), (mm.schoolbook_split_poly, 4, 64)):
        tr, c = [], mm.Counter()
        res = fn(a, b, 1, c, tr)
        assert mm.from_digits(res) == 31415926 * 27182818
        assert c.mults == mults
        counts = {}
        for depth, p, q, r, _ in tr:
            counts[depth] = counts.get(depth, 0) + 1
            assert mm.from_digits(p) * mm.from_digits(q) == mm.from_digits(r)
        assert [counts[d] for d in sorted(counts)] == [branches ** d for d in range(4)]


def test_fft_recursive_matches_numpy_and_count():
    rng = np.random.default_rng(0)
    for e in range(0, 11):
        n = 2 ** e
        a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        c = mm.Counter()
        got = np.array(mm.fft_recursive(list(a), c))
        assert np.allclose(got, np.fft.fft(a), atol=1e-9 * max(1, n))
        assert c.mults == (n // 2) * e


def test_fft_counted_formula():
    for k in range(1, 10):
        n = 2 ** k
        c = mm.Counter()
        mm.fft_poly_mul_counted([9] * n, [9] * n, c)
        N = 2 * n
        assert c.mults == 3 * (N // 2) * int(math.log2(N)) + N


def test_big_round_trip_beyond_str_limit():
    """Python 3.12 의 4300자리 문자열 한도를 모듈이 끄고 있는지: 8192자리 곱과 왕복."""
    rng = random.Random(3)
    x = rng.randrange(10 ** 4095, 10 ** 4096)
    y = rng.randrange(10 ** 4095, 10 ** 4096)
    p = x * y
    assert mm.from_digits(mm.to_digits(p)) == p
    assert len(mm.to_digits(p)) >= 8191


def test_fft_mul_bits_exact_small_and_breaks_large():
    rng = random.Random(4)
    for bits in (4, 8, 12, 16):
        for n in (1, 7, 64, 300):
            x = rng.getrandbits(n * bits) | 1
            y = rng.getrandbits(n * bits) | 1
            v, dist = mm.fft_mul_bits(x, y, bits)
            assert v == x * y and dist < 0.5
    # 24비트 계수는 계수 곱의 합이 2^53 을 넘어 256개만 돼도 틀린다(모든 자리가 최대인 입력)
    n, bits = 256, 24
    x = (1 << (n * bits)) - 1
    v, _ = mm.fft_mul_bits(x, x, bits)
    assert v != x * x


def test_to_chunks_round_trip():
    rng = random.Random(5)
    for bits in (4, 8, 12, 20):
        x = rng.getrandbits(1000)
        ch = mm.to_chunks(x, bits)
        assert sum(v << (bits * i) for i, v in enumerate(ch)) == x
        assert all(0 <= v < (1 << bits) for v in ch)
