"""알고리즘 9장 최소 구현: 학교식 곱셈, Karatsuba 곱셈(두 가지 셈법), 재귀 FFT와 FFT 곱셈.

숫자는 10진 자릿수 리스트로 다루고, 리스트는 낮은 자리부터 적는다(1234 -> [4, 3, 2, 1]).
Python 3.12 는 4300자리가 넘는 정수를 10진 문자열로 바꾸면 ValueError 를 낸다(sys.int_info).
8192자리 곱을 다루려고 이 모듈은 불러올 때 그 한도를 끈다(sys.set_int_max_str_digits(0)).

셈 규칙
    mults : 두 수(한 자리 숫자 또는 계수)를 곱한 횟수. 곱셈 알고리즘끼리 비교하는 기준이다.
    adds  : 자리마다 더하거나 뺀 횟수(자리올림 처리 포함). 쪼개고 합치는 데 드는 O(n) 작업이다.
    calls : 재귀 호출 수.

Karatsuba 는 두 가지로 구현한다.
    karatsuba_mul      : 자리올림을 매번 처리하는 "진짜 정수" 판. 반으로 쪼갠 두 조각의 합이 n/2+1 자리가
                         될 수 있어서, 곱 횟수가 3^k 에 가깝지만 정확히 같지는 않다.
    karatsuba_poly_mul : 자리올림을 맨 끝에 한 번만 하는 "계수" 판. 합한 계수가 9를 넘어도(예: 7+8=15)
                         그대로 두기 때문에, 자릿수 n = 2^k, 기저 크기 b = 2^j 이면 계수 곱이 정확히
                         3^(k-j) * 4^j 번이다. 기저 크기 1이면 3^k.
FFT 는 두 가지다.
    fft_recursive / fft_poly_mul_counted : 순수 Python 반씩 쪼개기 FFT. 복소수 곱 횟수를 센다.
    fft_mul_bits                         : NumPy FFT 로 2^bits 진법 계수를 곱한다. 부동소수 반올림 실험용.
"""

import math
import sys

import numpy as np

sys.set_int_max_str_digits(0)


class Counter:
    def __init__(self):
        self.mults = 0
        self.adds = 0
        self.calls = 0


def _c(counter):
    return counter if counter is not None else Counter()


# ---------------------------------------------------------------- 자릿수

def to_digits(x):
    """0 이상의 정수 -> 낮은 자리부터의 10진 자릿수 리스트. 0 은 [0]."""
    if x < 0:
        raise ValueError("to_digits 는 0 이상만 받는다")
    return [int(ch) for ch in reversed(str(x))]


def from_digits(d):
    """자릿수(또는 9를 넘는 계수) 리스트 -> 정수. sum(d[i] * 10^i)."""
    v = 0
    for x in reversed(d):
        v = v * 10 + x
    return v


def _trim(d):
    while len(d) > 1 and d[-1] == 0:
        d.pop()
    return d


def _add_digits(a, b, c):
    """자릿수 리스트 두 개를 더한다(자리올림 처리). 자리마다 adds 1."""
    n = max(len(a), len(b))
    out, carry = [], 0
    for i in range(n):
        s = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) + carry
        c.adds += 1
        out.append(s % 10)
        carry = s // 10
    if carry:
        out.append(carry)
    return out


def _sub_digits(a, b, c):
    """a - b (a >= b 가정). 자리마다 adds 1."""
    out, borrow = [], 0
    for i in range(len(a)):
        s = a[i] - (b[i] if i < len(b) else 0) - borrow
        c.adds += 1
        if s < 0:
            s += 10
            borrow = 1
        else:
            borrow = 0
        out.append(s)
    if borrow:
        raise ValueError("_sub_digits: a < b")
    return _trim(out)


def _shift(d, k):
    return [0] * k + d if d != [0] else [0]


def _signed(fn):
    def wrapper(x, y, *args, **kwargs):
        sign = -1 if (x < 0) != (y < 0) else 1
        return sign * fn(abs(x), abs(y), *args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------- 학교식

def schoolbook_digits(a, b, counter=None):
    """학교식 곱셈. 한 자리 곱을 정확히 len(a) * len(b) 번 한다."""
    c = _c(counter)
    out = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        carry = 0
        for j, bj in enumerate(b):
            c.mults += 1
            t = out[i + j] + ai * bj + carry
            c.adds += 1
            out[i + j] = t % 10
            carry = t // 10
        k = i + len(b)
        while carry:
            t = out[k] + carry
            c.adds += 1
            out[k] = t % 10
            carry = t // 10
            k += 1
    return _trim(out)


@_signed
def schoolbook_mul(x, y, counter=None):
    """정수 x * y 를 학교식으로. 음수와 0 도 받는다."""
    return from_digits(schoolbook_digits(to_digits(x), to_digits(y), counter))


# ---------------------------------------------------------------- Karatsuba (자리올림 판)

def karatsuba_digits(a, b, base_case=1, counter=None):
    """Karatsuba 곱셈. 두 수를 같은 길이 n 으로 맞춘 뒤 아래 h = n//2 자리와 위 자리로 쪼갠다.

    a = a1 * 10^h + a0, b = b1 * 10^h + b0 일 때
        z2 = a1 b1, z0 = a0 b0, z1 = (a1 + a0)(b1 + b0) - z2 - z0
        a b = z2 10^{2h} + z1 10^h + z0
    곱은 세 번이다. n <= base_case 이면 학교식으로 넘긴다.
    """
    c = _c(counter)
    c.calls += 1
    n = max(len(a), len(b))
    if n <= base_case:
        return schoolbook_digits(a, b, c)
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    h = n // 2
    a0, a1, b0, b1 = a[:h], a[h:], b[:h], b[h:]
    z0 = karatsuba_digits(a0, b0, base_case, c)
    z2 = karatsuba_digits(a1, b1, base_case, c)
    sa = _add_digits(a0, a1, c)          # n/2 또는 n/2+1 자리: 여기서 3^k 와 조금 어긋난다
    sb = _add_digits(b0, b1, c)
    z1 = karatsuba_digits(sa, sb, base_case, c)
    z1 = _sub_digits(_sub_digits(z1, z0, c), z2, c)
    out = _add_digits(_add_digits(z0, _shift(z1, h), c), _shift(z2, 2 * h), c)
    return _trim(out)


@_signed
def karatsuba_mul(x, y, base_case=1, counter=None):
    return from_digits(karatsuba_digits(to_digits(x), to_digits(y), base_case, counter))


# ---------------------------------------------------------------- Karatsuba (계수 판)

def schoolbook_poly(p, q, counter=None):
    """계수 리스트의 곱(합성곱). 자리올림 없이 계수 곱을 len(p) * len(q) 번."""
    c = _c(counter)
    out = [0] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            c.mults += 1
            c.adds += 1
            out[i + j] += pi * qj
    return out


def karatsuba_poly(p, q, base_case=1, counter=None, trace=None, depth=0):
    """len(p) == len(q) == 2의 거듭제곱인 계수 리스트의 곱. 결과 길이 2n-1.

    trace 에 리스트를 주면, 노드가 끝날 때마다 (깊이, p, q, 결과, 누적 곱) 을 붙인다(위젯용).
    """
    c = _c(counter)
    c.calls += 1
    n = len(p)
    assert len(q) == n and n & (n - 1) == 0, "길이는 같은 2의 거듭제곱이어야 한다"
    if n <= base_case:
        out = schoolbook_poly(p, q, c)
    else:
        h = n // 2
        p0, p1, q0, q1 = p[:h], p[h:], q[:h], q[h:]
        z0 = karatsuba_poly(p0, q0, base_case, c, trace, depth + 1)
        z2 = karatsuba_poly(p1, q1, base_case, c, trace, depth + 1)
        sp = [x + y for x, y in zip(p0, p1)]
        sq = [x + y for x, y in zip(q0, q1)]
        c.adds += 2 * h
        z1 = karatsuba_poly(sp, sq, base_case, c, trace, depth + 1)
        z1 = [m - u - v for m, u, v in zip(z1, z0, z2)]
        c.adds += 2 * len(z1)
        out = [0] * (2 * n - 1)
        for i, v in enumerate(z0):
            out[i] += v
        for i, v in enumerate(z1):
            out[i + h] += v
        for i, v in enumerate(z2):
            out[i + 2 * h] += v
        c.adds += 3 * len(z0)
    if trace is not None:
        trace.append((depth, list(p), list(q), list(out), c.mults))
    return out


def schoolbook_split_poly(p, q, base_case=1, counter=None, trace=None, depth=0):
    """Karatsuba 와 똑같이 반으로 쪼개되 곱을 네 번 하는 판(비교용). 계수 곱이 정확히 n^2."""
    c = _c(counter)
    c.calls += 1
    n = len(p)
    if n <= base_case:
        out = schoolbook_poly(p, q, c)
    else:
        h = n // 2
        p0, p1, q0, q1 = p[:h], p[h:], q[:h], q[h:]
        z0 = schoolbook_split_poly(p0, q0, base_case, c, trace, depth + 1)
        za = schoolbook_split_poly(p0, q1, base_case, c, trace, depth + 1)
        zb = schoolbook_split_poly(p1, q0, base_case, c, trace, depth + 1)
        z2 = schoolbook_split_poly(p1, q1, base_case, c, trace, depth + 1)
        out = [0] * (2 * n - 1)
        for i in range(len(z0)):
            out[i] += z0[i]
            out[i + h] += za[i] + zb[i]
            out[i + 2 * h] += z2[i]
        c.adds += 4 * len(z0)
    if trace is not None:
        trace.append((depth, list(p), list(q), list(out), c.mults))
    return out


def _pad_pow2(d, n=None):
    if n is None:
        n = 1
        while n < len(d):
            n *= 2
    return d + [0] * (n - len(d))


@_signed
def karatsuba_poly_mul(x, y, base_case=1, counter=None):
    """계수 판 Karatsuba 로 x * y. 두 수를 같은 2의 거듭제곱 길이로 맞추고, 자리올림은 from_digits 가 한 번에."""
    a, b = to_digits(x), to_digits(y)
    n = 1
    while n < max(len(a), len(b)):
        n *= 2
    return from_digits(karatsuba_poly(_pad_pow2(a, n), _pad_pow2(b, n), base_case, counter))


# ---------------------------------------------------------------- 재귀 트리

def recursion_tree(n, branches=3, base_case=1):
    """크기 n 문제를 반으로 쪼개 branches 개 부분 문제로 푸는 재귀 트리의 층별 요약.

    층 j: 노드 branches^j 개, 노드마다 크기 n / 2^j, 층 전체 크기 합 n (branches/2)^j.
    """
    levels, size, nodes = [], n, 1
    while True:
        levels.append({"level": len(levels), "nodes": nodes, "size": size, "work": nodes * size})
        if size <= base_case:
            break
        size //= 2
        nodes *= branches
    return levels


# ---------------------------------------------------------------- FFT

def fft_recursive(a, counter=None, inverse=False):
    """반씩 쪼개는(radix-2) FFT. len(a) 는 2의 거듭제곱.

    짝수 번째와 홀수 번째로 나눠 각각 FFT 한 뒤, X[k] = E[k] + w^k O[k], X[k+n/2] = E[k] - w^k O[k].
    한 층에서 복소수 곱 w^k * O[k] 를 n/2 번 하므로 전체는 (n/2) log2 n 번이다.
    T(n) = 2 T(n/2) + O(n).
    """
    c = _c(counter)
    n = len(a)
    if n == 1:
        return [complex(a[0])]
    even = fft_recursive(a[0::2], c, inverse)
    odd = fft_recursive(a[1::2], c, inverse)
    sgn = 1 if inverse else -1
    out = [0j] * n
    for k in range(n // 2):
        w = complex(math.cos(2 * math.pi * k / n), sgn * math.sin(2 * math.pi * k / n))
        t = w * odd[k]
        c.mults += 1
        c.adds += 2
        out[k] = even[k] + t
        out[k + n // 2] = even[k] - t
    return out


def fft_poly_mul_counted(p, q, counter=None):
    """계수 리스트 곱을 FFT 로: 점 N 개에서 값을 구하고(FFT 2번), 점마다 곱하고(N번), 되돌린다(역 FFT 1번).

    복소수 곱 횟수는 3 (N/2) log2 N + N, N 은 len(p)+len(q)-1 이상인 2의 거듭제곱.
    """
    c = _c(counter)
    m = len(p) + len(q) - 1
    N = 1
    while N < m:
        N *= 2
    fp = fft_recursive(list(p) + [0] * (N - len(p)), c)
    fq = fft_recursive(list(q) + [0] * (N - len(q)), c)
    prod = []
    for u, v in zip(fp, fq):
        prod.append(u * v)
        c.mults += 1
    back = fft_recursive(prod, c, inverse=True)
    return [round(z.real / N) for z in back[:m]]


@_signed
def fft_mul(x, y, counter=None):
    """10진 자릿수에 FFT 곱(fft_poly_mul_counted)을 쓰고 자리올림은 from_digits 로."""
    return from_digits(fft_poly_mul_counted(to_digits(x), to_digits(y), counter))


def to_chunks(x, bits):
    """0 이상의 정수 -> 2^bits 진법 계수(낮은 자리부터). bits 는 4의 배수(16진 문자열로 자른다)."""
    assert bits % 4 == 0
    h = format(x, "x")
    w = bits // 4
    pad = (-len(h)) % w
    h = "0" * pad + h
    return [int(h[i:i + w], 16) for i in range(len(h) - w, -1, -w)]


def fft_conv_float(p, q):
    """NumPy rfft 로 계수 합성곱을 부동소수로(반올림 전). 반환 길이 len(p)+len(q)-1."""
    m = len(p) + len(q) - 1
    N = 1
    while N < m:
        N *= 2
    fp = np.fft.rfft(np.asarray(p, dtype=np.float64), N)
    fq = np.fft.rfft(np.asarray(q, dtype=np.float64), N)
    return np.fft.irfft(fp * fq, N)[:m]


def fft_mul_bits(x, y, bits):
    """2^bits 진법 계수로 쪼개 NumPy FFT 로 곱하고, 반올림 뒤 자리올림해 정수로 되돌린다.

    float64 의 가수는 53비트라, 계수 곱의 합이 커지면 반올림이 엉뚱한 정수로 간다.
    반환: (곱, 반올림 전 값과 반올림한 정수의 최대 거리).
    """
    p, q = to_chunks(x, bits), to_chunks(y, bits)
    f = fft_conv_float(p, q)
    r = np.rint(f)
    dist = float(np.max(np.abs(f - r))) if len(f) else 0.0
    coeffs = [int(v) for v in r]
    # 자리올림: 2^bits 진법
    base = 1 << bits
    digits, carry = [], 0
    for v in coeffs:
        s = v + carry
        digits.append(s & (base - 1))
        carry = s >> bits
    while carry:
        digits.append(carry & (base - 1))
        carry >>= bits
    w = bits // 4
    h = "".join(format(d, "0%dx" % w) for d in reversed(digits))
    return int(h, 16) if h else 0, dist
