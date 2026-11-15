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

# 모듈을 불러오는 것만으로 프로세스 전체의 한도가 꺼지는 부작용이 있다(실험용 선택).
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
    # 낮은 자리부터 담아야 인덱스 i 가 10^i 의 계수가 된다(from_digits 도 같은 약속).
    return [int(ch) for ch in reversed(str(x))]


def from_digits(d):
    """자릿수(또는 9를 넘는 계수) 리스트 -> 정수. sum(d[i] * 10^i)."""
    v = 0
    for x in reversed(d):
        # 호너(Horner) 방식. 계수 x 가 9 를 넘어도(계수 판 결과) 올림이 자연스럽게 합쳐진다.
        v = v * 10 + x
    return v


def _trim(d):
    while len(d) > 1 and d[-1] == 0:
        d.pop()
    return d


def _add_digits(a, b, c):
    """자릿수 리스트 두 개를 더한다(자리올림 처리). 자리마다 adds 1."""
    # 길이가 다른 두 리스트를 더할 때 짧은 쪽은 0 으로 본다.
    n = max(len(a), len(b))
    out, carry = [], 0
    for i in range(n):
        s = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) + carry
        c.adds += 1
        out.append(s % 10)
        # 올림은 한 자리 합이 최대 9+9+1 = 19 라 항상 0 또는 1.
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
        # 끝에 빌림이 남으면 a < b. Karatsuba 에서 z1 = (합의 곱) - z0 - z2 는 항상 >= 0 이라 정상 경로에서는 일어나지 않는다.
        raise ValueError("_sub_digits: a < b")
    return _trim(out)


def _shift(d, k):
    # k 자리 왼쪽 이동 = 10^k 곱. 0 은 앞에 0 을 붙이지 않는다(_trim 과 형식을 맞추기 위함).
    return [0] * k + d if d != [0] else [0]


def _signed(fn):
    def wrapper(x, y, *args, **kwargs):
        # 부호는 따로 빼고 절댓값만 알고리즘에 넘긴다(자릿수 리스트는 음수를 못 담음).
        sign = -1 if (x < 0) != (y < 0) else 1
        return sign * fn(abs(x), abs(y), *args, **kwargs)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------- 학교식

def schoolbook_digits(a, b, counter=None):
    """학교식 곱셈. 한 자리 곱을 정확히 len(a) * len(b) 번 한다."""
    c = _c(counter)
    # 곱의 자릿수는 최대 len(a) + len(b). 시간 O(len(a) x len(b)) = O(n^2).
    out = [0] * (len(a) + len(b))
    for i, ai in enumerate(a):
        carry = 0
        for j, bj in enumerate(b):
            c.mults += 1
            # 한 줄을 곱할 때 기존 out 칸에 누적한다. 최대 9 + 81 + 올림(<= 9) 이라 정수 한 칸 안에서 안전.
            t = out[i + j] + ai * bj + carry
            c.adds += 1
            out[i + j] = t % 10
            carry = t // 10
        k = i + len(b)
        # 윗줄 끝의 남은 올림을 전파(마지막 칸 너머로 이어질 수 있어 while).
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
    # n 이 기저 크기 이하면 학교식으로. 점화식 T(n) = 3 T(n/2) + O(n) -> O(n^(log2 3)) = O(n^1.585).
    n = max(len(a), len(b))
    if n <= base_case:
        return schoolbook_digits(a, b, c)
    # 두 수를 같은 길이 n 으로 맞춰야 쪼개는 지점 h 가 일치한다.
    a = a + [0] * (n - len(a))
    b = b + [0] * (n - len(b))
    # 낮은 쪽 h 자리(a0)와 높은 쪽(a1). n 이 홀수면 높은 쪽이 한 자리 더 길다.
    h = n // 2
    a0, a1, b0, b1 = a[:h], a[h:], b[:h], b[h:]
    z0 = karatsuba_digits(a0, b0, base_case, c)
    z2 = karatsuba_digits(a1, b1, base_case, c)
    sa = _add_digits(a0, a1, c)          # n/2 또는 n/2+1 자리: 여기서 3^k 와 조금 어긋난다
    sb = _add_digits(b0, b1, c)
    z1 = karatsuba_digits(sa, sb, base_case, c)
    # (a1 + a0)(b1 + b0) - a1 b1 - a0 b0 = a1 b0 + a0 b1. 곱셈 4번 대신 3번으로 가운데 항을 얻는 핵심 항등식.
    z1 = _sub_digits(_sub_digits(z1, z0, c), z2, c)
    # a b = z2 10^(2h) + z1 10^h + z0. 10 의 거듭제곱 곱은 자리 이동일 뿐이라 곱셈 횟수에 안 센다(더하기 O(n)).
    out = _add_digits(_add_digits(z0, _shift(z1, h), c), _shift(z2, 2 * h), c)
    return _trim(out)


@_signed
def karatsuba_mul(x, y, base_case=1, counter=None):
    return from_digits(karatsuba_digits(to_digits(x), to_digits(y), base_case, counter))


# ---------------------------------------------------------------- Karatsuba (계수 판)

def schoolbook_poly(p, q, counter=None):
    """계수 리스트의 곱(합성곱). 자리올림 없이 계수 곱을 len(p) * len(q) 번."""
    c = _c(counter)
    # 합성곱: 결과 길이 = len(p) + len(q) - 1, out[k] = sum_{i+j=k} p[i] q[j].
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
    # n & (n-1) == 0 은 n 이 2 의 거듭제곱인지 보는 비트 요령.
    assert len(q) == n and n & (n - 1) == 0, "길이는 같은 2의 거듭제곱이어야 한다"
    if n <= base_case:
        out = schoolbook_poly(p, q, c)
    else:
        h = n // 2
        p0, p1, q0, q1 = p[:h], p[h:], q[:h], q[h:]
        z0 = karatsuba_poly(p0, q0, base_case, c, trace, depth + 1)
        z2 = karatsuba_poly(p1, q1, base_case, c, trace, depth + 1)
        # 계수 판은 올림을 안 하므로 합의 길이가 그대로 h. 자리올림 판과 다르게 곱 횟수가 정확히 3^(k-j) 4^j.
        sp = [x + y for x, y in zip(p0, p1)]
        sq = [x + y for x, y in zip(q0, q1)]
        c.adds += 2 * h
        z1 = karatsuba_poly(sp, sq, base_case, c, trace, depth + 1)
        # 가운데 항 z1 = (합의 곱) - z0 - z2. 길이 모두 2h-1 로 같다.
        z1 = [m - u - v for m, u, v in zip(z1, z0, z2)]
        c.adds += 2 * len(z1)
        # z0 은 위치 0, z1 은 h, z2 는 2h 부터 겹쳐 더한다(합성곱의 결과 길이 2n-1).
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
        # 비교용: 같은 쪼개기에서 곱을 4번(z0, p0q1, p1q0, z2). T(n) = 4 T(n/2) -> n^2 번이라 학교식과 곱 횟수가 같다.
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


# d 를 0 으로 채워 길이를 2 의 거듭제곱(또는 지정 n)으로 맞춘다.
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
    # 층 j 의 노드 수 branches^j, 크기 n/2^j. 층 작업량 = nodes x size = n (branches/2)^j. 3 갈래면 잎 층이 가장 크다.
    levels, size, nodes = [], n, 1
    while True:
        levels.append({"level": len(levels), "nodes": nodes, "size": size, "work": nodes * size})
        if size <= base_case:
            break
        # n 이 2 의 거듭제곱일 때만 정확. 아니면 내림이라 어림값.
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
    # 짝수 번째/홀수 번째로 나눠 각각 크기 n/2 FFT: T(n) = 2 T(n/2) + O(n) = O(n log n).
    even = fft_recursive(a[0::2], c, inverse)
    odd = fft_recursive(a[1::2], c, inverse)
    # 순변환은 e^(-2 pi i k / n), 역변환은 부호가 반대(e^(+...)). 역변환의 1/N 은 호출하는 쪽에서 나눈다.
    sgn = 1 if inverse else -1
    out = [0j] * n
    for k in range(n // 2):
        # 회전 인자 w^k = e^(sgn x 2 pi i k / n). 매 호출마다 cos/sin 을 새로 계산(미리 표로 만들지 않음, 단순함 우선).
        w = complex(math.cos(2 * math.pi * k / n), sgn * math.sin(2 * math.pi * k / n))
        # 나비(butterfly) 연산: 한 번의 복소수 곱으로 X[k] 와 X[k + n/2] 를 둘 다 얻는다.
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
        # 합성곱 결과 길이 m 이상인 가장 작은 2 의 거듭제곱 N 으로 0 채움(순환 합성곱이 선형 합성곱과 같아지려면 필요).
        N *= 2
    fp = fft_recursive(list(p) + [0] * (N - len(p)), c)
    fq = fft_recursive(list(q) + [0] * (N - len(q)), c)
    # 점별 곱 = 합성곱의 푸리에 변환(합성곱 정리). 곱 N 번.
    prod = []
    for u, v in zip(fp, fq):
        prod.append(u * v)
        c.mults += 1
    back = fft_recursive(prod, c, inverse=True)
    # 역 FFT 후 N 으로 나누고 반올림. 정수 계수 가정이며 값이 커지면 부동소수 오차로 틀릴 수 있다(10진 자릿수에서는 작아 안전).
    return [round(z.real / N) for z in back[:m]]


@_signed
def fft_mul(x, y, counter=None):
    """10진 자릿수에 FFT 곱(fft_poly_mul_counted)을 쓰고 자리올림은 from_digits 로."""
    return from_digits(fft_poly_mul_counted(to_digits(x), to_digits(y), counter))


def to_chunks(x, bits):
    """0 이상의 정수 -> 2^bits 진법 계수(낮은 자리부터). bits 는 4의 배수(16진 문자열로 자른다)."""
    # 16진 문자열 한 글자 = 4 비트라, 문자열 자르기로 계수를 만들려는 구현 선택.
    assert bits % 4 == 0
    h = format(x, "x")
    w = bits // 4
    pad = (-len(h)) % w
    # 앞에 0 을 채워 길이를 w 의 배수로 맞춘다.
    h = "0" * pad + h
    # 끝(낮은 자리)부터 w 글자씩 잘라 낮은 자리 먼저 나열.
    return [int(h[i:i + w], 16) for i in range(len(h) - w, -1, -w)]


def fft_conv_float(p, q):
    """NumPy rfft 로 계수 합성곱을 부동소수로(반올림 전). 반환 길이 len(p)+len(q)-1."""
    m = len(p) + len(q) - 1
    N = 1
    while N < m:
        N *= 2
    # rfft: 실수 입력용(반쪽 스펙트럼만 계산). float64 가수 53 비트가 정확도 한계.
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
    # 반올림 전 값이 정수에서 떨어진 최대 거리. 0.5 에 가까우면 반올림이 위험하다는 신호(< 0.5 여도 오차가 누적됐다면 이미 틀릴 수 있음에 유의).
    dist = float(np.max(np.abs(f - r))) if len(f) else 0.0
    coeffs = [int(v) for v in r]
    # 자리올림: 2^bits 진법
    base = 1 << bits
    digits, carry = [], 0
    for v in coeffs:
        s = v + carry
        # s & (2^bits - 1) = 하위 bits 비트(나머지), s >> bits = 올림. 2 의 거듭제곱 진법이라 나눗셈 대신 비트 연산.
        digits.append(s & (base - 1))
        carry = s >> bits
    while carry:
        digits.append(carry & (base - 1))
        carry >>= bits
    w = bits // 4
    # 각 자리를 w 글자 16진수로 채워 이어 붙인 뒤 한 번에 int 로 변환.
    h = "".join(format(d, "0%dx" % w) for d in reversed(digits))
    return int(h, 16) if h else 0, dist
