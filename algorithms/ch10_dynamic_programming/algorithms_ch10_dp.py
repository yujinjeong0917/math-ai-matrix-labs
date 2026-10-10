"""알고리즘 10장 최소 구현: 겹치는 부분 문제를 다시 풀지 않는 방법(메모, 표 채우기).

다루는 것
    피보나치    : fib_naive(재귀), fib_memo(위에서 아래로 묻고 적어 두기), fib_table(아래에서 위로 채우기)
    편집거리    : edit_distance_naive(재귀), edit_distance_memo, edit_distance_table(NumPy 표),
                  edit_distance_two_rows(행 두 개), backtrace / apply_ops(표에서 고치기 순서 읽기)
    비터비      : viterbi(로그 공간), viterbi_prob(확률을 그대로 곱하는 판, 언더플로 실험용),
                  viterbi_bruteforce(모든 경로 비교)
    거스름돈    : coin_change_dp(8장에서 복사), coin_change_bruteforce

셈 규칙
    calls : 재귀 호출 수(처음 부르는 것 포함).
    cells : 표에서 값을 계산해 채운 칸 수.
    ops   : 비터비에서 "후보 점수 하나를 계산해 비교에 올린" 횟수.
            첫 시점 K번, 그 뒤 시점마다 상태 K개 x 이전 상태 K개 = K^2번.
            전수 탐색은 경로 K^T개마다 점수 하나를 계산하는데, 경로 하나의 점수에 더하는 항이 2T개다
            (첫 상태 1, 전이 T-1, 관측 T). 그래서 전수 탐색 ops = K^T * 2T 로 센다.

이 모듈은 불러올 때 재귀 한도(sys.setrecursionlimit)를 바꾸지 않는다. 위에서 아래로 묻는 메모 판이
기본 한도(1000)에서 어디서 멈추는지가 이 장의 실험 하나이기 때문이다.
"""

import itertools

import numpy as np

INF = float("inf")


class Counter:
    def __init__(self):
        self.calls = 0
        self.cells = 0
        self.ops = 0


def _c(counter):
    return counter if counter is not None else Counter()


# ---------------------------------------------------------------- 피보나치

def fib_naive(n, counter=None, visits=None):
    """F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2) 를 그대로 재귀로. visits[k] 에 F(k) 를 부른 횟수를 쌓는다."""
    c = _c(counter)

    def f(k):
        c.calls += 1
        if visits is not None:
            visits[k] += 1
        if k < 2:
            return k
        return f(k - 1) + f(k - 2)

    return f(n)


def fib_memo(n, counter=None):
    """위에서 아래로 묻되, 처음 계산한 값을 적어 두고 다시 물으면 적어 둔 값을 돌려준다.

    재귀 깊이가 n 근처까지 가므로, 기본 재귀 한도(1000)에서는 n 이 1000 근처면 RecursionError 가 난다.
    """
    c = _c(counter)
    memo = {}

    def f(k):
        c.calls += 1
        if k in memo:
            return memo[k]
        v = k if k < 2 else f(k - 1) + f(k - 2)
        memo[k] = v
        return v

    return f(n)


def fib_table(n, counter=None):
    """아래에서 위로: F(0), F(1) 부터 F(n) 까지 차례로 채운다. 재귀가 없다."""
    c = _c(counter)
    if n < 2:
        c.cells += 1
        return n
    t = [0] * (n + 1)
    t[1] = 1
    c.cells += 2
    for k in range(2, n + 1):
        t[k] = t[k - 1] + t[k - 2]
        c.cells += 1
    return t[n]


def fib_value(n):
    """검증용: 두 변수만 들고 가는 반복."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------- 편집거리

def edit_distance_naive(a, b, counter=None, visits=None, shortcut=False):
    """d(i, j) = a 의 앞 i 글자를 b 의 앞 j 글자로 바꾸는 최소 횟수(넣기·지우기·바꾸기 각 1).

    shortcut=False : 세 갈래(지우기, 넣기, 바꾸기/일치)를 늘 모두 부른다. 호출 수가 글자와 무관하다.
    shortcut=True  : 마지막 글자가 같으면 대각선 하나만 부른다(같을 때는 그쪽이 늘 최소라서 답은 같다).
                     호출 수가 입력에 따라 달라진다.
    visits         : (len(a)+1) x (len(b)+1) 리스트. 칸 (i, j) 를 부른 횟수를 쌓는다.
    """
    c = _c(counter)

    def d(i, j):
        c.calls += 1
        if visits is not None:
            visits[i][j] += 1
        if i == 0:
            return j
        if j == 0:
            return i
        same = a[i - 1] == b[j - 1]
        if shortcut and same:
            return d(i - 1, j - 1)
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (0 if same else 1))

    return d(len(a), len(b))


def naive_call_count(m, n):
    """shortcut=False 판의 호출 수. C(i,j) = 1 + C(i-1,j) + C(i,j-1) + C(i-1,j-1), 가장자리는 1."""
    C = [[1] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            C[i][j] = 1 + C[i - 1][j] + C[i][j - 1] + C[i - 1][j - 1]
    return C[m][n]


def delannoy(p, q):
    """Delannoy 수 D(p, q): (0,0) 에서 (p,q) 까지 오른쪽·위·대각선 한 칸씩으로 가는 길의 수."""
    D = [[1] * (q + 1) for _ in range(p + 1)]
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            D[i][j] = D[i - 1][j] + D[i][j - 1] + D[i - 1][j - 1]
    return D[p][q]


def edit_distance_memo(a, b, counter=None):
    """위에서 아래로 묻되 칸마다 처음 계산한 값을 적어 둔다. 재귀 깊이는 최대 len(a)+len(b)."""
    c = _c(counter)
    memo = {}

    def d(i, j):
        c.calls += 1
        key = (i, j)
        if key in memo:
            return memo[key]
        if i == 0:
            v = j
        elif j == 0:
            v = i
        else:
            v = min(d(i - 1, j) + 1, d(i, j - 1) + 1,
                    d(i - 1, j - 1) + (0 if a[i - 1] == b[j - 1] else 1))
        memo[key] = v
        c.cells += 1
        return v

    return d(len(a), len(b))


def edit_distance_table(a, b, counter=None, order=None):
    """아래에서 위로 표 D 를 채운다. D[i][j] 는 a[:i] -> b[:j] 의 편집거리. NumPy int 배열을 돌려준다.

    order 리스트를 주면 채운 칸 (i, j) 를 채운 순서대로 넣는다(위젯과 순서 불변식 검사용).
    """
    c = _c(counter)
    m, n = len(a), len(b)
    D = np.zeros((m + 1, n + 1), dtype=np.int64)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                D[i, j] = j
            elif j == 0:
                D[i, j] = i
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                              D[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
            c.cells += 1
            if order is not None:
                order.append((i, j))
    return D


def edit_distance_two_rows(a, b, counter=None):
    """표의 바로 윗줄과 지금 줄만 들고 간다. 메모리는 짧은 쪽 길이 + 1 칸 두 줄."""
    c = _c(counter)
    if len(b) > len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    c.cells += len(prev)
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        c.cells += 1
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
            c.cells += 1
        prev = cur
    return prev[-1]


def backtrace(D, a, b):
    """표의 오른쪽 아래에서 거꾸로 걸어 고치기 순서를 읽는다.

    같은 값이 여러 갈래에서 나오면 일치/바꾸기(대각선) -> 지우기(위) -> 넣기(왼쪽) 순으로 고른다.
    돌려주는 것: [(연산, i, j), ...] 앞에서부터의 순서. 연산은 'match', 'sub', 'del', 'ins'.
    """
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if D[i, j] == D[i - 1, j - 1] + cost:
                ops.append(("match" if cost == 0 else "sub", i, j))
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i, j))
            i -= 1
            continue
        ops.append(("ins", i, j))
        j -= 1
    ops.reverse()
    return ops


def apply_ops(a, b, ops):
    """backtrace 가 준 순서대로 a 를 고쳐 나간다. 결과가 b 와 같아야 한다."""
    out = []
    for op, i, j in ops:
        if op == "match":
            out.append(a[i - 1])
        elif op == "sub":
            out.append(b[j - 1])
        elif op == "ins":
            out.append(b[j - 1])
        # del: 아무것도 붙이지 않는다
    return "".join(out)


# ---------------------------------------------------------------- 비터비

def viterbi(log_pi, log_A, log_B, obs, counter=None):
    """로그 공간 비터비. delta_t(j) = max_i [delta_{t-1}(i) + log a_ij] + log b_j(o_t).

    돌려주는 것: (가장 그럴듯한 상태 경로 리스트, 그 경로의 로그 확률).
    """
    c = _c(counter)
    log_pi, log_A, log_B = map(np.asarray, (log_pi, log_A, log_B))
    K, T = log_A.shape[0], len(obs)
    delta = log_pi + log_B[:, obs[0]]
    c.ops += K
    psi = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        cand = delta[:, None] + log_A          # cand[i, j]: i 에서 j 로 오는 후보 K*K 개
        c.ops += K * K
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(K)] + log_B[:, obs[t]]
    last = int(np.argmax(delta))
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return path, float(delta[last])


def viterbi_prob(pi, A, B, obs):
    """확률을 그대로 곱하는 판. 길이가 길면 delta 가 모두 0.0 이 된다(float64 언더플로)."""
    pi, A, B = map(np.asarray, (pi, A, B))
    K, T = A.shape[0], len(obs)
    delta = pi * B[:, obs[0]]
    first_zero = None
    psi = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        cand = delta[:, None] * A
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(K)] * B[:, obs[t]]
        if first_zero is None and not np.any(delta > 0):
            first_zero = t + 1                  # 길이(시점 수)로 센다
    last = int(np.argmax(delta))
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return path, float(delta[last]), first_zero


def path_logprob(log_pi, log_A, log_B, obs, path):
    s = log_pi[path[0]] + log_B[path[0], obs[0]]
    for t in range(1, len(obs)):
        s += log_A[path[t - 1], path[t]] + log_B[path[t], obs[t]]
    return float(s)


def viterbi_bruteforce(log_pi, log_A, log_B, obs, counter=None):
    """상태 경로 K^T 개를 모두 점수 매겨 가장 큰 것을 고른다."""
    c = _c(counter)
    log_pi, log_A, log_B = map(np.asarray, (log_pi, log_A, log_B))
    K, T = log_A.shape[0], len(obs)
    best, best_path = -INF, None
    for path in itertools.product(range(K), repeat=T):
        s = path_logprob(log_pi, log_A, log_B, obs, path)
        c.ops += 2 * T
        if s > best:
            best, best_path = s, list(path)
    return best_path, best


def random_hmm(rng, K, M):
    """상태 K 개, 관측 기호 M 개인 무작위 HMM(확률 행렬은 디리클레(1) 표본)."""
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(M), size=K)
    return pi, A, B


# ---------------------------------------------------------------- 거스름돈 (8장에서 가져옴)

def coin_change_dp(coins, amount):
    """금액 0..amount 각각의 최소 동전 수 표와, amount 를 만드는 최소 동전 목록.

    algorithms/ch08_greedy/algorithms_ch08_greedy.py 의 coin_change_dp 를 그대로 복사했다
    (장끼리 import 하지 않는다는 저장소 규칙). best[x] = min_c (best[x-c] + 1).
    """
    best = [0] + [INF] * amount
    last = [0] * (amount + 1)
    for x in range(1, amount + 1):
        for c in sorted(coins, reverse=True):
            if c <= x and best[x - c] + 1 < best[x]:
                best[x] = best[x - c] + 1
                last[x] = c
    combo = []
    x = amount
    while x > 0 and best[amount] < INF:
        combo.append(last[x])
        x -= last[x]
    return best, sorted(combo, reverse=True)


def coin_change_bruteforce(coins, amount, counter=None):
    """메모 없이 '마지막 동전'을 모두 시도하는 재귀. 같은 금액을 여러 번 다시 푼다."""
    c = _c(counter)

    def f(x):
        c.calls += 1
        if x == 0:
            return 0
        best = INF
        for coin in coins:
            if coin <= x:
                best = min(best, f(x - coin) + 1)
        return best

    return f(amount)


def greedy_coins(coins, amount):
    """큰 동전부터 쓸 수 있는 만큼 쓰기(8장의 탐욕)."""
    out = []
    for c in sorted(coins, reverse=True):
        while amount >= c:
            out.append(c)
            amount -= c
    return out


def log0(x):
    with np.errstate(divide="ignore"):
        return np.log(x)

