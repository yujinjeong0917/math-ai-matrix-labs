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

    # 시간 O(2^n) 꼴: 호출 수 = 2 F(n+1) - 1 (F 는 피보나치). 같은 부분 문제 F(k) 를 반복해서 푼다.
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
    # 메모(딕셔너리). 부분 문제 n+1 개를 각각 한 번만 계산 -> 시간 O(n), 공간 O(n). 재귀 깊이가 n 이라 깊은 입력에서 RecursionError.
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
    # 표 채우기: 불변식 = k 번째 칸을 채울 때 t[0..k-1] 은 이미 정답. 시간 O(n) 칸 연산(큰 정수 덧셈 비용은 따로), 공간 O(n). 두 변수만 쓰면 O(1) 공간(fib_value).
    t = [0] * (n + 1)
    t[1] = 1
    c.cells += 2
    for k in range(2, n + 1):
        t[k] = t[k - 1] + t[k - 2]
        c.cells += 1
    return t[n]


def fib_value(n):
    """검증용: 두 변수만 들고 가는 반복."""
    # a = F(k), b = F(k+1) 을 들고 한 칸씩 전진.
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
        # d(i, j) 는 a 의 앞 i 글자, b 의 앞 j 글자 사이 거리라 마지막 글자는 a[i-1], b[j-1] (0 부터 세는 인덱스 함정).
        same = a[i - 1] == b[j - 1]
        # 마지막 글자가 같을 때 대각선만 보는 가지치기(docstring 설명대로 답은 같다). 그래도 최악에는 지수 시간.
        if shortcut and same:
            return d(i - 1, j - 1)
        # 점화식: 지우기(위), 넣기(왼쪽), 바꾸기 또는 일치(대각선) 세 후보 중 최소.
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (0 if same else 1))

    return d(len(a), len(b))


def naive_call_count(m, n):
    """shortcut=False 판의 호출 수. C(i,j) = 1 + C(i-1,j) + C(i,j-1) + C(i-1,j-1), 가장자리는 1."""
    # 가장자리(i = 0 또는 j = 0)는 호출 한 번으로 끝나서 1. 안쪽은 자기 1 + 세 갈래 호출 수의 합.
    C = [[1] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            C[i][j] = 1 + C[i - 1][j] + C[i][j - 1] + C[i - 1][j - 1]
    return C[m][n]


def delannoy(p, q):
    """Delannoy 수 D(p, q): (0,0) 에서 (p,q) 까지 오른쪽·위·대각선 한 칸씩으로 가는 길의 수."""
    # 첫 행·첫 열은 길이 1 (한 방향으로만 갈 수 있는 길 하나). 안쪽 점화식은 편집거리 호출 수와 모양이 같다.
    D = [[1] * (q + 1) for _ in range(p + 1)]
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            D[i][j] = D[i - 1][j] + D[i][j - 1] + D[i - 1][j - 1]
    return D[p][q]


def edit_distance_memo(a, b, counter=None):
    """위에서 아래로 묻되 칸마다 처음 계산한 값을 적어 둔다. 재귀 깊이는 최대 len(a)+len(b)."""
    c = _c(counter)
    # 편집거리 메모: 칸 (i, j) 마다 한 번만 계산 -> 시간 O(m n), 공간 O(m n). 호출 수는 칸 수의 최대 3배 정도(칸마다 세 갈래 질문).
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
    # 표 D[i, j]: 시간 O(m n), 공간 O(m n). 불변식: (i, j) 를 채울 때 위, 왼쪽, 대각선 칸은 이미 채워져 있다(행 우선 순회가 보장).
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
    # 편집거리는 대칭이라(넣기와 지우기 비용이 같음) 짧은 쪽을 열로 삼아 메모리를 줄인다.
    if len(b) > len(a):
        a, b = b, a
    # 0번 행: D[0][j] = j (a 가 빈 문자열이면 j 글자를 넣어야 함).
    prev = list(range(len(b) + 1))
    c.cells += len(prev)
    for i in range(1, len(a) + 1):
        # 0번 열: D[i][0] = i. 표 전체를 저장하지 않으므로 역추적(backtrace)은 이 판으로 할 수 없다.
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
            # 표 값이 어느 칸에서 왔는지 역으로 확인. 같은 값이 여러 칸에서 나오면 대각선을 먼저 고르는 규칙이라 최적 순서가 여러 개일 때 하나만 나온다.
            if D[i, j] == D[i - 1, j - 1] + cost:
                ops.append(("match" if cost == 0 else "sub", i, j))
                i, j = i - 1, j - 1
                continue
        if i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i, j))
            i -= 1
            continue
        # 위 두 가지가 아니면 남은 후보는 넣기(왼쪽). i == 0 이면 이쪽으로만 온다.
        ops.append(("ins", i, j))
        j -= 1
    # 끝에서 거꾸로 모았으므로 앞에서부터의 순서로 뒤집는다.
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
        # ins 는 b 의 글자를 새로 끼워 넣고(b[j-1]), sub 도 b 쪽 글자로 바꾼다.
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
    # delta_t(j) = 시점 t 에 상태 j 에서 끝나는 가장 좋은 경로의 로그 확률. 확률의 곱을 로그의 합으로 바꾸는 이유: 긴 곱이 0 으로 언더플로하는 것을 피하려고. 시간 O(T K^2), 공간 O(T K)(psi 표).
    delta = log_pi + log_B[:, obs[0]]
    c.ops += K
    # psi[t, j] = 시점 t 에 상태 j 로 오는 최선의 직전 상태(역추적 포인터). (T, K)
    psi = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        # delta[:, None] 은 (K, 1), log_A 는 (K, K) 라 브로드캐스트로 cand 는 (K, K).
        cand = delta[:, None] + log_A          # cand[i, j]: i 에서 j 로 오는 후보 K*K 개
        c.ops += K * K
        # 열(도착 상태 j)마다 가장 큰 행(출발 상태 i) 선택. 동점이면 번호가 작은 상태가 선택된다(numpy 의 argmax 규칙).
        psi[t] = np.argmax(cand, axis=0)
        # max_i 값을 뽑아 관측 로그 확률 log b_j(o_t) 를 더한다(식의 마지막 항).
        delta = cand[psi[t], np.arange(K)] + log_B[:, obs[t]]
    last = int(np.argmax(delta))
    path = [last]
    # 마지막 시점에서 psi 포인터를 따라 거꾸로 상태를 모은다.
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
        # 확률 공간 판: 같은 구조지만 곱셈이라 길이가 길면 값이 float64 의 표현 한계 아래로 내려가 0.0 이 된다.
        cand = delta[:, None] * A
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(K)] * B[:, obs[t]]
        if first_zero is None and not np.any(delta > 0):
            # t 는 0 부터 센 시점 번호라 길이는 t + 1 (off-by-one 주의).
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
    # K^T 개 경로를 모두 평가: 시간 O(K^T x T). 비터비의 O(T K^2) 와 대비되는 기준선(작은 K, T 전용).
    for path in itertools.product(range(K), repeat=T):
        # 경로 로그 점수 = log pi + 관측 로그 + 전이 로그들의 합(항 2T 개).
        s = path_logprob(log_pi, log_A, log_B, obs, path)
        c.ops += 2 * T
        if s > best:
            best, best_path = s, list(path)
    return best_path, best


def random_hmm(rng, K, M):
    """상태 K 개, 관측 기호 M 개인 무작위 HMM(확률 행렬은 디리클레(1) 표본)."""
    # Dirichlet(1,...,1) = 단체 위의 균등 분포에서 뽑은 확률 벡터. 행 합이 1 이라 확률 행렬이 된다.
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
    # 8장 판과 달리 불가능한 금액(INF)이면 복원 루프를 건너뛴다(last 가 0 이라 무한 루프에 빠지는 것을 막는 방어).
    while x > 0 and best[amount] < INF:
        combo.append(last[x])
        x -= last[x]
    return best, sorted(combo, reverse=True)


def coin_change_bruteforce(coins, amount, counter=None):
    """메모 없이 '마지막 동전'을 모두 시도하는 재귀. 같은 금액을 여러 번 다시 푼다."""
    c = _c(counter)

    # 메모 없는 재귀: 금액 x 마다 동전 종류 수만큼 갈래. 호출 수가 지수적으로 늘어난다(같은 금액을 반복해 푼다).
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
    # log(0) = -inf 를 경고 없이 허용: 확률 0 인 전이를 -inf 점수로 두면 그 길은 max 에서 선택되지 않는다.
    with np.errstate(divide="ignore"):
        return np.log(x)

