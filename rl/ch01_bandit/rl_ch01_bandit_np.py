"""강화학습 1장 NumPy 최소 구현: 베르누이 밴딧과 네 가지 고르기 규칙.

greedy        a_t = argmax_j xbar_j                         (지금까지 평균이 가장 높은 손잡이)
eps_greedy    확률 eps로 아무 손잡이, 아니면 greedy
ucb1          a_t = argmax_j xbar_j + sqrt(2 ln n / n_j)      (Auer, Cesa-Bianchi, Fischer 2002, Fig. 1)
thompson_beta theta_j ~ Beta(1 + 성공_j, 1 + 실패_j), a_t = argmax_j theta_j

여러 반복(replicate)을 한꺼번에 돌리려고 모든 상태를 (S, K) 배열로 둔다. S는 반복 수, K는 손잡이 수.
반복 s의 보상은 np.random.default_rng(seeds[s])로 미리 만든 표에서 꺼낸다. 손잡이 a를 k번째로
당기면 table[s, k, a]가 나온다. 그래서 같은 시드라면 어떤 규칙으로 고르든 "손잡이 a의 첫 번째 결과"가
같다. 규칙 쪽의 무작위(동점 깨기, eps 탐험, 베타 표본)는 run()에 넘긴 rng 하나에서 나온다.

무엇을 재는지: 탐험 규칙별 누적 후회(pseudo regret)와 좋은 손잡이를 고른 비율.
입력: 손잡이 성공확률 p(길이 K), 반복용 시드 목록, 횟수 n. 출력: run()이 선택 기록 (S, n)과 보상 (S, n)을 돌려준다.
탐험 파라미터: eps_greedy의 eps(탐험 확률). UCB1의 탐험 계수는 인자가 아니라 지수 식 안에 sqrt(2)로 고정.
이 장은 상태가 하나인 문제라 할인율 gamma와 벨만 식은 나오지 않는다(2장에서 등장).
보상은 0/1(베르누이)이라는 가정이 UCB1 보너스 식과 베타 사후분포 모두의 전제다.
"""

import numpy as np


class BernoulliBandit:
    def __init__(self, p, seeds, n_max):
        self.p = np.asarray(p, dtype=float)
        self.K = len(self.p)
        self.seeds = list(seeds)
        self.S = len(self.seeds)
        self.n_max = n_max
        # table[s, k, a]: 반복 s에서 손잡이 a를 k번째 당길 때의 보상. 모양 (S, n_max, K), 시드로 미리 고정(공통 난수)
        self.table = np.stack([np.random.default_rng(sd).random((n_max, self.K)) < self.p for sd in self.seeds])
        # 식: Delta_j = p* - p_j  (손잡이 j의 간격, 최적 손잡이는 0)
        self.gaps = self.p.max() - self.p
        self.best = int(np.argmax(self.p))
        self.reset()

    def reset(self):
        # 손잡이별 당긴 횟수. 모양 (S, K). reset 때마다 0으로 돌려 같은 표를 규칙마다 처음부터 다시 쓴다
        self.pulls = np.zeros((self.S, self.K), dtype=np.int64)

    def pull(self, a):
        rows = np.arange(self.S)
        # a는 반복마다 고른 손잡이 (S,). 해당 손잡이의 다음 순번 결과를 꺼낸다
        r = self.table[rows, self.pulls[rows, a], a]
        self.pulls[rows, a] += 1
        return r.astype(float)


def argmax_random_tie(values, u):
    """행마다 최댓값인 칸 중 하나를 고른다. u는 행마다 [0,1) 균등 난수 하나."""
    # 동점 깨기: 최댓값과 같은 칸을 모두 표시(부동소수 비교라 값이 정확히 같을 때만 동점)
    is_max = values == values.max(axis=1, keepdims=True)
    cnt = is_max.sum(axis=1)
    # u*cnt를 내림해 동점 칸 중 몇 번째를 고를지 정한다. 균등. min은 u가 1에 극히 가까울 때의 범위 초과 방지
    pick = np.minimum((u * cnt).astype(np.int64), cnt - 1)
    # 누적합이 pick을 처음 넘는 열 = 최댓값 칸 중 (pick+1)번째
    return np.argmax(np.cumsum(is_max, axis=1) > pick[:, None], axis=1)


def ucb1_index(N, X, n):
    """xbar_j + sqrt(2 ln n / n_j). 한 번도 안 당긴 손잡이(n_j = 0)는 +inf로 둬서 먼저 당기게 한다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # 식: 지수 = xbar_j + sqrt(2 ln n / n_j). 앞은 평균(활용), 뒤는 탐험 보너스. 당길수록(n_j 증가) 보너스가 줄고, 안 당긴 사이 n이 커지면 느리게 커진다
        # N=0일 때 0/0, 나눗셈 경고는 아래 where에서 덮어쓰므로 errstate로 끈다
        idx = X / N + np.sqrt(2.0 * np.log(n) / N)
    # 안 당긴 손잡이는 +inf라 먼저 당겨진다(round_robin을 쓰지 않는 정책에서도 안전)
    return np.where(N == 0, np.inf, idx)


def beta_posterior(N, X, a0=1.0, b0=1.0):
    """Beta(a0, b0)에서 시작해 성공 X번, 실패 N - X번을 본 뒤의 (alpha, beta)."""
    # 식(베르누이-베타 켤레): 사전 Beta(1,1)(균등)에서 성공 X, 실패 N-X를 보면 Beta(1+X, 1+N-X)
    return a0 + X, b0 + (N - X)


class Policy:
    name = "policy"
    # 처음 K번은 각 손잡이를 한 번씩 당겨 X/N의 0으로 나눔을 피하고 UCB1 초기화(Auer)와 맞춘다
    round_robin = True  # 처음 K번은 손잡이 0, 1, ..., K-1을 한 번씩

    def select(self, N, X, n, rng):
        raise NotImplementedError


class Greedy(Policy):
    name = "greedy"

    def select(self, N, X, n, rng):
        # 식: a = argmax_j xbar_j (활용만, 탐험 파라미터 없음). round_robin 이후라 N >= 1
        return argmax_random_tie(X / N, rng.random(N.shape[0]))


class EpsGreedy(Policy):
    def __init__(self, eps):
        self.eps = eps
        self.name = f"eps_greedy({eps})"

    def select(self, N, X, n, rng):
        S, K = N.shape
        greedy = argmax_random_tie(X / N, rng.random(S))
        # 식: 확률 eps로 탐험. 탐험 때는 K개 손잡이 중 균등(좋은 손잡이도 포함)이라 나쁜 손잡이를 고를 확률은 eps*(K-1)/K
        explore = rng.random(S) < self.eps
        rand_arm = rng.integers(0, K, size=S)
        # 탐험이면 무작위 손잡이, 아니면 greedy. 세 난수(동점, 탐험 여부, 무작위 손잡이)를 항상 모두 뽑아 rng 소비량을 일정하게 한다
        return np.where(explore, rand_arm, greedy)


class UCB1(Policy):
    name = "ucb1"

    def select(self, N, X, n, rng):
        # 식: a = argmax_j [xbar_j + sqrt(2 ln n / n_j)], n = 지금까지 당긴 총횟수
        return argmax_random_tie(ucb1_index(N, X, n), rng.random(N.shape[0]))


class ThompsonBeta(Policy):
    name = "thompson"
    # Thompson은 사전 Beta(1,1)에서 바로 시작해 초기 한 바퀴가 필요 없다
    round_robin = False

    def select(self, N, X, n, rng):
        a, b = beta_posterior(N, X)
        # 식: theta_j ~ Beta(1+성공_j, 1+실패_j)를 손잡이마다 한 번씩 뽑아 argmax(사후분포 표본 선택)
        theta = rng.beta(a, b)
        return argmax_random_tie(theta, rng.random(N.shape[0]))


def greedy():
    return Greedy()


def eps_greedy(eps):
    return EpsGreedy(eps)


def ucb1():
    return UCB1()


def thompson_beta():
    return ThompsonBeta()


def run(policy, env, n_steps, rng):
    """정책을 n_steps번 돌린다. 반환: actions (S, n) int, rewards (S, n) float."""
    env.reset()
    S, K = env.S, env.K
    N = np.zeros((S, K))
    X = np.zeros((S, K))
    actions = np.zeros((S, n_steps), dtype=np.int64)
    rewards = np.zeros((S, n_steps))
    rows = np.arange(S)
    for t in range(n_steps):
        if policy.round_robin and t < K:
            a = np.full(S, t, dtype=np.int64)
        else:
            a = policy.select(N, X, t, rng)  # t = 지금까지 당긴 총횟수 (Auer의 n)
        r = env.pull(a)
        # 식: N_a <- N_a + 1, X_a <- X_a + r. 표본 평균 xbar = X/N을 따로 저장하지 않고 합과 횟수로 보관
        N[rows, a] += 1
        X[rows, a] += r
        actions[:, t] = a
        rewards[:, t] = r
    return actions, rewards


def counts_over_time(actions, K):
    """T[s, t, j] = 처음 t+1번 중 손잡이 j를 당긴 횟수."""
    # (S, n, K) 원핫. 메모리 S*n*K에 비례(2,000 x 1,000 x 10 불리언)
    onehot = actions[:, :, None] == np.arange(K)[None, None, :]
    # 시간 축(axis=1)으로 누적해 T[s, t, j]
    return np.cumsum(onehot, axis=1)


def pseudo_regret(actions, gaps):
    """누적 후회 sum_j Delta_j T_j(t)를 t마다. (S, n)."""
    # 식: 후회 = sum_t Delta_{a_t}. 실제 받은 보상이 아니라 기대값 기준 간격을 쓰는 '의사 후회'라 보상 잡음이 빠진다. gaps[actions] 모양 (S, n)
    return np.cumsum(np.asarray(gaps)[actions], axis=1)


def bernoulli_kl(p, q):
    """D(Bern(p) || Bern(q))."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    # 식: D(Bern(p) || Bern(q)) = p ln(p/q) + (1-p) ln((1-p)/(1-q)). p나 q가 0 또는 1이면 정의되지 않으니 0과 1 사이 값만 쓴다
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def lai_robbins_curve(p, n):
    """sum_{j 나쁜 손잡이} Delta_j ln n / D(p_j || p*). 점근 하한의 앞부분 상수만 쓴 곡선."""
    p = np.asarray(p, float)
    ps = p.max()
    bad = p < ps
    # 식: C = sum_{p_j<p*} Delta_j / D(p_j || p*). 하한 곡선은 C ln n (점근 형태, 상수만 반영)
    c = np.sum((ps - p[bad]) / bernoulli_kl(p[bad], ps))
    return c * np.log(np.asarray(n, float)), float(c)


def auer_ucb1_bound(p, n):
    """Auer 등 2002 정리 1: 8 sum_{나쁜 i} ln n / Delta_i + (1 + pi^2/3) sum_j Delta_j."""
    p = np.asarray(p, float)
    d = p.max() - p
    bad = d > 0
    # 식: 8 sum_{Delta_i>0} (ln n / Delta_i) + (1 + pi^2/3) sum_j Delta_j. 느슨한 상한이라 실측 후회보다 훨씬 크다
    return 8 * np.sum(1.0 / d[bad]) * np.log(np.asarray(n, float)) + (1 + np.pi**2 / 3) * d.sum()
