"""강화학습 1장 NumPy 최소 구현: 베르누이 밴딧과 네 가지 고르기 규칙.

greedy        a_t = argmax_j xbar_j                         (지금까지 평균이 가장 높은 손잡이)
eps_greedy    확률 eps로 아무 손잡이, 아니면 greedy
ucb1          a_t = argmax_j xbar_j + sqrt(2 ln n / n_j)      (Auer, Cesa-Bianchi, Fischer 2002, Fig. 1)
thompson_beta theta_j ~ Beta(1 + 성공_j, 1 + 실패_j), a_t = argmax_j theta_j

여러 반복(replicate)을 한꺼번에 돌리려고 모든 상태를 (S, K) 배열로 둔다. S는 반복 수, K는 손잡이 수.
반복 s의 보상은 np.random.default_rng(seeds[s])로 미리 만든 표에서 꺼낸다. 손잡이 a를 k번째로
당기면 table[s, k, a]가 나온다. 그래서 같은 시드라면 어떤 규칙으로 고르든 "손잡이 a의 첫 번째 결과"가
같다. 규칙 쪽의 무작위(동점 깨기, eps 탐험, 베타 표본)는 run()에 넘긴 rng 하나에서 나온다.
"""

import numpy as np


class BernoulliBandit:
    def __init__(self, p, seeds, n_max):
        self.p = np.asarray(p, dtype=float)
        self.K = len(self.p)
        self.seeds = list(seeds)
        self.S = len(self.seeds)
        self.n_max = n_max
        self.table = np.stack([np.random.default_rng(sd).random((n_max, self.K)) < self.p for sd in self.seeds])
        self.gaps = self.p.max() - self.p
        self.best = int(np.argmax(self.p))
        self.reset()

    def reset(self):
        self.pulls = np.zeros((self.S, self.K), dtype=np.int64)

    def pull(self, a):
        rows = np.arange(self.S)
        r = self.table[rows, self.pulls[rows, a], a]
        self.pulls[rows, a] += 1
        return r.astype(float)


def argmax_random_tie(values, u):
    """행마다 최댓값인 칸 중 하나를 고른다. u는 행마다 [0,1) 균등 난수 하나."""
    is_max = values == values.max(axis=1, keepdims=True)
    cnt = is_max.sum(axis=1)
    pick = np.minimum((u * cnt).astype(np.int64), cnt - 1)
    return np.argmax(np.cumsum(is_max, axis=1) > pick[:, None], axis=1)


def ucb1_index(N, X, n):
    """xbar_j + sqrt(2 ln n / n_j). 한 번도 안 당긴 손잡이(n_j = 0)는 +inf로 둬서 먼저 당기게 한다."""
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = X / N + np.sqrt(2.0 * np.log(n) / N)
    return np.where(N == 0, np.inf, idx)


def beta_posterior(N, X, a0=1.0, b0=1.0):
    """Beta(a0, b0)에서 시작해 성공 X번, 실패 N - X번을 본 뒤의 (alpha, beta)."""
    return a0 + X, b0 + (N - X)


class Policy:
    name = "policy"
    round_robin = True  # 처음 K번은 손잡이 0, 1, ..., K-1을 한 번씩

    def select(self, N, X, n, rng):
        raise NotImplementedError


class Greedy(Policy):
    name = "greedy"

    def select(self, N, X, n, rng):
        return argmax_random_tie(X / N, rng.random(N.shape[0]))


class EpsGreedy(Policy):
    def __init__(self, eps):
        self.eps = eps
        self.name = f"eps_greedy({eps})"

    def select(self, N, X, n, rng):
        S, K = N.shape
        greedy = argmax_random_tie(X / N, rng.random(S))
        explore = rng.random(S) < self.eps
        rand_arm = rng.integers(0, K, size=S)
        return np.where(explore, rand_arm, greedy)


class UCB1(Policy):
    name = "ucb1"

    def select(self, N, X, n, rng):
        return argmax_random_tie(ucb1_index(N, X, n), rng.random(N.shape[0]))


class ThompsonBeta(Policy):
    name = "thompson"
    round_robin = False

    def select(self, N, X, n, rng):
        a, b = beta_posterior(N, X)
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
        N[rows, a] += 1
        X[rows, a] += r
        actions[:, t] = a
        rewards[:, t] = r
    return actions, rewards


def counts_over_time(actions, K):
    """T[s, t, j] = 처음 t+1번 중 손잡이 j를 당긴 횟수."""
    onehot = actions[:, :, None] == np.arange(K)[None, None, :]
    return np.cumsum(onehot, axis=1)


def pseudo_regret(actions, gaps):
    """누적 후회 sum_j Delta_j T_j(t)를 t마다. (S, n)."""
    return np.cumsum(np.asarray(gaps)[actions], axis=1)


def bernoulli_kl(p, q):
    """D(Bern(p) || Bern(q))."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def lai_robbins_curve(p, n):
    """sum_{j 나쁜 손잡이} Delta_j ln n / D(p_j || p*). 점근 하한의 앞부분 상수만 쓴 곡선."""
    p = np.asarray(p, float)
    ps = p.max()
    bad = p < ps
    c = np.sum((ps - p[bad]) / bernoulli_kl(p[bad], ps))
    return c * np.log(np.asarray(n, float)), float(c)


def auer_ucb1_bound(p, n):
    """Auer 등 2002 정리 1: 8 sum_{나쁜 i} ln n / Delta_i + (1 + pi^2/3) sum_j Delta_j."""
    p = np.asarray(p, float)
    d = p.max() - p
    bad = d > 0
    return 8 * np.sum(1.0 / d[bad]) * np.log(np.asarray(n, float)) + (1 + np.pi**2 / 3) * d.sum()
