"""강화학습 9장 NumPy 최소 구현: 신경망 Q-러닝(DQN)과 두 안정화 장치, 그리고 최댓값의 치우침.

- MLP: ReLU 은닉층 두 개, 손으로 쓴 역전파, 손으로 쓴 Adam
- ReplayBuffer: 용량을 넘으면 가장 오래된 전이부터 덮어쓴다. 균등 추출(use_replay=True) 또는
  방금 겪은 연속 전이 batch개(use_replay=False, "재생 없음")를 꺼낸다. 두 경우 모두 걸음마다
  같은 크기의 미니배치로 한 번 고치므로, 다른 것은 "어떤 전이를 꺼내느냐" 하나뿐이다.
- dqn(): 타깃 네트워크(use_target, C번 고칠 때마다 복사)와 Double 목표(double)를 켜고 끈다.
- max_bias_*: 신경망 없이 최댓값이 값을 부풀리는 크기를 잰다.

손실은 평균제곱오차 0.5 * (y - Q(s,a))^2 를 그대로 썼다. Nature 2015 판은 TD 오차를 [-1, 1]로
자르고 보상도 잘랐지만, 이 장은 장치 두 개(재생, 타깃)와 Double만 켜고 끄려고 그 자르기는 넣지 않았다.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import rl_ch09_cartpole as cp

SCALE = np.array([2.4, 3.0, 0.21, 3.0])  # 상태 네 값을 대략 [-1, 1]로 (10장과 같은 눈금)


def feat(s):
    return np.asarray(s, dtype=float) / SCALE


# ------------------------------------------------------------------ 신경망

class MLP:
    """sizes = [4, 64, 64, 2]. 은닉층은 ReLU, 마지막 층은 그대로(행동마다 Q 하나)."""

    def __init__(self, sizes, seed):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            lim = math.sqrt(6.0 / n_in)  # He 균등 초기화
            self.W.append(rng.uniform(-lim, lim, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def params(self):
        return self.W + self.b

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def clone(self):
        m = MLP.__new__(MLP)
        m.copy_from(self)
        return m

    def forward(self, X, keep=False):
        h = np.asarray(X, dtype=float)
        acts = [h]
        n = len(self.W)
        for i in range(n):
            z = h @ self.W[i] + self.b[i]
            h = np.maximum(z, 0.0) if i < n - 1 else z
            acts.append(h)
        return (h, acts) if keep else h

    def grad(self, X, A, y):
        """손실 L = mean_j 0.5 * (y_j - Q(s_j, a_j))^2 의 기울기. y는 상수로 취급한다."""
        Q, acts = self.forward(X, keep=True)
        B = len(A)
        idx = np.arange(B)
        err = Q[idx, A] - y                      # 예측 - 목표
        loss = 0.5 * float(np.mean(err ** 2))
        dZ = np.zeros_like(Q)
        dZ[idx, A] = err / B                     # 고른 행동 칸에만 기울기가 흐른다
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ dZ
            gb[i] = dZ.sum(axis=0)
            if i > 0:
                dZ = (dZ @ self.W[i].T) * (acts[i] > 0)   # ReLU의 기울기: 켜진 곳만 통과
        return loss, gW + gb, Q


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)   # 제자리에서 고친다


# ------------------------------------------------------------------ 재생 버퍼

class ReplayBuffer:
    """고정 크기 원형 버퍼. 꽉 차면 가장 오래된 칸부터 덮어쓴다."""

    def __init__(self, cap, seed, dim=4):
        self.cap = cap
        self.rng = np.random.default_rng(seed)
        self.S = np.zeros((cap, dim))
        self.A = np.zeros(cap, dtype=np.int64)
        self.R = np.zeros(cap)
        self.S2 = np.zeros((cap, dim))
        self.F = np.zeros(cap)          # 1이면 쓰러짐(진짜 끝). 500걸음에서 잘린 판은 0
        self.ptr = 0
        self.n = 0
        self.t_added = np.full(cap, -1, dtype=np.int64)   # 몇 번째로 들어온 전이인지(테스트용)
        self.count = 0

    def add(self, s, a, r, s2, fell):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.F[i] = s, a, r, s2, float(fell)
        self.t_added[i] = self.count
        self.count += 1
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample_uniform(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)
        return self._get(idx)

    def latest(self, batch):
        idx = (self.ptr - 1 - np.arange(batch)) % self.cap   # 방금 들어온 것부터 거꾸로 batch개
        return self._get(idx)

    def _get(self, idx):
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.F[idx]

    def nbytes(self):
        return int(self.S.nbytes + self.A.nbytes + self.R.nbytes + self.S2.nbytes + self.F.nbytes)


def td_target(net, tgt, R, S2, F, gamma, double):
    """y = r + gamma * (다음 상태의 점수). 쓰러진 전이는 y = r.
    단일: 타깃망이 고르고 타깃망이 매긴다.  Double: 지금 망이 고르고 타깃망이 매긴다."""
    Q2_t = tgt.forward(feat(S2))
    if double:
        a_star = np.argmax(net.forward(feat(S2)), axis=1)
        nxt = Q2_t[np.arange(len(R)), a_star]
    else:
        nxt = Q2_t.max(axis=1)
    return R + gamma * (1.0 - F) * nxt


# ------------------------------------------------------------------ DQN 학습

@dataclass
class Config:
    steps: int = 50_000
    hidden: int = 64
    lr: float = 1e-3
    batch: int = 32
    buffer: int = 50_000
    gamma: float = 0.99
    C: int = 500               # 타깃망을 이만큼 고칠 때마다 복사
    learn_start: int = 1_000   # 이 걸음까지는 무작위 행동만, 고치지 않음
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_steps: int = 10_000
    eval_every: int = 2_500
    eval_starts: int = 3
    cap: int = 500
    q_limit: float = 1e4


@dataclass
class Run:
    ep_len: list = field(default_factory=list)
    ep_end_step: list = field(default_factory=list)
    loss: list = field(default_factory=list)        # eval_every마다 그 구간 평균 손실
    eval_step: list = field(default_factory=list)
    eval_q0: list = field(default_factory=list)     # 시작 상태에서 max_a Q (지금 망)
    eval_g0: list = field(default_factory=list)     # 탐욕 정책으로 실제로 받은 할인 리턴
    eval_len: list = field(default_factory=list)
    diverged_at: int | None = None
    max_abs_q: float = 0.0
    seconds: float = 0.0
    updates: int = 0
    target_copies: int = 0


def greedy_episode(net, seed, cap, gamma):
    env = cp.CartPole(seed, cap)
    s = env.reset()
    q0 = float(net.forward(feat(s)[None])[0].max())
    g, disc, n = 0.0, 1.0, 0
    while True:
        a = int(np.argmax(net.forward(feat(s)[None])[0]))
        s, r, fell, trunc = env.step(a)
        g += disc * r
        disc *= gamma
        n += 1
        if fell or trunc:
            return q0, g, n


def dqn(seed, use_replay=True, use_target=True, double=False, cfg=None, log=None):
    """log: 함수(row) 를 주면 전이마다 공통 스키마로 넘긴다."""
    import time
    cfg = cfg or Config()
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = cp.CartPole(seed, cfg.cap)
    net = MLP([4, cfg.hidden, cfg.hidden, 2], seed)
    tgt = net.clone() if use_target else net
    opt = Adam(net.params(), cfg.lr)
    buf = ReplayBuffer(cfg.buffer, seed + 10_000)
    run = Run()
    s = env.reset()
    ep, t_ep, losses = 0, 0, []
    for step in range(1, cfg.steps + 1):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_steps)
        if step <= cfg.learn_start or rng.random() < eps:
            a = int(rng.integers(2))
        else:
            a = int(np.argmax(net.forward(feat(s)[None])[0]))
        s2, r, fell, trunc = env.step(a)
        buf.add(s, a, r, s2, fell)
        if log is not None:
            log({"seed": seed, "episode": ep, "t": t_ep, "s": [round(float(x), 4) for x in s], "a": a, "r": r,
                 "s_next": [round(float(x), 4) for x in s2], "done": bool(fell or trunc),
                 "info": {"step": step, "fell": bool(fell), "truncated": bool(trunc), "eps": round(eps, 4)}})
        s = s2
        t_ep += 1
        if fell or trunc:
            run.ep_len.append(t_ep)
            run.ep_end_step.append(step)
            ep += 1
            t_ep = 0
            s = env.reset()
        if step > cfg.learn_start:
            S, A, R, S2, F = buf.sample_uniform(cfg.batch) if use_replay else buf.latest(cfg.batch)
            y = td_target(net, tgt, R, S2, F, cfg.gamma, double)
            loss, grads, Q = net.grad(feat(S), A, y)
            opt.step(net.params(), grads)
            run.updates += 1
            losses.append(loss)
            mq = float(np.max(np.abs(Q)))
            run.max_abs_q = max(run.max_abs_q, mq)
            if not np.isfinite(loss) or mq > cfg.q_limit:
                run.diverged_at = step
                break
            if use_target and run.updates % cfg.C == 0:
                tgt.copy_from(net)
                run.target_copies += 1
        if step % cfg.eval_every == 0:
            q0s, g0s, lens = [], [], []
            for k in range(cfg.eval_starts):
                q0, g0, n = greedy_episode(net, 900_000 + k, cfg.cap, cfg.gamma)
                q0s.append(q0)
                g0s.append(g0)
                lens.append(n)
            run.eval_step.append(step)
            run.eval_q0.append(float(np.mean(q0s)))
            run.eval_g0.append(float(np.mean(g0s)))
            run.eval_len.append(float(np.mean(lens)))
            run.loss.append(float(np.mean(losses)) if losses else None)
            losses = []
    run.seconds = time.time() - t0
    run.net = net
    return run


# ------------------------------------------------------------------ 최댓값의 치우침 (신경망 없이)

def coin_max_exact(K):
    """참값 0, 추정값이 +1 또는 -1 반반인 동전 K개. 가장 큰 값의 평균을 2^K 경우를 다 세어 구한다."""
    from itertools import product
    vals = [max(c) for c in product((-1, 1), repeat=K)]
    return sum(vals) / len(vals)


def expected_max_normal(K, lo=-12.0, hi=12.0, n=200_001):
    """표준정규 K개 중 최댓값의 기댓값 = 적분 x * K * phi(x) * Phi(x)^(K-1) dx (사다리꼴 수치적분)."""
    x = np.linspace(lo, hi, n)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return float(np.trapezoid(x * K * phi * Phi ** (K - 1), x))


def max_bias_experiment(K, n, seed, mu=None):
    """참값 mu(기본 모두 0), 잡음 N(0,1). 한 벌로 고르고 매기기(단일) vs 두 벌로 나누기(Double).
    반환: 단일 평균, Double 평균, 참 최댓값, Double <= 단일 인 비율."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(K) if mu is None else np.asarray(mu, dtype=float)
    A = mu + rng.standard_normal((n, K))
    B = mu + rng.standard_normal((n, K))
    single = A.max(axis=1)
    double = B[np.arange(n), A.argmax(axis=1)]
    return {"single": float(single.mean()), "double": float(double.mean()), "true_max": float(mu.max()),
            "single_se": float(single.std(ddof=1) / math.sqrt(n)), "double_se": float(double.std(ddof=1) / math.sqrt(n)),
            "frac_double_le_single": float(np.mean(double <= single))}
