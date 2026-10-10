"""강화학습 10장 NumPy 최소 구현: 정책경사(REINFORCE), 베이스라인, one-step actor-critic.

두 가지 정책을 쓴다. 둘 다 행동이 두 개(0 = 왼쪽, 1 = 오른쪽)라 소프트맥스가 시그모이드 하나로 줄어든다.
  - 복도: 칸을 못 보는 정책. 매개변수 theta 하나, 오른쪽 확률 p = sigmoid(theta).
  - cart-pole: 선형 정책. 오른쪽 확률 p = sigmoid(theta . phi(s)).
시그모이드 정책에서 log pi(a|s)의 기울기는 (a - p) * phi(s) 이다(a는 0 또는 1).
  오른쪽(a=1)을 골랐으면 (1-p) phi, 왼쪽(a=0)을 골랐으면 -p phi.
"""

from dataclasses import dataclass, field

import numpy as np

import rl_ch10_cartpole as cp
import rl_ch10_corridor as cor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_policy(theta, phi):
    """행동 두 개짜리 소프트맥스 정책의 [왼쪽, 오른쪽] 확률. 선호도 h = [0, theta.phi]."""
    p = sigmoid(float(np.dot(theta, phi)))
    return np.array([1.0 - p, p])


def grad_log_pi(theta, phi, a):
    """d/dtheta log pi(a | s) = (a - p) * phi."""
    p = sigmoid(float(np.dot(theta, phi)))
    return (a - p) * np.asarray(phi, dtype=float)


def returns(rewards, gamma):
    """G_t = r_{t+1} + gamma r_{t+2} + ... 를 뒤에서부터 계산."""
    G = np.zeros(len(rewards))
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        G[t] = g
    return G


# ---------------------------------------------------------------- 복도

@dataclass
class Episode:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    done: bool = False


def corridor_episode(theta, rng, cap=1000):
    """칸을 못 보는 정책(오른쪽 확률 sigmoid(theta))으로 A에서 출발해 한 판을 걷는다."""
    p = sigmoid(theta)
    ep = Episode()
    s = 0
    for _ in range(cap):
        a = int(rng.random() < p)
        s2, r, done = cor.step(s, a)
        ep.states.append(s)
        ep.actions.append(a)
        ep.rewards.append(r)
        if done:
            ep.done = True
            break
        s = s2
    return ep


def corridor_grad_estimate(ep, theta, baseline="none"):
    """한 판으로 만든 REINFORCE 기울기 추정값(스칼라). 할인 없음(gamma = 1).

    g_hat = sum_t (G_t - b(s_t)) * (a_t - p)
    baseline: "none" b = 0
              "const" b = J(theta) 정확값 하나를 모든 칸에 똑같이 뺀다
              "value" b = V_pi(s_t) 정확값. 정책은 칸을 못 보지만, 학습 중에 빼는 숫자는 칸을 봐도 된다
                      (행동에만 기대지 않으면 평균이 그대로이므로). 분산을 얼마나 줄일 수 있는지 보는 이상적인 경우.
    """
    p = sigmoid(theta)
    G = returns(ep.rewards, 1.0)
    a = np.asarray(ep.actions, dtype=float)
    if baseline == "none":
        b = np.zeros_like(G)
    elif baseline == "const":
        b = np.full_like(G, cor.exact_J(theta))
    elif baseline == "value":
        V = cor.exact_v(p)
        b = V[np.asarray(ep.states)]
    else:
        raise ValueError(baseline)
    return float(np.sum((G - b) * (a - p)))


def corridor_reinforce(theta0, alpha, n_episodes, seed, baseline="none", alpha_w=0.0, cap=1000):
    """복도에서 REINFORCE로 theta를 고친다. 판마다 정확한 J(theta)를 기록한다.

    baseline: "none" 또는 "learned"(배운 숫자 하나 w를 모든 칸에 똑같이 뺀다. w도 판의 리턴으로 고친다).
    """
    rng = np.random.default_rng(seed)
    theta, w = float(theta0), 0.0
    Js, thetas, steps = [], [], 0
    for _ in range(n_episodes):
        ep = corridor_episode(theta, rng, cap)
        steps += len(ep.actions)
        p = sigmoid(theta)
        G = returns(ep.rewards, 1.0)
        a = np.asarray(ep.actions, dtype=float)
        if baseline == "learned":
            delta = G - w
            w += alpha_w * float(np.sum(delta))
        else:
            delta = G
        theta += alpha * float(np.sum(delta * (a - p)))
        theta = float(np.clip(theta, -20, 20))
        thetas.append(theta)
        Js.append(cor.exact_J(theta))
    return np.array(thetas), np.array(Js), steps


def corridor_sarsa_aliased(n_episodes, seed, eps=0.1, alpha=0.01, cap=1000):
    """가치 기반 비교: 칸을 못 보는 Q(오른쪽), Q(왼쪽) 두 숫자로 epsilon-탐욕 Sarsa(할인 없음).

    정책은 늘 "Q가 큰 쪽을 1 - eps/2 확률로" 고르므로, 고를 수 있는 오른쪽 확률은 eps/2나 1 - eps/2 둘뿐이다.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros(2)

    def choose():
        if rng.random() < eps:
            return int(rng.integers(2))
        return int(np.argmax(Q)) if Q[0] != Q[1] else int(rng.integers(2))

    lengths = []
    for _ in range(n_episodes):
        s, a = 0, choose()
        for t in range(cap):
            s2, r, done = cor.step(s, a)
            if done:
                Q[a] += alpha * (r - Q[a])
                break
            a2 = choose()
            Q[a] += alpha * (r + Q[a2] - Q[a])
            s, a = s2, a2
        lengths.append(t + 1)
    p_right = 1 - eps / 2 if Q[1] > Q[0] else eps / 2
    return Q, p_right, np.array(lengths)


# ---------------------------------------------------------------- cart-pole

def phi_policy(s):
    """정책 특징: 네 상태값을 대략 [-1, 1]로 맞춘 값 + 상수 1."""
    return np.array([s[0] / 2.4, s[1] / 3.0, s[2] / 0.21, s[3] / 3.0, 1.0])


def phi_value(s):
    """가치 특징: 정책 특징 + 네 상태값의 제곱(막대가 가운데서 멀수록 남은 걸음이 줄어드는 모양을 담으려고)."""
    f = phi_policy(s)
    return np.concatenate([f, f[:4] ** 2])


def run_cartpole_episode(env, theta, rng):
    """정책 theta로 한 판. (특징 목록, 행동, 보상, 상태, 쓰러졌나)."""
    s = env.reset()
    S, A, R = [s], [], []
    fell = False
    while True:
        p = sigmoid(float(theta @ phi_policy(s)))
        a = int(rng.random() < p)
        s, r, fell, trunc = env.step(a)
        A.append(a)
        R.append(r)
        if fell or trunc:
            break
        S.append(s)
    return np.array(S), np.array(A), np.array(R), fell


def cartpole_reinforce(seed, n_episodes, alpha, gamma=0.99, baseline="none", alpha_w=0.0, cap=cp.CAP):
    """REINFORCE(판이 끝나면 한 번에 고침). baseline: "none" 또는 "value"(선형 V_w(s), 판의 리턴으로 학습).

    교과서 의사코드에는 gamma^t 곱이 있지만, 여기서는 흔히 하듯 빼고 G_t를 그대로 쓴다(본문에 적음).
    """
    env = cp.CartPole(seed, cap)
    rng = np.random.default_rng(10_000 + seed)
    theta = np.zeros(5)
    w = np.zeros(9)
    lens = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        S, A, R, _ = run_cartpole_episode(env, theta, rng)
        G = returns(R, gamma)
        Phi = np.array([phi_policy(s) for s in S])
        p = sigmoid(Phi @ theta)
        if baseline == "value":
            Psi = np.array([phi_value(s) for s in S])
            delta = G - Psi @ w
            w += alpha_w * (delta @ Psi)
        else:
            delta = G
        theta += alpha * ((delta * (A - p)) @ Phi)
        lens[i] = len(A)
    return lens, theta


def cartpole_actor_critic(seed, n_episodes, alpha, gamma=0.99, alpha_w=0.0, cap=cp.CAP):
    """one-step actor-critic. 걸음마다 TD 오차 delta = r + gamma V(s') - V(s)로 비평가(V)와 배우(정책)를 함께 고친다.

    쓰러져서 끝난 걸음은 V(s') = 0. 상한에서 잘린 걸음은 끝난 게 아니라서 V(s')를 그대로 쓴다.
    """
    env = cp.CartPole(seed, cap)
    rng = np.random.default_rng(10_000 + seed)
    theta = np.zeros(5)
    w = np.zeros(9)
    lens = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        s = env.reset()
        n = 0
        while True:
            f = phi_policy(s)
            psi = phi_value(s)
            p = sigmoid(float(theta @ f))
            a = int(rng.random() < p)
            s2, r, fell, trunc = env.step(a)
            v_next = 0.0 if fell else float(w @ phi_value(s2))
            delta = r + gamma * v_next - float(w @ psi)
            w += alpha_w * delta * psi
            theta += alpha * delta * (a - p) * f
            n += 1
            if fell or trunc:
                break
            s = s2
        lens[i] = n
    return lens, theta
