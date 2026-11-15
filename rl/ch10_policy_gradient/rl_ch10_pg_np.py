"""강화학습 10장 NumPy 최소 구현: 정책경사(REINFORCE), 베이스라인, one-step actor-critic.

두 가지 정책을 쓴다. 둘 다 행동이 두 개(0 = 왼쪽, 1 = 오른쪽)라 소프트맥스가 시그모이드 하나로 줄어든다.
  - 복도: 칸을 못 보는 정책. 매개변수 theta 하나, 오른쪽 확률 p = sigmoid(theta).
  - cart-pole: 선형 정책. 오른쪽 확률 p = sigmoid(theta . phi(s)).
시그모이드 정책에서 log pi(a|s)의 기울기는 (a - p) * phi(s) 이다(a는 0 또는 1).
  오른쪽(a=1)을 골랐으면 (1-p) phi, 왼쪽(a=0)을 골랐으면 -p phi.

대응 관계
- 정책경사 정리의 표본 추정: g_hat = sum_t (G_t - b) grad log pi(a_t|s_t). 복도는 corridor_grad_estimate, cart-pole은 cartpole_reinforce.
- 입력: 시드, 판 수, 학습률 alpha(정책), alpha_w(기준선/비평가), 할인율 gamma. 출력: 판별 J(theta)나 판 길이 곡선과 최종 theta.
- 탐험은 정책의 확률성이 맡는다(epsilon 없음). Sarsa 비교(corridor_sarsa_aliased)만 epsilon-탐욕을 쓴다.
- 결과 숫자는 experiments.py의 E0~E6(results/ch10.json)로 이어진다.
"""

from dataclasses import dataclass, field

import numpy as np

import rl_ch10_cartpole as cp
import rl_ch10_corridor as cor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_policy(theta, phi):
    """행동 두 개짜리 소프트맥스 정책의 [왼쪽, 오른쪽] 확률. 선호도 h = [0, theta.phi]."""
    # 식: pi(오른쪽|s) = sigmoid(theta . phi(s)). 두 행동의 선호도를 [0, theta.phi]로 두면 softmax가 이 시그모이드와 같다
    p = sigmoid(float(np.dot(theta, phi)))
    return np.array([1.0 - p, p])


def grad_log_pi(theta, phi, a):
    """d/dtheta log pi(a | s) = (a - p) * phi."""
    p = sigmoid(float(np.dot(theta, phi)))
    # 식: grad log pi(a|s) = (a - p) phi. a=1이면 (1-p)phi, a=0이면 -p phi (머리말 유도)
    return (a - p) * np.asarray(phi, dtype=float)


def returns(rewards, gamma):
    """G_t = r_{t+1} + gamma r_{t+2} + ... 를 뒤에서부터 계산."""
    G = np.zeros(len(rewards))
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        # 식: G_t = r_{t+1} + gamma G_{t+1}. 끝에서 앞으로 한 번에 누적(O(T)). gamma는 할인율
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
    # cap: 한 판 최대 걸음 수. 정책이 거의 한쪽만 고르면 판이 끝나지 않을 수 있어서 둔 안전 상한
    for _ in range(cap):
        # 행동 표집: 난수가 p보다 작으면 오른쪽(1). 정책이 확률적이라 이 자체가 탐험이다(epsilon 같은 별도 탐험 매개변수 없음)
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
    # gamma = 1(할인 없음): 복도는 끝나는 과제라 할인 없이도 G가 유한. 값은 -(남은 걸음 수)
    G = returns(ep.rewards, 1.0)
    a = np.asarray(ep.actions, dtype=float)
    if baseline == "none":
        b = np.zeros_like(G)
    elif baseline == "const":
        b = np.full_like(G, cor.exact_J(theta))
    elif baseline == "value":
        # 정확한 V_pi를 선형방정식으로 풀어 쓴다(이상적 기준선). 실제 학습에서는 알 수 없는 값이라 분산 감소의 상한을 보는 용도
        V = cor.exact_v(p)
        b = V[np.asarray(ep.states)]
    else:
        raise ValueError(baseline)
    # 식: g_hat = sum_t (G_t - b(s_t)) (a_t - p). 정책경사 정리의 한 판 표본 추정. 기준선 b는 행동에 의존하지 않아 기댓값을 바꾸지 않고 분산만 줄인다
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
            # 학습 기준선: w는 모든 칸에 공통인 숫자 하나. 보정된 이득 G_t - w
            delta = G - w
            # 식: w <- w + alpha_w sum_t (G_t - w). 평균 리턴 쪽으로 w를 당기는 확률적 경사 갱신(alpha_w = w의 학습률)
            w += alpha_w * float(np.sum(delta))
        else:
            delta = G
        # 식: theta <- theta + alpha g_hat. alpha = 정책 학습률. J를 올리는 방향(경사 상승)이라 +
        theta += alpha * float(np.sum(delta * (a - p)))
        # theta가 커지면 p가 0이나 1에 붙어 exact_J가 -inf가 된다. 수치 안정을 위한 한계(모형의 선택)
        theta = float(np.clip(theta, -20, 20))
        thetas.append(theta)
        Js.append(cor.exact_J(theta))
    return np.array(thetas), np.array(Js), steps


def corridor_sarsa_aliased(n_episodes, seed, eps=0.1, alpha=0.01, cap=1000):
    """가치 기반 비교: 칸을 못 보는 Q(오른쪽), Q(왼쪽) 두 숫자로 epsilon-탐욕 Sarsa(할인 없음).

    정책은 늘 "Q가 큰 쪽을 1 - eps/2 확률로" 고르므로, 고를 수 있는 오른쪽 확률은 eps/2나 1 - eps/2 둘뿐이다.
    """
    rng = np.random.default_rng(seed)
    # 칸을 못 보는 가치 기반 비교: 칸마다가 아니라 행동마다 Q 하나뿐(상태 별칭, aliasing). 이 때문에 확률적 최적 정책을 표현하지 못한다
    Q = np.zeros(2)

    def choose():
        # epsilon-탐욕 탐험: 확률 eps로 무작위 행동. 무작위가 두 행동 중 하나라 오른쪽 확률은 eps/2 또는 1-eps/2
        if rng.random() < eps:
            return int(rng.integers(2))
        # Q가 같으면 무작위로 깨서 한쪽에 치우친 동률 처리를 피한다
        return int(np.argmax(Q)) if Q[0] != Q[1] else int(rng.integers(2))

    lengths = []
    for _ in range(n_episodes):
        s, a = 0, choose()
        for t in range(cap):
            s2, r, done = cor.step(s, a)
            if done:
                # 끝난 걸음: 목표 = r (다음 Q 없음)
                Q[a] += alpha * (r - Q[a])
                break
            a2 = choose()
            # 식(Sarsa, gamma=1): Q(a) <- Q(a) + alpha [r + Q(a') - Q(a)]. 실제로 다음에 고른 a2를 쓰는 온정책
            Q[a] += alpha * (r + Q[a2] - Q[a])
            s, a = s2, a2
        lengths.append(t + 1)
    p_right = 1 - eps / 2 if Q[1] > Q[0] else eps / 2
    return Q, p_right, np.array(lengths)


# ---------------------------------------------------------------- cart-pole

def phi_policy(s):
    """정책 특징: 네 상태값을 대략 [-1, 1]로 맞춘 값 + 상수 1."""
    # 특징 정규화: 위치/2.4(실패 경계), 속도/3, 각도/0.21(약 12도의 라디안), 각속도/3. 마지막 1.0은 편향 항. 속도 나눔값 3.0의 근거는 코드에 없다(확인 필요)
    return np.array([s[0] / 2.4, s[1] / 3.0, s[2] / 0.21, s[3] / 3.0, 1.0])


def phi_value(s):
    """가치 특징: 정책 특징 + 네 상태값의 제곱(막대가 가운데서 멀수록 남은 걸음이 줄어드는 모양을 담으려고)."""
    f = phi_policy(s)
    return np.concatenate([f, f[:4] ** 2])


def run_cartpole_episode(env, theta, rng):
    """정책 theta로 한 판. (특징 목록, 행동, 보상, 상태, 쓰러졌나)."""
    s = env.reset()
    # S는 행동 직전 상태들(행동과 같은 길이로 맞춘다). 마지막 상태는 행동할 필요가 없어 S에 넣지 않는다
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
    # 환경 난수(seed)와 행동 난수(10_000+seed)를 분리. 같은 seed면 정확히 재현
    rng = np.random.default_rng(10_000 + seed)
    # 정책 매개변수 5개(상태 4 + 편향). 0에서 시작하면 p = 0.5의 균등 무작위 정책
    theta = np.zeros(5)
    w = np.zeros(9)
    lens = np.zeros(n_episodes, dtype=int)
    for i in range(n_episodes):
        S, A, R, _ = run_cartpole_episode(env, theta, rng)
        # gamma = 0.99(약한 할인, 모형의 선택)
        G = returns(R, gamma)
        Phi = np.array([phi_policy(s) for s in S])
        p = sigmoid(Phi @ theta)
        if baseline == "value":
            Psi = np.array([phi_value(s) for s in S])
            # 기준선 V_w(s) = w . phi_value(s)(선형). 보정된 이득 G_t - V_w(s_t)
            delta = G - Psi @ w
            # 식: w <- w + alpha_w sum_t (G_t - V_w(s_t)) psi(s_t). 리턴에 대한 최소제곱의 경사 한 걸음
            w += alpha_w * (delta @ Psi)
        else:
            delta = G
        # 식: theta <- theta + alpha sum_t delta_t (a_t - p_t) phi(s_t). gamma^t 곱은 뺀 구현(독스트링). (T,)@(T,5) -> (5,)
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
            # 쓰러짐은 진짜 종료라 V(s')=0. 시간 상한 잘림은 종료가 아니므로 V(s') 그대로(부트스트랩을 끊지 않는다)
            v_next = 0.0 if fell else float(w @ phi_value(s2))
            # 식: TD 오차 delta = r + gamma V(s') - V(s)
            delta = r + gamma * v_next - float(w @ psi)
            # 비평가 갱신: w <- w + alpha_w delta psi(s) (준경사 TD(0))
            w += alpha_w * delta * psi
            # 배우 갱신: theta <- theta + alpha delta grad log pi(a|s). 기준선이 곧 V이고 delta가 어드밴티지 추정(1-step). gamma^t 곱은 뺀 구현
            theta += alpha * delta * (a - p) * f
            n += 1
            if fell or trunc:
                break
            s = s2
        lens[i] = n
    return lens, theta
