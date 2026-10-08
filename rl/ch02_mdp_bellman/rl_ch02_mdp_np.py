"""강화학습 2장 NumPy 최소 구현: 벨만 백업 두 개와 롤아웃.

V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [r(s,a,s') + gamma V^pi(s')]      (기대방정식)
V*(s)   = max_a      sum_s' P(s'|s,a) [r(s,a,s') + gamma V*(s')]            (최적방정식)

방정식을 "어떻게 푸는가"는 3장(정책 평가)과 4장(가치 반복)의 주제다. 여기서는 백업을 오차가
1e-13 아래로 떨어질 때까지 되풀이하는 가장 단순한 방법으로 해를 얻어 쓰기만 한다.
"""

import numpy as np


def q_from_v(V, P, R, gamma):
    """Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]."""
    return np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_expectation_backup(V, pi, P, R, gamma):
    return (pi * q_from_v(V, P, R, gamma)).sum(axis=1)


def bellman_optimality_backup(V, P, R, gamma):
    return q_from_v(V, P, R, gamma).max(axis=1)


def iterate(backup, S, tol=1e-13, max_iter=100_000):
    V = np.zeros(S)
    for k in range(1, max_iter + 1):
        V2 = backup(V)
        if np.max(np.abs(V2 - V)) < tol:
            return V2, k
        V = V2
    raise RuntimeError("수렴하지 않았어요")


def evaluate(pi, P, R, gamma):
    return iterate(lambda V: bellman_expectation_backup(V, pi, P, R, gamma), P.shape[0])


def optimal_values(P, R, gamma):
    return iterate(lambda V: bellman_optimality_backup(V, P, R, gamma), P.shape[0])


def argmax_policy(Q, tol=1e-9):
    """Q가 가장 큰 행동들에 확률을 똑같이 나눈다(동점이면 균등)."""
    best = Q >= Q.max(axis=1, keepdims=True) - tol
    return best / best.sum(axis=1, keepdims=True)


def myopic_policy(P, R):
    """1장 밴딧식: 지금 한 걸음의 기대 보상 E[r | s, a]만 보고 고른다."""
    return argmax_policy(np.einsum("sap,sap->sa", P, R))


def optimal_policy(P, R, gamma):
    V, _ = optimal_values(P, R, gamma)
    return argmax_policy(q_from_v(V, P, R, gamma))


def rollout(pi, P, R, gamma, s0, terminals, rng, max_steps=500, log=None, seed=None, episode=None):
    """한 에피소드를 돌리고 할인 리턴, 걸음 수, 방문한 상태 목록을 돌려준다."""
    s, G, disc, visits = s0, 0.0, 1.0, [s0]
    for t in range(max_steps):
        a = int(rng.choice(pi.shape[1], p=pi[s]))
        s2 = int(rng.choice(P.shape[2], p=P[s, a]))
        r = float(R[s, a, s2])
        G += disc * r
        disc *= gamma
        done = s2 in terminals
        if log is not None:
            log.append({"seed": seed, "episode": episode, "t": t, "s": s, "a": a, "r": r,
                        "s_next": s2, "done": done, "info": {}})
        s = s2
        visits.append(s)
        if done:
            return G, t + 1, visits
    return G, max_steps, visits
