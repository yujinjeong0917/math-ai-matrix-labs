"""강화학습 2장 NumPy 최소 구현: 벨만 백업 두 개와 롤아웃.

V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [r(s,a,s') + gamma V^pi(s')]      (기대방정식)
V*(s)   = max_a      sum_s' P(s'|s,a) [r(s,a,s') + gamma V*(s')]            (최적방정식)

방정식을 "어떻게 푸는가"는 3장(정책 평가)과 4장(가치 반복)의 주제다. 여기서는 백업을 오차가
1e-13 아래로 떨어질 때까지 되풀이하는 가장 단순한 방법으로 해를 얻어 쓰기만 한다.

기호: P[s,a,s']는 전이확률, R[s,a,s']는 보상, gamma는 할인율(이 장은 1 미만), pi[s,a]는 정책 확률(S, A).
이 파일에는 탐험 파라미터(eps 등)와 학습률이 없다. 모델(P, R)을 안다고 가정한 계획(planning)이며 표본 학습이 아니다.
출력: 가치 벡터 V (S,), 행동가치 Q (S, A), 정책 pi (S, A), 롤아웃 한 번의 할인 리턴.
"""

import numpy as np


def q_from_v(V, P, R, gamma):
    """Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]."""
    # 식: Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]. V[None,None,:]을 (1,1,S')로 늘려 (S,A,S')와 맞추고, einsum이 s' 축을 합한다
    return np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_expectation_backup(V, pi, P, R, gamma):
    # 식(벨만 기대방정식 백업): V(s) <- sum_a pi(a|s) Q(s,a). 행동 축(axis=1)으로 pi 가중 합
    return (pi * q_from_v(V, P, R, gamma)).sum(axis=1)


def bellman_optimality_backup(V, P, R, gamma):
    # 식(벨만 최적방정식 백업): V(s) <- max_a Q(s,a). 합 대신 최댓값
    return q_from_v(V, P, R, gamma).max(axis=1)


def iterate(backup, S, tol=1e-13, max_iter=100_000):
    # V를 0에서 시작. 종료 칸은 흡수 상태(보상 0)라 V=0에 머문다
    V = np.zeros(S)
    for k in range(1, max_iter + 1):
        # 백업 한 번 = 위 식의 우변을 모든 s에 동기식으로 적용(모두 옛 V를 읽고 새 V2를 만든다)
        V2 = backup(V)
        # 무한 노름(가장 큰 한 칸의 변화)이 tol 미만이면 수렴으로 본다. gamma < 1이면 백업이 수축이라 해가 유일하고 수렴. 반환 k는 반복 횟수
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
    # 부동소수 오차 때문에 정확히 같지 않은 동점을 허용 오차 tol로 묶는다
    best = Q >= Q.max(axis=1, keepdims=True) - tol
    return best / best.sum(axis=1, keepdims=True)


def myopic_policy(P, R):
    """1장 밴딧식: 지금 한 걸음의 기대 보상 E[r | s, a]만 보고 고른다."""
    # 식: E[r|s,a] = sum_s' P(s'|s,a) R(s,a,s'). gamma V(s') 항이 없다 = gamma=0인 탐욕. 보상이 한 걸음 안에 없으면 모든 행동이 0으로 동점(균등 무작위)
    return argmax_policy(np.einsum("sap,sap->sa", P, R))


def optimal_policy(P, R, gamma):
    # 식: pi*(s) = argmax_a Q*(s,a), Q*는 최적 V*에서 q_from_v로 계산
    V, _ = optimal_values(P, R, gamma)
    return argmax_policy(q_from_v(V, P, R, gamma))


def rollout(pi, P, R, gamma, s0, terminals, rng, max_steps=500, log=None, seed=None, episode=None):
    """한 에피소드를 돌리고 할인 리턴, 걸음 수, 방문한 상태 목록을 돌려준다."""
    # G = 할인 리턴 sum_t gamma^t r_t, disc = gamma^t (현재 시점의 할인 계수)
    s, G, disc, visits = s0, 0.0, 1.0, [s0]
    for t in range(max_steps):
        # a ~ pi(.|s)
        a = int(rng.choice(pi.shape[1], p=pi[s]))
        # s' ~ P(.|s,a)
        s2 = int(rng.choice(P.shape[2], p=P[s, a]))
        r = float(R[s, a, s2])
        # 식: G += gamma^t * r (첫 보상 t=0은 할인 없음)
        G += disc * r
        disc *= gamma
        # 종료 칸에 들어가는 순간 에피소드 끝. 이 보상 r은 이미 G에 더해졌다
        done = s2 in terminals
        if log is not None:
            log.append({"seed": seed, "episode": episode, "t": t, "s": s, "a": a, "r": r,
                        "s_next": s2, "done": done, "info": {}})
        s = s2
        visits.append(s)
        if done:
            return G, t + 1, visits
    # max_steps에 걸려 끝난 경우(종료 칸 미도달)의 반환. 리턴은 잘린 값
    return G, max_steps, visits
