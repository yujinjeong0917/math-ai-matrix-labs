"""강화학습 3장용 격자. 고정 정책 하나를 넣어 P^pi (S x S)와 r^pi (S)를 바로 만든다.

이동 규칙은 2장 rl/ch02_mdp_bellman/rl_ch02_gridworld.py에서 복사했다(장끼리 동시에 쓰는 중이라 import하지 않음).
  - 행동 번호 0=위, 1=오른쪽, 2=아래, 3=왼쪽. 벽에 부딪히면 제자리.
  - 종료 칸에 들어가는 순간 그 칸의 보상을 받고 끝난다. 종료 칸은 흡수 상태(보상 0)로 둔다.

2장은 P[S,A,S]를 통째로 만들었지만, 70x70(4,900칸)이면 그 배열 하나가 약 770MB라서
3장은 정책을 먼저 곱해 P^pi[S,S]만 만든다(make_policy_mats). 희소 반복용으로 칸마다
"갈 수 있는 이웃 4곳과 그 확률"(nbr, prob)도 함께 돌려준다.
"""

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def make_grid_full(h, w, rewards, terminals, slip=0.0, step_reward=0.0):
    """2장과 같은 P[S,A,S], R[S,A,S]. 작은 격자에서 policy_matrix를 확인할 때만 쓴다."""
    S, A = h * w, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    term = {t[0] * w + t[1] for t in terminals}
    for s in range(S):
        if s in term:
            P[s, :, s] = 1.0
            continue
        r, c = divmod(s, w)
        for a in range(A):
            for b in range(A):
                # 행동 a를 골랐을 때 실제로 b 방향으로 갈 확률: 확률 slip만큼은 네 방향 균등(a 자신 포함)
                prob = (1.0 - slip) * (a == b) + slip / A
                if prob == 0.0:
                    continue
                dr, dc = ACTIONS[b]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    nr, nc = r, c
                s2 = nr * w + nc
                P[s, a, s2] += prob  # += : 벽에 부딪혀 여러 방향이 같은 칸(제자리)으로 모일 수 있다
                R[s, a, s2] = rewards.get((nr, nc), step_reward)  # 도착한 칸 기준 보상 R(s,a,s')
    return P, R


def policy_matrix(P, pi):
    """P^pi[s, s'] = sum_a pi(a|s) P(s'|s,a)."""
    return np.einsum("sa,sap->sp", pi, P)


def policy_reward(P, R, pi):
    """r^pi[s] = sum_a pi(a|s) sum_s' P(s'|s,a) R(s,a,s')."""
    return np.einsum("sa,sap,sap->s", pi, P, R)


def neighbors(h, w, terminals):
    """칸마다 네 행동이 데려가는 칸 nbr[S,4]. 종료 칸은 자기 자신으로 둔다."""
    S = h * w
    term = np.zeros(S, dtype=bool)
    for t in terminals:
        term[t[0] * w + t[1]] = True
    nbr = np.zeros((S, 4), dtype=np.int64)
    for s in range(S):
        r, c = divmod(s, w)
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            # 종료 칸이거나 격자 밖이면 제자리. 이 장은 미끄러짐이 없으므로 행동 a는 항상 방향 a로 이동
            if term[s] or not (0 <= nr < h and 0 <= nc < w):
                nr, nc = r, c
            nbr[s, a] = nr * w + nc
    return nbr, term


def make_policy_mats(h, w, rewards, terminals, pi=None, dense=True):
    """균등 무작위 정책(pi=None) 또는 pi[S,4]로 P^pi, r^pi, 희소 표현을 만든다.

    반환: dict(P_pi[S,S] 또는 None, r_pi[S], nbr[S,4], prob[S,4], rew[S,4], term[S])
    rew[s,a]는 행동 a로 도착한 칸에 들어갈 때 받는 보상.
    """
    nbr, term = neighbors(h, w, terminals)
    S = h * w
    prob = np.full((S, 4), 0.25) if pi is None else np.asarray(pi, dtype=float).copy()
    prob[term] = 0.25  # 종료 칸: 어디로 가든 자기 자신, 보상 0
    rew = np.zeros((S, 4))
    for (r, c), v in rewards.items():
        rew[nbr == r * w + c] = v  # (S, 4) 불리언 마스크: 그 칸으로 들어가는 모든 (s, a)에 보상
    rew[term] = 0.0  # 종료 칸의 자기 순환에는 보상을 다시 주지 않는다(보상은 들어가는 순간 한 번만)
    # 식: r^pi(s) = sum_a pi(a|s) R(s,a)  (이동이 결정적이라 sum_s' P(s'|s,a) R(s,a,s') 가 rew[s,a] 하나로 줄어듦)
    r_pi = (prob * rew).sum(axis=1)
    P_pi = None
    if dense:
        P_pi = np.zeros((S, S))
        # 식: P^pi[s, s'] = sum_a pi(a|s) [nbr[s,a] = s'].  add.at은 같은 (s, s')가 겹치면(벽 부딪힘) 확률을 누적한다
        np.add.at(P_pi, (np.repeat(np.arange(S), 4), nbr.ravel()), prob.ravel())
    return {"P_pi": P_pi, "r_pi": r_pi, "nbr": nbr, "prob": prob, "rew": rew, "term": term}


# 이 장의 실험 격자: n x n, 오른쪽 아래 구석에 +1(들어가면 끝), 나머지 칸 보상 0, 균등 무작위 정책.
def chapter_grid(n, dense=True):
    goal = (n - 1, n - 1)
    return make_policy_mats(n, n, {goal: 1.0}, (goal,), dense=dense)


# 첫 화면의 손계산 예: 한 줄 복도 [A][B][도착]. 왼쪽 반, 오른쪽 반. 도착하면 +1.
def corridor():
    pi = np.tile([0.0, 0.5, 0.0, 0.5], (3, 1))  # 위·아래는 쓰지 않음: 오른쪽 0.5, 왼쪽 0.5
    return make_policy_mats(1, 3, {(0, 2): 1.0}, ((0, 2),), pi=pi, dense=True)
