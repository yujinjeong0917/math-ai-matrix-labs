"""강화학습 5장용 환경. 이 장의 에이전트는 P와 R을 보지 않고 step()으로만 경험한다.

P[S,A,S], R[S,A,S]를 만드는 make_grid는 2장 rl/ch02_mdp_bellman/rl_ch02_gridworld.py에서 복사했다
(장끼리 동시에 쓰는 중이라 import하지 않음). P와 R은 "정답 확인용"과 "표본을 뽑는 시뮬레이터"로만 쓴다.
  - 행동 번호 0=위, 1=오른쪽, 2=아래, 3=왼쪽. 벽에 부딪히면 제자리.
  - 종료 칸에 들어가는 순간 그 칸의 보상을 받고 끝난다. 종료 칸은 흡수 상태(보상 0)로 둔다.
"""

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = ("↑", "→", "↓", "←")


def make_grid(h, w, rewards, terminals, slip=0.0, step_reward=0.0):
    """2장과 같은 P[S,A,S], R[S,A,S]와 종료 칸 표시 term[S]."""
    S, A = h * w, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    term = np.zeros(S, dtype=bool)
    for t in terminals:
        term[t[0] * w + t[1]] = True
    for s in range(S):
        if term[s]:
            # 종료 칸: 어떤 행동이든 자기 자신으로 돌아오는 흡수 상태. R은 0 그대로
            P[s, :, s] = 1.0
            continue
        r, c = divmod(s, w)
        for a in range(A):
            for b in range(A):
                # 식: P(실제 행동 b | 고른 행동 a) = (1 - slip) * 1[a = b] + slip / A. slip 확률로 네 방향을 균등하게 뽑는다(고른 행동 포함)
                prob = (1.0 - slip) * (a == b) + slip / A
                if prob == 0.0:
                    continue
                dr, dc = ACTIONS[b]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    nr, nc = r, c
                s2 = nr * w + nc
                # += 인 이유: 벽에 부딪힌 행동 b들이 모두 제자리 칸 s2로 모일 수 있어 확률을 합친다
                P[s, a, s2] += prob
                # 보상은 들어간 칸에 붙는다(R[s,a,s']). rewards에 없는 칸은 step_reward
                R[s, a, s2] = rewards.get((nr, nc), step_reward)
    return P, R, term


def uniform_policy(S, A=4):
    """모든 칸에서 행동 A개를 같은 확률 1/A로 고르는 정책 pi[S,A]."""
    return np.full((S, A), 1.0 / A)


def exact_v(P, R, pi, gamma, term):
    """정답 확인용. 3장 rl/ch03_policy_evaluation/rl_ch03_eval_np.py의 직접 풀이를 옮겼다.

    (I - gamma P^pi) V = r^pi. 종료 칸은 V=0으로 고정한다(gamma=1에서도 풀리게).
    """
    S = P.shape[0]
    # 식: P_pi[s,s'] = sum_a pi(a|s) P[s,a,s']
    P_pi = np.einsum("sa,sap->sp", pi, P)
    # 식: r_pi[s] = sum_a pi(a|s) sum_s' P[s,a,s'] R[s,a,s']
    r_pi = np.einsum("sa,sap,sap->s", pi, P, R)
    keep = ~term
    # 식: (I - gamma P_pi) V = r_pi. 종료 칸은 V = 0이라 그 열이 식에서 빠지므로 종료 칸을 뺀 부분 행렬만 푼다
    # (종료 칸으로 들어가는 보상은 r_pi에 이미 들어 있다)
    M = np.eye(int(keep.sum())) - gamma * P_pi[np.ix_(keep, keep)]
    V = np.zeros(S)
    V[keep] = np.linalg.solve(M, r_pi[keep])
    return V


# 2장의 격자: 5x5, 시작 (0,0), 바로 오른쪽 (0,1)에 +1, 6걸음 떨어진 (2,4)에 +10. 둘 다 들어가면 끝.
CH02_H, CH02_W = 5, 5
CH02_START = 0


def ch02_grid(slip=0.0):
    return make_grid(CH02_H, CH02_W, {(0, 1): 1.0, (2, 4): 10.0}, ((0, 1), (2, 4)), slip=slip)


# 3·4장과 같은 꼴: n x n, 오른쪽 아래 구석에 +1(들어가면 끝), 나머지 0.
def corner_grid(n, slip=0.0):
    g = (n - 1, n - 1)
    return make_grid(n, n, {g: 1.0}, (g,), slip=slip)


# 첫 화면의 복도 [A][B][도착]: 3장과 같다. 위·아래는 벽이라 쓰지 않고, 왼쪽 반·오른쪽 반.
def corridor():
    P, R, term = make_grid(1, 3, {(0, 2): 1.0}, ((0, 2),))
    pi = np.tile([0.0, 0.5, 0.0, 0.5], (3, 1))
    return P, R, term, pi


# 실패 2의 상태 하나짜리 문제. 상태 0에서 행동 0 = "한 번 더"(보상 +1, 상태 0으로 돌아옴),
# 행동 1 = "그만"(보상 0, 끝). 상태 1은 종료 칸. 리턴 G = "한 번 더"를 고른 횟수 k (gamma = 1).
def one_state():
    P = np.zeros((2, 2, 2))
    R = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    R[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, :, 1] = 1.0
    term = np.array([False, True])
    return P, R, term


def one_state_policy(p_more):
    """상태 0에서 "한 번 더"를 p_more, "그만"을 1 - p_more로 고르는 정책."""
    return np.array([[p_more, 1.0 - p_more], [0.5, 0.5]])
