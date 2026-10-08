"""강화학습 2장용 Gridworld. 전이확률 P[S,A,S']와 보상 R[S,A,S']를 표로 만든다.

트랙 설계도는 공용 모듈 rl/common/gridworld.py를 두라고 했지만, 여러 장을 동시에 쓰는 중이라
공용 폴더를 만들지 않고 이 장 안에 둔다. 다른 장이 필요하면 이 파일을 복사해 쓰고 출처를 적는다.

행동 번호: 0=위, 1=오른쪽, 2=아래, 3=왼쪽. 벽에 부딪히면 제자리.
slip: 확률 slip로 의도한 행동 대신 네 방향 중 하나를 균등하게 고른다(의도한 방향도 포함).
종료 칸에 들어가는 순간 그 칸의 보상을 받고 에피소드가 끝난다. 종료 칸은 흡수 상태(보상 0)로 둔다.
"""

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
ARROWS = ("↑", "→", "↓", "←")


def to_index(rc, w):
    return rc[0] * w + rc[1]


def to_rc(s, w):
    return divmod(s, w)


def make_grid(h, w, rewards, terminals, slip=0.0, step_reward=0.0):
    """rewards: {(r, c): 보상} 그 칸에 들어갈 때 받는 보상. terminals: 종료 칸 집합.

    반환: P (S, A, S), R (S, A, S). 모든 (s, a)에서 P[s, a].sum() == 1.
    """
    S, A = h * w, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    term = {to_index(t, w) for t in terminals}
    for s in range(S):
        if s in term:
            P[s, :, s] = 1.0  # 흡수 상태: 더 움직이지 않고 보상도 없다
            continue
        r, c = to_rc(s, w)
        for a in range(A):
            for b in range(A):  # b: 실제로 일어나는 방향
                prob = (1.0 - slip) * (a == b) + slip / A
                if prob == 0.0:
                    continue
                dr, dc = ACTIONS[b]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    nr, nc = r, c
                s2 = nr * w + nc
                P[s, a, s2] += prob
                R[s, a, s2] = rewards.get((nr, nc), step_reward)
    return P, R


def step(s, a, P, R, rng):
    s2 = int(rng.choice(P.shape[2], p=P[s, a]))
    return s2, float(R[s, a, s2])


# 이 장의 실험 격자: 5x5, 시작 (0,0), 바로 오른쪽 (0,1)에 +1, 6걸음 떨어진 (2,4)에 +10.
H, W = 5, 5
START = (0, 0)
SMALL = (0, 1)
BIG = (2, 4)
REWARDS = {SMALL: 1.0, BIG: 10.0}
TERMINALS = (SMALL, BIG)


def chapter_grid(slip=0.0):
    return make_grid(H, W, REWARDS, TERMINALS, slip=slip)
