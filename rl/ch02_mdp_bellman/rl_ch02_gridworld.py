"""강화학습 2장용 Gridworld. 전이확률 P[S,A,S']와 보상 R[S,A,S']를 표로 만든다.

트랙 설계도는 공용 모듈 rl/common/gridworld.py를 두라고 했지만, 여러 장을 동시에 쓰는 중이라
공용 폴더를 만들지 않고 이 장 안에 둔다. 다른 장이 필요하면 이 파일을 복사해 쓰고 출처를 적는다.

행동 번호: 0=위, 1=오른쪽, 2=아래, 3=왼쪽. 벽에 부딪히면 제자리.
slip: 확률 slip로 의도한 행동 대신 네 방향 중 하나를 균등하게 고른다(의도한 방향도 포함).
종료 칸에 들어가는 순간 그 칸의 보상을 받고 에피소드가 끝난다. 종료 칸은 흡수 상태(보상 0)로 둔다.

입력: 격자 크기, 칸별 보상, 종료 칸, slip(미끄러짐 확률), step_reward(일반 칸에 들어갈 때의 보상).
출력: P[s,a,s'] (S, A, S)와 R[s,a,s'] (S, A, S). 상태 번호 s = 행 * 폭 + 열.
보상은 (s,a,s') 중 도착 칸 s' 기준으로만 정해진다(도착 칸이 같으면 어느 방향으로 왔든 같은 보상).
식: P(s'|s,a) = sum_b [(1-slip)*1(a=b) + slip/4] * 1(b로 움직인 결과 = s')  (의도 a, 실제 방향 b)
할인율은 이 파일에 없다(백업과 롤아웃에서 gamma를 받는다).
"""

import numpy as np

# (행 변화, 열 변화). 행은 아래로 갈수록 커진다. 행동 번호는 위 docstring의 0=위, 1=오른쪽, 2=아래, 3=왼쪽
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
    # 상태 수 S는 칸 수, 종료 칸도 상태 하나로 센다
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
                # 식: 실제 방향 b의 확률 = (1-slip)*1(a=b) + slip/A. slip=0이면 결정적, slip>0이면 의도한 방향에도 slip/A가 더해진다(합은 항상 1)
                prob = (1.0 - slip) * (a == b) + slip / A
                if prob == 0.0:
                    continue
                dr, dc = ACTIONS[b]
                nr, nc = r + dr, c + dc
                # 격자 밖이면 제자리(벽). 제자리여도 도착 칸 보상은 현재 칸 기준으로 정해진다
                if not (0 <= nr < h and 0 <= nc < w):
                    nr, nc = r, c
                s2 = nr * w + nc
                # +=: 서로 다른 방향이 같은 칸(예: 벽에 부딪혀 제자리)으로 이어지면 확률이 합쳐진다
                P[s, a, s2] += prob
                # 도착 칸에 보상이 지정돼 있으면 그 값, 아니면 step_reward. 종료 칸 안(흡수 상태)에서는 위에서 continue해 R이 0으로 남는다
                R[s, a, s2] = rewards.get((nr, nc), step_reward)
    return P, R


def step(s, a, P, R, rng):
    s2 = int(rng.choice(P.shape[2], p=P[s, a]))
    return s2, float(R[s, a, s2])


# 이 장의 실험 격자: 5x5, 시작 (0,0), 바로 오른쪽 (0,1)에 +1, 6걸음 떨어진 (2,4)에 +10.
# 최적 경로: (0,0)에서 (2,4)까지 6걸음이라 +10의 현재가치는 10*gamma^5. gamma=0.9면 5.9049로 +1보다 크다
H, W = 5, 5
START = (0, 0)
SMALL = (0, 1)
BIG = (2, 4)
REWARDS = {SMALL: 1.0, BIG: 10.0}
TERMINALS = (SMALL, BIG)


def chapter_grid(slip=0.0):
    return make_grid(H, W, REWARDS, TERMINALS, slip=slip)
