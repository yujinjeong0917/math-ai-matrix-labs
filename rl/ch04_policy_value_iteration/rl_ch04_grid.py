"""강화학습 4장용 격자. P[S,A,S], R[S,A,S]를 통째로 만든다.

이동 규칙은 2장 rl/ch02_mdp_bellman/rl_ch02_gridworld.py와 3장 rl/ch03_policy_evaluation/rl_ch03_grid.py에서
복사했다(장끼리 동시에 쓰는 중이라 import하지 않음).
  - 행동 번호 0=위, 1=오른쪽, 2=아래, 3=왼쪽. 벽에 부딪히면 제자리.
  - slip: 확률 slip로 의도한 행동 대신 네 방향 중 하나를 균등하게 고른다(의도한 방향도 포함).
  - 종료 칸에 들어가는 순간 그 칸의 보상을 받고 끝난다. 종료 칸은 흡수 상태(보상 0)로 둔다.
  - 종료 칸이 아닌 칸에 보상을 두면, 그 칸에 들어갈 때마다(벽에 부딪혀 제자리에 남을 때도) 보상을 받는다.

4장은 정책을 계속 바꾸므로 3장처럼 P^pi만 만들지 않고 행동별 표 P[S,A,S]를 둔다.
이 장의 격자는 최대 20x20(400칸)이라 P 하나가 400*4*400*8B = 5.1MB로 충분히 작다.
"""

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = "↑→↓←"


def make_grid(h, w, rewards, terminals, slip=0.0, step_reward=0.0):
    S, A = h * w, len(ACTIONS)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    term = np.zeros(S, dtype=bool)
    for t in terminals:
        term[t[0] * w + t[1]] = True
    for s in range(S):
        if term[s]:
            P[s, :, s] = 1.0
            continue
        r, c = divmod(s, w)
        for a in range(A):
            for b in range(A):
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
    return P, R, term


def step(s, a, P, R, rng):
    s2 = int(rng.choice(P.shape[2], p=P[s, a]))
    return s2, float(R[s, a, s2])


# 첫 화면 손계산: 한 줄 복도 [A][B][도착]. 오른쪽/왼쪽만 의미가 있고(위·아래는 벽이라 제자리), 미끄러짐 없음.
def corridor():
    return make_grid(1, 3, {(0, 2): 1.0}, ((0, 2),))


# 실패 1(동률)과 첫 화면 실험: 6x6, 오른쪽 아래 구석이 도착 칸(+1), 미끄러짐 없음.
# 왼쪽 위에서 도착까지 최단 경로가 여러 개라 "오른쪽"과 "아래"가 똑같이 좋은 칸이 많다.
def tie_grid(n=6):
    return make_grid(n, n, {(n - 1, n - 1): 1.0}, ((n - 1, n - 1),))


# 실패 2: 종료 칸이 없는 3x3. 왼쪽 위 구석(0,0)에 들어가면(벽에 부딪혀 머물러도) +1.
def loop_grid():
    return make_grid(3, 3, {(0, 0): 1.0}, ())


# 비교 측정용: n x n, 미끄러짐 0.1, 오른쪽 아래 +1(종료), 함정 몇 칸 -1(종료).
def compare_grid(n=20, slip=0.1):
    pits = [(r, c) for r, c in ((n // 2, n // 2), (n // 2, n // 2 - 1), (n // 4, 3 * n // 4), (3 * n // 4, n // 4),
                                (n - 2, n - 3)) if 0 <= r < n and 0 <= c < n]
    rewards = {(n - 1, n - 1): 1.0}
    rewards.update({p: -1.0 for p in pits})
    return make_grid(n, n, rewards, [(n - 1, n - 1)] + pits, slip=slip), pits
