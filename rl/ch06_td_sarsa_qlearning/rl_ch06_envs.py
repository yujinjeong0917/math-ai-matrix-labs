"""강화학습 6장용 환경. 에이전트는 step()으로만 경험하고 전이확률 표를 보지 않는다.

- 무작위 보행: 칸 0 ~ n+1, 양 끝(0, n+1)은 종료 칸. 가운데 칸에서 출발해 반반으로 왼쪽·오른쪽.
  오른쪽 끝에 들어가면 보상 1, 나머지는 0, gamma = 1. 참값 V(i) = i / (n + 1).
  n = 5는 Sutton(1988) 3.2절의 B~F 다섯 칸과 같은 꼴이다.
- 절벽 격자: H x W, 왼쪽 아래 출발, 오른쪽 아래 도착(종료). 아래 줄의 출발과 도착 사이는 절벽.
  한 걸음 -1, 절벽에 들어가면 -cliff를 받고 출발 칸으로 돌아간다(판은 계속). Sutton·Barto 예제 6.6과
  같은 꼴이지만 크기(4 x 12 -> 4 x 10)와 절벽 보상(-100 -> -50)을 바꿨다.
- 복도 [A][B][도착]: 5장 첫 화면과 같다(gamma 0.9, 도착하면 1점).
- 결정적 사슬: 테스트용. 칸 0 ~ n-1, 행동 0 = 왼쪽, 1 = 오른쪽, 오른쪽 끝 칸 n-1에 들어가면 보상 1로 끝.

행동 번호(격자): 0 = 위, 1 = 오른쪽, 2 = 아래, 3 = 왼쪽. 벽에 부딪히면 제자리. 5장 rl_ch05_grid.py와 같은 순서.
"""

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = ("↑", "→", "↓", "←")


class Uniforms:
    """rng.random()을 4,096개씩 미리 뽑아 두고 하나씩 꺼낸다. 5장 rl_ch05_mc_np._Uniforms에서 복사."""

    def __init__(self, rng, block=4096):
        self.rng, self.block = rng, block
        self.buf, self.i = rng.random(block).tolist(), 0

    def __call__(self):
        if self.i == self.block:
            self.buf, self.i = self.rng.random(self.block).tolist(), 0
        u = self.buf[self.i]
        self.i += 1
        return u


# ---------- 무작위 보행 ----------

def walk_true_values(n=5):
    """종료 칸 두 개를 포함한 길이 n+2 배열. 종료 칸은 0."""
    v = np.arange(n + 2) / (n + 1)
    v[0] = v[-1] = 0.0
    return v


def walk_episode(uni, n=5):
    """한 판의 전이 목록 [(s, r, s_next, done), ...]. 가운데 칸에서 출발."""
    s = (n + 1) // 2
    out = []
    while True:
        s2 = s + 1 if uni() < 0.5 else s - 1
        done = s2 == 0 or s2 == n + 1
        r = 1.0 if s2 == n + 1 else 0.0
        out.append((s, r, s2, done))
        if done:
            return out
        s = s2


# ---------- 절벽 격자 ----------

class Cliff:
    def __init__(self, h=4, w=10, cliff=50.0, step=-1.0):
        self.h, self.w, self.cliff_r, self.step_r = h, w, -float(cliff), float(step)
        self.S, self.A = h * w, 4
        self.start = (h - 1) * w
        self.goal = (h - 1) * w + (w - 1)
        self.cliff = set(range((h - 1) * w + 1, (h - 1) * w + w - 1))

    def step(self, s, a):
        """(s_next, r, done, fell). 절벽이면 출발 칸으로 돌아가고 판은 계속된다."""
        r, c = divmod(s, self.w)
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.h and 0 <= nc < self.w):
            nr, nc = r, c
        s2 = nr * self.w + nc
        if s2 in self.cliff:
            return self.start, self.cliff_r, False, True
        if s2 == self.goal:
            return s2, self.step_r, True, False
        return s2, self.step_r, False, False


# ---------- 복도 (5장 첫 화면) ----------

CORRIDOR_NAMES = "AB"


def corridor_transitions(path):
    """'AABAB◎' 같은 문자열을 전이 목록 [(s, r, s_next, done)]로. A=0, B=1, ◎=2(도착, 1점)."""
    idx = {"A": 0, "B": 1, "◎": 2}
    out = []
    for x, y in zip(path[:-1], path[1:]):
        s, s2 = idx[x], idx[y]
        out.append((s, 1.0 if s2 == 2 else 0.0, s2, s2 == 2))
    return out


# 5장 첫 화면의 네 판(5장 results/ch05.json의 E0_corridor.first_episodes_seed0와 같은 경로).
CH05_FIRST_FOUR = ("AABAAAAAAAB◎", "AB◎", "AB◎", "AB◎")


# ---------- 결정적 사슬 (테스트용) ----------

def chain_step(s, a, n):
    """(s_next, r, done). 0 = 왼쪽(왼쪽 끝은 벽), 1 = 오른쪽. 칸 n-1에 들어가면 보상 1로 끝."""
    s2 = max(s - 1, 0) if a == 0 else s + 1
    if s2 == n - 1:
        return s2, 1.0, True
    return s2, 0.0, False


def chain_q_star(n, gamma):
    """Q*(s, a)의 해석해. 오른쪽으로만 가면 칸 s에서 도착까지 n-1-s걸음, 보상은 마지막 걸음에 1."""
    Q = np.zeros((n, 2))
    for s in range(n - 1):
        Q[s, 1] = gamma ** (n - 2 - s)              # 오른쪽: 남은 걸음 n-1-s, 할인은 (걸음 - 1)번
        Q[s, 0] = gamma * gamma ** (n - 2 - max(s - 1, 0))  # 왼쪽 한 걸음 뒤 최적
    return Q
