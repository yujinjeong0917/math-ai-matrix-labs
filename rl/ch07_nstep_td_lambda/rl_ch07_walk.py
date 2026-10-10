"""강화학습 7장용 환경. 에이전트는 전이 기록만 보고 전이확률 표를 보지 않는다.

- 무작위 보행: 칸 0 ~ n+1, 양 끝(0, n+1)은 종료 칸. 가운데 칸에서 출발해 반반으로 왼쪽·오른쪽.
  왼쪽 끝에 들어가면 보상 -1, 오른쪽 끝에 들어가면 +1, 나머지는 0, gamma = 1.
  참값 V(i) = 2i/(n+1) - 1. 이 장은 n = 13(가운데 칸 7, 참값 -6/7 ~ 6/7)을 쓴다.
  교과서(Sutton·Barto 2판 그림 7.2, 12.3, 12.6)의 19칸과 겹치지 않게 칸 수를 바꿨다.
- 외길: 테스트·첫 화면용. 칸 0 ~ n-1, 늘 오른쪽으로 가고, 오른쪽 끝 칸 n-1에 들어가면 보상 1로 끝.
  6장 rl_ch06_envs.chain_step과 같은 꼴(6장 첫 화면의 "외길 9판")이다.
- 손계산 판: 첫 화면용. 칸 A~E(1~5), C에서 출발해 C B C D E 를 지나 오른쪽 끝으로 나가 +1.

전이 하나는 (s, r, s_next, done) 튜플이다(6장과 같은 꼴).
"""

import numpy as np

N_WALK = 13


class Uniforms:
    """rng.random()을 4,096개씩 미리 뽑아 두고 하나씩 꺼낸다. 6장 rl_ch06_envs.Uniforms에서 복사(원래 5장)."""

    def __init__(self, rng, block=4096):
        self.rng, self.block = rng, block
        self.buf, self.i = rng.random(block).tolist(), 0

    def __call__(self):
        if self.i == self.block:
            self.buf, self.i = self.rng.random(self.block).tolist(), 0
        u = self.buf[self.i]
        self.i += 1
        return u


def walk_true_values(n=N_WALK):
    """종료 칸 두 개를 포함한 길이 n+2 배열. 종료 칸은 0."""
    v = 2.0 * np.arange(n + 2) / (n + 1) - 1.0
    v[0] = v[-1] = 0.0
    return v


def walk_episode(uni, n=N_WALK):
    """한 판의 전이 목록. 6장 walk_episode에서 복사하고 왼쪽 끝 보상을 -1로 바꿨다."""
    s = (n + 1) // 2
    out = []
    while True:
        s2 = s + 1 if uni() < 0.5 else s - 1
        done = s2 == 0 or s2 == n + 1
        r = 1.0 if s2 == n + 1 else (-1.0 if s2 == 0 else 0.0)
        out.append((s, r, s2, done))
        if done:
            return out
        s = s2


def walk_episodes(seed, n_episodes, n=N_WALK):
    """시드 하나의 판 묶음. 모든 방법·보폭이 같은 묶음을 본다."""
    uni = Uniforms(np.random.default_rng(seed))
    return [walk_episode(uni, n) for _ in range(n_episodes)]


def rmse(V, n=N_WALK):
    """종료 칸을 뺀 n칸의 RMSE."""
    v = np.asarray(V, dtype=float)
    return float(np.sqrt(np.mean((v[1:n + 1] - walk_true_values(n)[1:n + 1]) ** 2)))


def chain_episode(n=10):
    """외길 한 판: 0 -> 1 -> ... -> n-1(종료, 보상 1)."""
    return [(s, 1.0 if s + 1 == n - 1 else 0.0, s + 1, s + 1 == n - 1) for s in range(n - 1)]


def chain_true_values(n=10, gamma=0.9):
    v = np.array([gamma ** (n - 2 - s) for s in range(n)], dtype=float)
    v[n - 1] = 0.0
    return v


# 손계산 판: 칸 번호 A=1, B=2, C=3, D=4, E=5, 오른쪽 끝 6(종료, +1), 왼쪽 끝 0
HAND_PATH = [3, 2, 3, 4, 5, 6]
HAND_NAMES = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def hand_episode():
    return [(s, 1.0 if s2 == 6 else 0.0, s2, s2 == 6) for s, s2 in zip(HAND_PATH[:-1], HAND_PATH[1:])]
