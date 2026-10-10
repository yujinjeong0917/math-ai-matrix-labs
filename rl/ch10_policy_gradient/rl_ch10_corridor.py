"""강화학습 10장의 짧은 복도. 행동 효과가 가운데 칸에서 뒤집히는 3칸 복도.

환경은 Sutton·Barto 2판(2018) 예제 13.1 "Short corridor with switched actions"의 설정을 그대로 빌렸다
(걸음마다 보상 -1, 할인 없음, 칸 세 개, 행동 두 개, 모든 칸이 정책에게 똑같이 보임).
  - 칸 0(A): 오른쪽 -> 칸 1, 왼쪽 -> 제자리(벽)
  - 칸 1(B): 행동이 뒤집힌다. 오른쪽 -> 칸 0, 왼쪽 -> 칸 2
  - 칸 2(C): 오른쪽 -> 도착(끝), 왼쪽 -> 칸 1
  - 걸음마다 보상 -1. 그래서 시작 칸의 가치 J = -(도착까지 걸린 걸음 수의 기댓값).

정책은 칸을 구분하지 못한다. 매개변수 theta 하나로 "오른쪽 확률" p = sigmoid(theta)를 모든 칸에서 쓴다.
행동 번호 0 = 왼쪽, 1 = 오른쪽.
"""

import numpy as np

S = 3  # 끝나지 않는 칸 수. 도착 칸은 따로 번호를 두지 않고 done으로 처리한다
LEFT, RIGHT = 0, 1
NAMES = "ABC"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def step(s, a):
    """(다음 칸, 보상, 끝났나). 다음 칸이 도착이면 -1을 돌려준다."""
    if s == 0:
        return (1 if a == RIGHT else 0), -1.0, False
    if s == 1:
        return (0 if a == RIGHT else 2), -1.0, False
    # s == 2
    if a == RIGHT:
        return -1, -1.0, True
    return 1, -1.0, False


def transition_matrix(p):
    """정책(오른쪽 확률 p)을 따를 때 끝나지 않는 칸 사이의 이동 확률 P_pi (3x3).

    행 합이 1보다 작은 칸(C)은 그만큼 도착으로 빠져나간다.
    """
    P = np.zeros((S, S))
    P[0, 1] += p
    P[0, 0] += 1 - p
    P[1, 0] += p
    P[1, 2] += 1 - p
    P[2, 1] += 1 - p
    return P


def exact_v(p):
    """V = -1 + P_pi V 를 연립방정식으로 푼다(할인 없음, 보상 -1). p가 0이나 1이면 끝나지 않아 -inf."""
    if p <= 0.0 or p >= 1.0:
        return np.full(S, -np.inf)
    P = transition_matrix(p)
    return np.linalg.solve(np.eye(S) - P, -np.ones(S))


def exact_J(theta):
    """시작 칸 A의 가치 J(theta) = V_pi(A)."""
    return float(exact_v(sigmoid(theta))[0])


def exact_dJ_dp(p):
    """dV/dp = (I-P)^{-1} (dP/dp) V. 선형 연립방정식을 미분한 식."""
    P = transition_matrix(p)
    V = exact_v(p)
    dP = np.zeros((S, S))
    dP[0, 1], dP[0, 0] = 1.0, -1.0
    dP[1, 0], dP[1, 2] = 1.0, -1.0
    dP[2, 1] = -1.0
    return np.linalg.solve(np.eye(S) - P, dP @ V)


def exact_grad(theta):
    """dJ/dtheta = dJ/dp * p(1-p)."""
    p = sigmoid(theta)
    return float(exact_dJ_dp(p)[0] * p * (1 - p))


def best_p(grid=None):
    """J(p)를 최대로 하는 p. 촘촘한 격자에서 찾고 이분법으로 다듬는다."""
    lo, hi = 0.01, 0.99
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if exact_dJ_dp(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
