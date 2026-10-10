"""강화학습 11장 cart-pole 환경(NumPy, 여러 판을 한꺼번에).

상수와 운동 방정식은 Gymnasium의 classic_control/cartpole.py(main 브랜치, 2026-10-08 확인)를
그대로 옮겼다: g=9.8, 수레 1.0kg, 막대 0.1kg, 막대 반길이 0.5m, 힘 10N, 시간 간격 0.02초,
오일러 적분, 막대가 12도를 넘거나 수레가 2.4m를 벗어나면 끝, 시작 상태는 각 성분 U(-0.05, 0.05).
그 파일은 "Barto, Sutton, Anderson(1983)이 설명한 cart-pole 문제에 대응한다"고 적어 두었다.
1983 논문 원문과 상수 하나하나의 대응은 이 장에서 직접 확인하지 않았다([확인 필요]).
판 길이 상한은 Gymnasium v0과 같은 200걸음, 보상은 매 걸음 +1(끝나는 걸음 포함).

트랙의 다른 장과 코드를 나누지 않는다(장끼리 동시에 쓰므로). 필요한 것은 모두 이 폴더 안에 있다.
"""

import math

import numpy as np

GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL_M = M_CART + M_POLE
POLE_ML = M_POLE * HALF_LEN
THETA_LIMIT = 12 * 2 * math.pi / 360
X_LIMIT = 2.4
MAX_STEPS = 200


def step(state, action):
    """state: (n, 4) = [x, x_dot, theta, theta_dot], action: (n,) 0=왼쪽 1=오른쪽. 반환: 다음 상태, 끝남 여부."""
    x, x_dot, th, th_dot = state.T
    force = np.where(action == 1, FORCE, -FORCE)
    cos, sin = np.cos(th), np.sin(th)
    temp = (force + POLE_ML * th_dot**2 * sin) / TOTAL_M
    th_acc = (GRAVITY * sin - cos * temp) / (HALF_LEN * (4.0 / 3.0 - M_POLE * cos**2 / TOTAL_M))
    x_acc = temp - POLE_ML * th_acc * cos / TOTAL_M
    nxt = np.stack([x + TAU * x_dot, x_dot + TAU * x_acc, th + TAU * th_dot, th_dot + TAU * th_acc], axis=1)
    done = (np.abs(nxt[:, 0]) > X_LIMIT) | (np.abs(nxt[:, 2]) > THETA_LIMIT)
    return nxt, done


def reset(n, rng):
    return rng.uniform(-0.05, 0.05, size=(n, 4))
