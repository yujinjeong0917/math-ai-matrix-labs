"""NumPy로 쓴 cart-pole(수레 위 막대 세우기) 동역학. 외부 패키지 없음.

10장 rl/ch10_policy_gradient/rl_ch10_cartpole.py를 그대로 복사했다(장끼리 동시에 쓰여 import하지 않음).
상수는 Barto 등(1983) p.846에서 다시 확인했다(확인일 2026-10-10).

식과 상수는 Barto·Sutton·Anderson(1983) IEEE SMC-13(5) 부록 "Details of the Cart-Pole Simulation"(p.845-846)을 따랐다.
  수레 질량 1.0 kg, 막대 질량 0.1 kg, 막대 길이의 절반 0.5 m, 힘 +-10 N, 오일러 적분 0.02 s.
이 구현에서 바꾼 점(본문에도 적었다):
  - 마찰 두 항(수레 0.0005, 막대 0.000002)은 0으로 뺐다. 상수가 작아 이 장의 비교에는 영향이 작다고 보고 단순하게 갔다.
  - 논문은 g = -9.8 m/s^2로 적었다. 이 구현은 g = +9.8을 넣어 막대가 똑바로 선 자세가 불안정하게(가만두면 쓰러지게) 했다.
    부호를 이렇게 고른 이유는 막대가 쓰러지는 실제 과제를 재현하려는 것이다. [해석]
  - 실패 조건 |x| > 2.4 m, |각도| > 12도는 논문이 상자(box) 경계로 쓴 가장 바깥값에서 가져왔다. [해석]
  - 한 판 상한 CAP 걸음(기본 500)에서 자른다. 잘린 판은 실패가 아니라 시간이 다 된 것으로 본다.

"""

import math

import numpy as np

GRAVITY = 9.8
M_CART = 1.0
M_POLE = 0.1
HALF_LEN = 0.5
FORCE = 10.0
TAU = 0.02
X_LIM = 2.4
TH_LIM = 12 * 2 * math.pi / 360
CAP = 500


class CartPole:
    def __init__(self, seed, cap=CAP):
        self.rng = np.random.default_rng(seed)
        self.cap = cap
        self.state = None
        self.t = 0

    def reset(self):
        # 네 상태값을 [-0.05, 0.05]에서 고르게 뽑아 시작한다(매번 같은 자세에서 시작하지 않게)
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, a):
        """a: 0 = 왼쪽으로 민다, 1 = 오른쪽으로 민다. (다음 상태, 보상 1, 쓰러졌나, 잘렸나)."""
        x, x_dot, th, th_dot = self.state
        F = FORCE if a == 1 else -FORCE
        cos, sin = math.cos(th), math.sin(th)
        m_total = M_CART + M_POLE
        tmp = (F + M_POLE * HALF_LEN * th_dot * th_dot * sin) / m_total
        th_acc = (GRAVITY * sin - cos * tmp) / (HALF_LEN * (4.0 / 3.0 - M_POLE * cos * cos / m_total))
        x_acc = tmp - M_POLE * HALF_LEN * th_acc * cos / m_total
        x = x + TAU * x_dot
        x_dot = x_dot + TAU * x_acc
        th = th + TAU * th_dot
        th_dot = th_dot + TAU * th_acc
        self.state = np.array([x, x_dot, th, th_dot])
        self.t += 1
        fell = bool(abs(x) > X_LIM or abs(th) > TH_LIM)
        truncated = (not fell) and self.t >= self.cap
        return self.state.copy(), 1.0, fell, truncated
