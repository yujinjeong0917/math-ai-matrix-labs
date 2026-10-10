"""AI 도구 실전 7장: 숫자 하나짜리 "그림"으로 해 보는 잡음 제거 장난감.

학습 데이터: 값 하나짜리 그림. 75%는 -1("평범한 그림"), 25%는 +1("마음에 드는 그림")이라고 둔다.
정방향(잡음 섞기): x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps,  eps ~ 정규분포(평균 0, 분산 1)
역방향(잡음 걷어내기): 지금 값 x_t를 보고 "원래 그림이 무엇이었을지"의 평균 추정 x0_hat을 구한 뒤,
  그 추정과 거기서 거꾸로 푼 잡음 eps_hat으로 한 단계 덜 섞인 값을 다시 만든다(DDIM 방식, 새 잡음 없음).

실제 모델은 x0_hat을 신경망이 맞히지만, 여기서는 데이터가 두 점뿐이라 정확한 답을 식으로 바로 쓴다:
  x0_hat = sum_i w_i m_i,  w_i ∝ pi_i * exp(-(x_t - sqrt(abar_t) m_i)^2 / (2 (1 - abar_t)))
학습이 없으니 이 장난감이 보여 주는 건 "잡음에서 시작해 여러 단계로 걷어내는 방식"과 "시작 잡음(시드)이
결과를 정한다"는 점뿐이다.

표준 라이브러리만 쓴다(math, random).
"""

import math
import random

MODES = (-1.0, 1.0)
WEIGHTS = (0.75, 0.25)
LIKED = 1.0          # 마음에 드는 그림
ABAR_END = 1e-4      # 마지막 단계에서 원래 그림이 남는 비율(거의 0, 거의 순수한 잡음)


def schedule(T):
    """abar_0 = 1(잡음 없음)부터 abar_T = ABAR_END(거의 잡음)까지. 코사인 모양으로 줄인다."""
    out = []
    for t in range(T + 1):
        a = math.cos((t / T) * math.pi / 2) ** 2
        out.append(max(a, ABAR_END) if t > 0 else 1.0)
    out[T] = ABAR_END
    return out


def forward(x0, abar, eps):
    return math.sqrt(abar) * x0 + math.sqrt(1 - abar) * eps


def denoise_mean(x, abar, modes=MODES, weights=WEIGHTS):
    """x를 보고 원래 그림의 평균 추정 x0_hat (두 점 데이터의 정확한 답)."""
    if abar >= 1.0:
        return x
    var = 1 - abar
    logs = [math.log(w) - (x - math.sqrt(abar) * m) ** 2 / (2 * var) for m, w in zip(modes, weights)]
    top = max(logs)
    ws = [math.exp(l - top) for l in logs]
    s = sum(ws)
    return sum(wi * m for wi, m in zip(ws, modes)) / s


def ddim_step(x, abar_t, abar_prev):
    x0_hat = denoise_mean(x, abar_t)
    eps_hat = (x - math.sqrt(abar_t) * x0_hat) / math.sqrt(1 - abar_t)
    return math.sqrt(abar_prev) * x0_hat + math.sqrt(1 - abar_prev) * eps_hat


def start_noise(seed):
    return random.Random(seed).gauss(0.0, 1.0)


def generate(seed, T=50, trace=False):
    """시드로 시작 잡음을 정하고 T단계로 걷어낸다. trace=True면 단계별 값 목록도 준다."""
    ab = schedule(T)
    x = start_noise(seed)
    path = [x]
    for t in range(T, 0, -1):
        x = ddim_step(x, ab[t], ab[t - 1])
        path.append(x)
    return (x, path) if trace else x


def nearest_mode(x):
    return min(MODES, key=lambda m: abs(x - m))


def liked_fraction(seeds, T=50, tol=0.05):
    """여러 시드 중 결과가 '마음에 드는 그림'(+1) 근처(tol 안)에 떨어진 비율."""
    hits = sum(1 for s in seeds if abs(generate(s, T) - LIKED) < tol)
    return hits / len(seeds)


def settled_fraction(seeds, T, tol=0.05):
    """결과가 어느 한 그림(-1 또는 +1) 근처에 떨어진 비율. 단계가 적으면 중간값에 머문다."""
    hits = sum(1 for s in seeds if min(abs(generate(s, T) - m) for m in MODES) < tol)
    return hits / len(seeds)


def two_step_noise_share(a1, a2):
    """개념 카드의 손계산: 남기는 비율 a1, a2로 두 번 섞으면 원래 값 몫 a1*a2, 잡음 몫 1 - a1*a2."""
    return a1 * a2, 1 - a1 * a2


def run_from(x_start, T=50):
    ab = schedule(T)
    x = x_start
    for t in range(T, 0, -1):
        x = ddim_step(x, ab[t], ab[t - 1])
    return x


def boundary(T=50, lo=-3.0, hi=3.0, iters=60):
    """시작 잡음이 이 값보다 크면 +1, 작으면 -1로 끝나는 경계(이분법)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if run_from(mid, T) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def normal_upper_quantile(q):
    """표준정규분포에서 위쪽 꼬리 확률이 q가 되는 값(이분법, math.erf 사용)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        upper = 0.5 * (1 - math.erf(mid / math.sqrt(2)))
        if upper > q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
