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

# 데이터 분포: 값 -1(확률 0.75)과 값 +1(확률 0.25)의 두 점. 25%가 마음에 들 확률 p에 대응한다(비용 계산 쪽의 p = 1/4 가정과 맞춘 장난감 설정).
MODES = (-1.0, 1.0)
WEIGHTS = (0.75, 0.25)
LIKED = 1.0          # 마음에 드는 그림
# abar는 수식의 알파 바(원래 값이 남는 누적 비율). 0이면 0으로 나누는 항이 생기므로 아주 작은 양수로 둔다.
ABAR_END = 1e-4      # 마지막 단계에서 원래 그림이 남는 비율(거의 0, 거의 순수한 잡음)


def schedule(T):
    """abar_0 = 1(잡음 없음)부터 abar_T = ABAR_END(거의 잡음)까지. 코사인 모양으로 줄인다."""
    out = []
    for t in range(T + 1):
        # 코사인 스케줄: abar_t = cos^2((t/T) * pi/2). t = 0에서 1, t = T에서 0으로 부드럽게 줄어든다(모형의 선택).
        a = math.cos((t / T) * math.pi / 2) ** 2
        out.append(max(a, ABAR_END) if t > 0 else 1.0)
    # 마지막 값을 정확히 ABAR_END로 고정(cos 값이 0이 되는 것을 막는 처리).
    out[T] = ABAR_END
    return out


def forward(x0, abar, eps):
    # 정방향 식: x_t = sqrt(abar) x0 + sqrt(1 - abar) eps. 계수의 제곱합이 1이라 분산이 유지된다.
    return math.sqrt(abar) * x0 + math.sqrt(1 - abar) * eps


def denoise_mean(x, abar, modes=MODES, weights=WEIGHTS):
    """x를 보고 원래 그림의 평균 추정 x0_hat (두 점 데이터의 정확한 답)."""
    # abar = 1이면 잡음이 없으니 분산 1 - abar가 0이라 나눗셈이 불가능. 관측값이 곧 추정값.
    if abar >= 1.0:
        return x
    var = 1 - abar
    # 가중치 w_i ∝ pi_i * exp(-(x - sqrt(abar) m_i)^2 / (2 (1 - abar))) 를 로그로 계산한다.
    logs = [math.log(w) - (x - math.sqrt(abar) * m) ** 2 / (2 * var) for m, w in zip(modes, weights)]
    # 수치 안정화(log-sum-exp 요령): 가장 큰 로그 값을 빼고 exp를 취해 오버플로·언더플로를 피한다. 정규화하면 결과는 같다.
    top = max(logs)
    ws = [math.exp(l - top) for l in logs]
    s = sum(ws)
    return sum(wi * m for wi, m in zip(ws, modes)) / s


def ddim_step(x, abar_t, abar_prev):
    x0_hat = denoise_mean(x, abar_t)
    # 정방향 식을 eps에 대해 풀어 추정 잡음을 구한다.
    eps_hat = (x - math.sqrt(abar_t) * x0_hat) / math.sqrt(1 - abar_t)
    # DDIM(결정적 역방향) 한 단계: 같은 식을 더 작은 abar에서 다시 조립한다. 새 난수를 쓰지 않으므로 시드가 같으면 결과가 같다.
    return math.sqrt(abar_prev) * x0_hat + math.sqrt(1 - abar_prev) * eps_hat


def start_noise(seed):
    # 시드마다 독립된 난수 객체를 만들어 전역 난수 상태에 영향받지 않게 한다.
    return random.Random(seed).gauss(0.0, 1.0)


def generate(seed, T=50, trace=False):
    """시드로 시작 잡음을 정하고 T단계로 걷어낸다. trace=True면 단계별 값 목록도 준다."""
    ab = schedule(T)
    # t = T(거의 순수 잡음)에서 1까지 내려오며 abar_t -> abar_(t-1)로 한 단계씩 걷어낸다. 시간복잡도 O(T).
    x = start_noise(seed)
    path = [x]
    for t in range(T, 0, -1):
        x = ddim_step(x, ab[t], ab[t - 1])
        path.append(x)
    return (x, path) if trace else x


def nearest_mode(x):
    return min(MODES, key=lambda m: abs(x - m))


# tol = 0.05: '+1 근처'의 허용 범위(모형의 선택). 단계가 적으면 +1에 못 닿아 비율이 낮게 나온다.
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
    # 이분법: 결과가 단조(시작 잡음이 클수록 결과가 큼)라는 가정. 60번이면 구간 길이가 6/2^60로 줄어 충분히 정확하다. 시간복잡도 O(iters * T).
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
        # 표준정규 위쪽 꼬리 확률 = 0.5 * (1 - erf(x/sqrt(2))).
        upper = 0.5 * (1 - math.erf(mid / math.sqrt(2)))
        if upper > q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
