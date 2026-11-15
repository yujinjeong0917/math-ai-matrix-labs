"""모델 평가 2장의 합성 데이터. 1장처럼 정답 y는 0으로 두고 예측값 = 오차로 쓴다.

case_* 함수는 (y, yhat) 쌍(둘 다 모양 (n,))을, sample_errors는 오차 배열 (n,) 하나를 돌려준다.
목적은 RMSE/MAE 비율이 오차 분포의 모양에 따라 1 ~ sqrt(n) 사이 어디에 놓이는지 만드는 것이다.
"""

import math

import numpy as np


def case_uniform_errors(n, size=1.0, seed=0):
    """모든 오차의 크기가 size로 같다. 부호만 무작위. 비율은 정확히 1."""
    rng = np.random.default_rng(seed)
    return np.zeros(n), rng.choice([-1.0, 1.0], size=n) * size


def case_single_spike(n, big=None, seed=0):
    """오차 하나만 크기 big, 나머지는 0. big을 안 주면 n으로 둬서 MAE = 1. 비율은 정확히 sqrt(n)."""
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    # 오차 n개 중 한 곳만 크기 big. big=n이면 합 |e| = n이라 MAE = 1, RMSE = n/sqrt(n) = sqrt(n)
    e[rng.integers(n)] = float(n if big is None else big) * rng.choice([-1.0, 1.0])
    return np.zeros(n), e


def case_two_level(n, frac_nonzero, size=1.0, seed=0):
    """round(frac_nonzero * n)개는 크기 size, 나머지는 0. 비율은 sqrt(n / k).

    정규분포가 아니어도 비율이 정규분포의 기준값과 거의 같아질 수 있음을 보이는 데 쓴다.
    """
    rng = np.random.default_rng(seed)
    # 0이 아닌 오차 개수 k. 비율 = sqrt(mean e^2)/mean|e| = sqrt(k size^2/n)/(k size/n) = sqrt(n/k)
    k = int(round(frac_nonzero * n))
    e = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    e[idx] = size * rng.choice([-1.0, 1.0], size=k)
    return np.zeros(n), e


def sample_errors(dist, n, seed=0, df=None, unit_mae=True):
    """분포별 오차 n개. unit_mae=True면 '이론상 MAE가 1'이 되게 크기를 맞춘다.

    normal: sigma = sqrt(pi/2) 이면 E|e| = 1
    laplace: b = 1 이면 E|e| = 1
    uniform: U(-2, 2) 이면 E|e| = 1
    t: E|T|로 나눠서 E|e| = 1 (df > 1)
    """
    rng = np.random.default_rng(seed)
    if dist == "normal":
        # E|N(0,s^2)| = s sqrt(2/pi) 이므로 s = sqrt(pi/2)면 1
        return rng.normal(0.0, math.sqrt(math.pi / 2) if unit_mae else 1.0, n)
    if dist == "laplace":
        return rng.laplace(0.0, 1.0, n)
    if dist == "uniform":
        return rng.uniform(-2.0, 2.0, n) if unit_mae else rng.uniform(-1.0, 1.0, n)
    # t 분포는 자유도가 작으면 평균 크기 E|T|가 달라서 해석적 E|T|로 나눠 맞춘다(df > 1에서만 유한)
    if dist == "t":
        x = rng.standard_t(df, n)
        if unit_mae:
            # E|T| = 2 sqrt(df) G((df+1)/2) / (sqrt(pi)(df-1) G(df/2)), G는 감마 함수. lgamma로 로그를 취해 큰 df에서 넘침을 피한다
            e_abs = 2 * math.sqrt(df) * math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / (math.sqrt(math.pi) * (df - 1))
            x = x / e_abs
        return x
    raise ValueError(dist)


def case_heavy_tail(df, n, seed=0):
    """자유도 df인 t 분포 오차. df가 작을수록 꼬리가 두껍다(df <= 2면 분산이 무한)."""
    return np.zeros(n), sample_errors("t", n, seed=seed, df=df)
