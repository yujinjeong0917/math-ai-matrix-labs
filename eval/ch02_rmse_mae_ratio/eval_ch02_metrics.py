"""모델 평가 2장. RMSE/MAE 비율과 그 기준값·범위의 NumPy 최소 구현.

mae, rmse는 1장(eval/ch01_mae_rmse/eval_ch01_metrics.py)에서 복사했다.
장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 옮겼다.
"""

import math

import numpy as np


def errors(y, yhat):
    """오차 e_i = yhat_i - y_i. (1장에서 복사)"""
    return np.asarray(yhat, dtype=float) - np.asarray(y, dtype=float)


def mae(y, yhat):
    """평균 절대 오차. (1장에서 복사)"""
    return float(np.mean(np.abs(errors(y, yhat))))


def rmse(y, yhat):
    """평균 제곱근 오차. (1장에서 복사)"""
    return float(np.sqrt(np.mean(errors(y, yhat) ** 2)))


def rmse_mae_ratio(y, yhat):
    """RMSE / MAE. 오차가 모두 0이면(MAE = 0) 비율이 정의되지 않으므로 ValueError를 낸다.

    분모는 오차의 '크기' 평균(mean |e|)이다. 부호 있는 평균(mean e)을 쓰면
    +와 -가 상쇄돼 0 근처에서 비율이 폭주한다(7절 흔한 실패).
    """
    m = mae(y, yhat)
    if m == 0.0:
        raise ValueError("MAE가 0이라 RMSE/MAE 비율이 정의되지 않는다")
    return rmse(y, yhat) / m


def ratio_via_variance(e):
    """같은 비율을 sqrt(1 + Var(|e|) / MAE^2)로 계산한다. RMSE^2 = MAE^2 + Var(|e|) 이기 때문."""
    a = np.abs(np.asarray(e, dtype=float))
    m = a.mean()
    return float(np.sqrt(1.0 + a.var() / m**2))


def ratio_bounds(n):
    """오차 n개에서 비율이 가질 수 있는 범위 [1, sqrt(n)]."""
    return 1.0, math.sqrt(n)


def ratio_reference(dist, df=None):
    """오차가 한 분포에서 무한히 많이 나왔을 때의 비율 sqrt(E[e^2]) / E|e| (해석적 값).

    uniform: U(-a, a)        -> E|e| = a/2,          E e^2 = a^2/3   -> 2/sqrt(3)
    normal:  N(0, s^2)       -> E|e| = s*sqrt(2/pi), E e^2 = s^2     -> sqrt(pi/2)
    laplace: Laplace(0, b)   -> E|e| = b,            E e^2 = 2 b^2   -> sqrt(2)
    t:       t(df), df > 2   -> E|e| = 2 sqrt(df) G((df+1)/2) / (sqrt(pi) (df-1) G(df/2)),
                                E e^2 = df/(df-2).  df <= 2면 분산이 무한이라 inf.
    """
    if dist == "uniform":
        return 2.0 / math.sqrt(3.0)
    if dist == "normal":
        return math.sqrt(math.pi / 2.0)
    if dist == "laplace":
        return math.sqrt(2.0)
    if dist == "t":
        if df is None or df <= 1:
            raise ValueError("t 분포는 df > 1 이어야 E|e|가 유한하다")
        if df <= 2:
            return math.inf
        e_abs = 2 * math.sqrt(df) * math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / (math.sqrt(math.pi) * (df - 1))
        return math.sqrt(df / (df - 2)) / e_abs
    raise ValueError(dist)
