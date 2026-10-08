"""모델 평가 4장. 백분율 오차(APE)와 MAPE의 NumPy 최소 구현.

mae는 1장(eval/ch01_mae_rmse/eval_ch01_metrics.py)에서 복사했다.
장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 옮겼다.
오차의 부호는 Hyndman·Koehler(2006)를 따라 e = y - yhat (정답 - 예측)로 둔다.
"""

import numpy as np

EPS = float(np.finfo(np.float64).eps)  # scikit-learn 1.9.1이 0 대신 나누는 값 (2.22e-16)


def _arr(a):
    return np.asarray(a, dtype=float)


def mae(y, yhat):
    """평균 절대 오차. (1장에서 복사)"""
    return float(np.mean(np.abs(_arr(yhat) - _arr(y))))


def ape(y, yhat):
    """시점마다 백분율 오차의 크기 100 * |y - yhat| / |y|. y = 0인 자리는 inf(오차 있음) 또는 nan(0/0)."""
    y, yhat = _arr(y), _arr(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.abs(y - yhat) / np.abs(y)


def mape(y, yhat, zero="raise"):
    """평균 절대 백분율 오차(%). 정답에 0이 있을 때의 처리를 zero로 고른다.

    zero="raise" : 정의대로. 0이 하나라도 있으면 ValueError.
    zero="drop"  : 정답이 0인 시점을 빼고 평균(이 장의 정책. 빠진 시점 수는 따로 봐야 한다).
    zero="eps"   : scikit-learn 1.9.1처럼 |y|를 max(|y|, EPS)로 바꿔 나눈다. 단, 이 함수는 %로 돌려준다
                   (sklearn은 비율이라 이 값의 1/100).
    """
    y, yhat = _arr(y), _arr(yhat)
    zeros = y == 0.0
    if zero == "raise":
        if zeros.any():
            raise ValueError(f"정답에 0이 {int(zeros.sum())}개 있어 MAPE가 정의되지 않는다")
        return float(np.mean(ape(y, yhat)))
    if zero == "drop":
        if zeros.all():
            raise ValueError("정답이 모두 0이라 남는 시점이 없다")
        return float(np.mean(ape(y[~zeros], yhat[~zeros])))
    if zero == "eps":
        return float(100.0 * np.mean(np.abs(yhat - y) / np.maximum(np.abs(y), EPS)))
    raise ValueError(f"알 수 없는 zero 정책: {zero}")


def ape_bounds_demo(y=100.0, forecasts=(0.0, 50.0, 100.0, 150.0, 200.0, 400.0, 1000.0)):
    """정답 y를 고정하고 예측만 바꿨을 때의 APE. 0 이상 예측에서 모자란 쪽은 100%에서 막힌다."""
    return {float(f): float(ape([y], [f])[0]) for f in forecasts}


def best_constant_mape(y):
    """MAPE를 가장 작게 만드는 상수 예측 c (정답이 모두 양수일 때).

    MAPE(c) = (100/n) * sum |y_i - c| / y_i 는 가중치 w_i = 1/y_i 인 가중 절대오차 합이라,
    최솟값은 가중 중앙값에서 나온다(누적 가중치가 절반을 처음 넘는 y). MAE의 최적 상수가
    (가중치 없는) 중앙값인 것과 같은 논리다. 작은 정답일수록 가중치가 커서 c가 아래로 끌려간다.
    """
    y = _arr(y)
    if np.any(y <= 0):
        raise ValueError("정답이 모두 양수여야 한다")
    order = np.argsort(y, kind="stable")
    ys, w = y[order], 1.0 / y[order]
    cum = np.cumsum(w)
    return float(ys[np.searchsorted(cum, 0.5 * cum[-1])])
