"""모델 평가 5장. 척도 오차(scaled error)와 MASE의 NumPy 최소 구현.

정의는 Hyndman·Koehler(2006) 3절(저자 사전 공개본 2005-11-02, robjhyndman.com/papers/mase.pdf)을 따른다.
  q_t  = e_t / ( (1/(n-1)) * sum_{i=2..n} |Y_i - Y_{i-1}| )
  MASE = mean(|q_t|)
오차의 부호는 그 논문처럼 e = y - yhat (정답 - 예측)로 둔다.
lag=m(계절 naive) 분모는 이 논문이 아니라 Hyndman·Athanasopoulos, FPP3 5.8절의 정의다.
mae, ape, mape(zero 정책)는 4장(eval/ch04_mape/eval_ch04_metrics.py)에서 복사했다.
장끼리 동시에 작성되므로 import하지 않는다.
"""

import numpy as np

EPS = float(np.finfo(np.float64).eps)  # 4장에서 복사: scikit-learn 1.9.1이 0 대신 나누는 값


def _arr(a):
    return np.asarray(a, dtype=float)


def mae(y, yhat):
    """평균 절대 오차. (1장·4장에서 복사)"""
    return float(np.mean(np.abs(_arr(yhat) - _arr(y))))


def ape(y, yhat):
    """(4장에서 복사) 시점마다 100 * |y - yhat| / |y|."""
    y, yhat = _arr(y), _arr(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.abs(y - yhat) / np.abs(y)


def mape(y, yhat, zero="raise"):
    """(4장에서 복사) 평균 절대 백분율 오차(%). zero = raise | drop | eps."""
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


def naive_mae_in_sample(y_train, lag=1):
    """학습 구간에서 'lag 시점 전 값을 그대로 쓰는' 예측의 MAE. MASE의 분모.

    lag=1: Hyndman·Koehler(2006)의 1스텝 naive.  lag=m: 계절 naive(FPP3 5.8절).
    lag=h를 '1스텝이 아닌 h스텝 naive'로 읽으면, 같은 시계의 naive로 나누는 변형
    (H&K 3절 "scale by the in-sample MAE computed from multi-step naive forecasts")이 된다.
    학습 구간의 값이 모두 같으면 분모가 0이라 ValueError를 낸다(H&K: "all historical observations are equal").
    """
    y = _arr(y_train)
    if lag < 1 or len(y) <= lag:
        raise ValueError(f"학습 구간 길이 {len(y)}가 lag {lag}보다 길어야 한다")
    d = float(np.mean(np.abs(y[lag:] - y[:-lag])))
    if d == 0.0:
        raise ValueError("학습 구간의 naive 오차가 0이다(값이 모두 같다). MASE가 정의되지 않는다")
    return d


def scaled_errors(y_train, y_test, yhat, lag=1):
    """q_t = (y_t - yhat_t) / 분모. 부호가 남아 있다."""
    return (_arr(y_test) - _arr(yhat)) / naive_mae_in_sample(y_train, lag)


def mase(y_train, y_test, yhat, lag=1):
    """평균 절대 척도 오차. 한 계열 안에서는 MAE(y_test, yhat) / 분모와 같다."""
    return float(np.mean(np.abs(scaled_errors(y_train, y_test, yhat, lag))))


def naive_forecast(y_train, h):
    """학습 구간 끝에서 한 번 예측: 1..h 시점 모두 마지막 학습 값."""
    return np.full(h, _arr(y_train)[-1])


def naive_rolling(y_train, y_test, lag=1):
    """평가 구간에서 매 시점 lag 전 '실제' 값을 쓰는 예측(평가 구간 값을 보면서 다시 예측)."""
    full = np.concatenate([_arr(y_train), _arr(y_test)])
    n = len(_arr(y_train))
    return full[n - lag: n - lag + len(_arr(y_test))]


def mean_forecast(y_train, h):
    """학습 구간 평균을 h 시점 모두에 쓰는 예측."""
    return np.full(h, float(np.mean(_arr(y_train))))
