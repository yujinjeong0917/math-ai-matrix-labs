"""모델 평가 3장. 결정계수 R², 제곱상관 r², 잔차 진단의 NumPy 최소 구현.

rmse는 1장(eval/ch01_mae_rmse/eval_ch01_metrics.py)에서 복사했다.
장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 옮겼다.
"""

import numpy as np


def _arr(a):
    return np.asarray(a, dtype=float)


def rmse(y, yhat):
    """평균 제곱근 오차. (1장에서 복사)"""
    return float(np.sqrt(np.mean((_arr(yhat) - _arr(y)) ** 2)))


def sse(y, yhat):
    """모델의 제곱오차 합 sum (y - yhat)^2."""
    return float(np.sum((_arr(y) - _arr(yhat)) ** 2))


def sst(y):
    """평가 데이터의 평균 하나로 찍었을 때의 제곱오차 합 sum (y - mean(y))^2."""
    y = _arr(y)
    return float(np.sum((y - y.mean()) ** 2))


def r2(y, yhat):
    """R^2 = 1 - SSE / SST. 기준선은 '평가 데이터 자신의 평균으로 찍기'다.

    정답이 모두 같으면(SST = 0) 비교할 기준이 없으므로 ValueError를 낸다.
    scikit-learn r2_score는 이 경우 기본값(force_finite=True)으로 1.0 또는 0.0을 조용히 돌려준다.
    """
    t = sst(y)
    if t == 0.0:
        raise ValueError("정답이 모두 같아 SST = 0이라 R^2가 정의되지 않는다")
    return 1.0 - sse(y, yhat) / t


def squared_corr(y, yhat):
    """피어슨 상관계수의 제곱 r^2. 예측이 정답과 '같이 움직이는지'만 본다(위치·크기는 무시)."""
    y, yhat = _arr(y), _arr(yhat)
    dy, dp = y - y.mean(), yhat - yhat.mean()
    den = np.sqrt(np.sum(dy**2) * np.sum(dp**2))
    if den == 0.0:
        raise ValueError("정답이나 예측이 모두 같아 상관계수가 정의되지 않는다")
    return float((np.sum(dy * dp) / den) ** 2)


def fit_line(x, y, intercept=True):
    """최소제곱 직선. intercept=True면 (b0, b1), False면 원점을 지나는 직선의 (0, b1)."""
    x, y = _arr(x), _arr(y)
    if intercept:
        b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        return float(y.mean() - b1 * x.mean()), float(b1)
    return 0.0, float(np.sum(x * y) / np.sum(x * x))


def predict_line(coef, x):
    b0, b1 = coef
    return b0 + b1 * _arr(x)


def fit_poly(x, y, deg):
    """최소제곱 다항식 계수(낮은 차수부터). 설계행렬 [1, x, x^2, ...]에 lstsq."""
    x, y = _arr(x), _arr(y)
    X = np.vander(x, deg + 1, increasing=True)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict_poly(coef, x):
    return np.vander(_arr(x), len(coef), increasing=True) @ coef


def residual_diagnostics(x, y, yhat, bins=3):
    """잔차 r = y - yhat의 평균, 잔차와 x의 상관, x 구간별 잔차 평균.

    R^2가 숨기는 두 가지를 본다. 잔차 평균이 0에서 멀면 예측이 한쪽으로 밀렸고(편향),
    구간별 평균이 +, -, + 처럼 부호를 바꾸면 직선이 곡선을 놓쳤다는 뜻이다.
    잔차-x 상관은 절편 있는 OLS 학습 데이터에서는 늘 0이라 곡률을 못 잡는다(그래서 구간별 평균을 같이 본다).
    """
    x, r = _arr(x), _arr(y) - _arr(yhat)
    order = np.argsort(x, kind="stable")
    chunks = np.array_split(order, bins)
    sx, sr = x.std(), r.std()
    corr = 0.0 if sx == 0.0 or sr == 0.0 else float(np.mean((x - x.mean()) * (r - r.mean())) / (sx * sr))
    return {
        "resid_mean": float(r.mean()),
        "resid_x_corr": corr,
        "bin_resid_mean": [float(r[c].mean()) for c in chunks],
        "max_abs_resid": float(np.max(np.abs(r))),
    }
