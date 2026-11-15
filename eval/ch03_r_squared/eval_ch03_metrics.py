"""모델 평가 3장. 결정계수 R², 제곱상관 r², 잔차 진단의 NumPy 최소 구현.

rmse는 1장(eval/ch01_mae_rmse/eval_ch01_metrics.py)에서 복사했다.
장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 옮겼다.
"""

import numpy as np


# 리스트·정수 배열도 float ndarray로 통일
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
    # SST = 평가 데이터의 평균 하나로 찍은 모델의 SSE(기준선). 정답이 상수면 0
    t = sst(y)
    if t == 0.0:
        raise ValueError("정답이 모두 같아 SST = 0이라 R^2가 정의되지 않는다")
    # R^2 = 1 - SSE/SST. 기준선보다 나쁘면 음수가 될 수 있다(하한 없음). 상관계수 제곱과는 다른 양
    return 1.0 - sse(y, yhat) / t


def squared_corr(y, yhat):
    """피어슨 상관계수의 제곱 r^2. 예측이 정답과 '같이 움직이는지'만 본다(위치·크기는 무시)."""
    y, yhat = _arr(y), _arr(yhat)
    dy, dp = y - y.mean(), yhat - yhat.mean()
    # r = sum(dy dp) / sqrt(sum dy^2 sum dp^2). 분모가 0이면(한쪽이 상수) 정의 불가
    den = np.sqrt(np.sum(dy**2) * np.sum(dp**2))
    if den == 0.0:
        raise ValueError("정답이나 예측이 모두 같아 상관계수가 정의되지 않는다")
    return float((np.sum(dy * dp) / den) ** 2)


def fit_line(x, y, intercept=True):
    """최소제곱 직선. intercept=True면 (b0, b1), False면 원점을 지나는 직선의 (0, b1)."""
    x, y = _arr(x), _arr(y)
    if intercept:
        # OLS(최소제곱) 기울기 b1 = Sxy/Sxx, 절편 b0 = ybar - b1 xbar
        b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        return float(y.mean() - b1 * x.mean()), float(b1)
    # 절편 없는 직선: b1 = sum(xy)/sum(x^2). 평균 수평선을 포함하지 않아 R^2가 음수가 될 수 있다
    return 0.0, float(np.sum(x * y) / np.sum(x * x))


def predict_line(coef, x):
    b0, b1 = coef
    return b0 + b1 * _arr(x)


def fit_poly(x, y, deg):
    """최소제곱 다항식 계수(낮은 차수부터). 설계행렬 [1, x, x^2, ...]에 lstsq."""
    x, y = _arr(x), _arr(y)
    # 설계행렬 (n, deg+1): 열이 1, x, x^2, ... (increasing=True라 낮은 차수가 왼쪽)
    X = np.vander(x, deg + 1, increasing=True)
    # lstsq는 (해, 잔차합, 랭크, 특이값)을 돌려준다. [0]이 계수 벡터
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
    # x 오름차순 정렬 인덱스. stable로 같은 x의 순서를 고정(Anscombe 4번은 x=8이 10개)
    order = np.argsort(x, kind="stable")
    # 정렬된 관측을 개수가 거의 같은 bins개 구간으로 나눈다(구간 폭이 아니라 개수 기준)
    chunks = np.array_split(order, bins)
    sx, sr = x.std(), r.std()
    # 표준편차가 0이면 0 나누기라 상관을 0으로 둔다(정의 불가를 0으로 대체한 선택)
    corr = 0.0 if sx == 0.0 or sr == 0.0 else float(np.mean((x - x.mean()) * (r - r.mean())) / (sx * sr))
    return {
        "resid_mean": float(r.mean()),
        "resid_x_corr": corr,
        "bin_resid_mean": [float(r[c].mean()) for c in chunks],
        "max_abs_resid": float(np.max(np.abs(r))),
    }
