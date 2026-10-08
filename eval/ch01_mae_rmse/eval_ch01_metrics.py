"""모델 평가 1장. MAE·MSE·RMSE와 '상수 하나로 예측할 때 가장 좋은 값'의 NumPy 최소 구현.

파일 이름에 장 번호를 붙인 이유: 여러 장의 테스트를 한꺼번에 돌릴 때 pytest가
같은 이름의 모듈(metrics.py)을 섞어 불러오지 않게 하려는 것이다.
"""

import numpy as np


def errors(y, yhat):
    """오차 e_i = yhat_i - y_i. 부호는 이 장에서 쓰지 않지만 6장(편향)에서 다시 쓴다."""
    return np.asarray(yhat, dtype=float) - np.asarray(y, dtype=float)


def mae(y, yhat):
    """평균 절대 오차: 오차 크기를 그대로 평균 낸다."""
    return float(np.mean(np.abs(errors(y, yhat))))


def mse(y, yhat):
    """평균 제곱 오차: 오차를 제곱해 평균 낸다. 큰 오차가 제곱만큼 커진다."""
    return float(np.mean(errors(y, yhat) ** 2))


def rmse(y, yhat):
    """평균 제곱근 오차: MSE에 제곱근을 씌워 단위를 원래대로 되돌린다."""
    return float(np.sqrt(mse(y, yhat)))


def best_constant(y, loss, num=20001):
    """모든 경우에 같은 값 c 하나로 예측할 때, loss를 가장 작게 만드는 c를 격자 탐색으로 찾는다.

    loss="mae"면 중앙값 근처, loss="mse"면 평균 근처가 나와야 한다.
    격자 탐색이라 정확도는 격자 간격(step) 정도다. step도 함께 돌려준다.
    """
    y = np.asarray(y, dtype=float)
    grid = np.linspace(y.min(), y.max(), num)
    step = float(grid[1] - grid[0])
    diff = grid[:, None] - y[None, :]
    if loss == "mae":
        scores = np.mean(np.abs(diff), axis=1)
    elif loss == "mse":
        scores = np.mean(diff**2, axis=1)
    else:
        raise ValueError(loss)
    return float(grid[np.argmin(scores)]), step


def rmse_bounds_hold(e):
    """MAE <= RMSE <= sqrt(n) * MAE 이 성립하는지(2장 예고). 부동소수 여유 1e-12."""
    e = np.asarray(e, dtype=float)
    m, r, n = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), e.size
    return m <= r + 1e-12 and r <= np.sqrt(n) * m + 1e-12
