"""모델 평가 1장. MAE·MSE·RMSE와 '상수 하나로 예측할 때 가장 좋은 값'의 NumPy 최소 구현.

파일 이름에 장 번호를 붙인 이유: 여러 장의 테스트를 한꺼번에 돌릴 때 pytest가
같은 이름의 모듈(metrics.py)을 섞어 불러오지 않게 하려는 것이다.
"""

import numpy as np


def errors(y, yhat):
    """오차 e_i = yhat_i - y_i. 부호는 이 장에서 쓰지 않지만 6장(편향)에서 다시 쓴다."""
    # 부호는 예측 - 정답(과대 예측이면 양수). 정수 입력도 float로 바꿔 계산한다
    return np.asarray(yhat, dtype=float) - np.asarray(y, dtype=float)


def mae(y, yhat):
    """평균 절대 오차: 오차 크기를 그대로 평균 낸다."""
    # MAE = (1/n) sum |e_i|
    return float(np.mean(np.abs(errors(y, yhat))))


def mse(y, yhat):
    """평균 제곱 오차: 오차를 제곱해 평균 낸다. 큰 오차가 제곱만큼 커진다."""
    # MSE = (1/n) sum e_i^2
    return float(np.mean(errors(y, yhat) ** 2))


def rmse(y, yhat):
    """평균 제곱근 오차: MSE에 제곱근을 씌워 단위를 원래대로 되돌린다."""
    # RMSE = sqrt(MSE). 단위가 y와 같아 MAE와 직접 비교할 수 있다
    return float(np.sqrt(mse(y, yhat)))


def best_constant(y, loss, num=20001):
    """모든 경우에 같은 값 c 하나로 예측할 때, loss를 가장 작게 만드는 c를 격자 탐색으로 찾는다.

    loss="mae"면 중앙값 근처, loss="mse"면 평균 근처가 나와야 한다.
    격자 탐색이라 정확도는 격자 간격(step) 정도다. step도 함께 돌려준다.
    """
    y = np.asarray(y, dtype=float)
    # 후보 c를 y의 최솟값~최댓값 사이 num개 격자로 둔다(두 손실의 최소점은 이 구간 안에 있다)
    grid = np.linspace(y.min(), y.max(), num)
    step = float(grid[1] - grid[0])
    # (num, n) 모든 (후보 c, 관측 y_i) 쌍의 차이 c - y_i
    diff = grid[:, None] - y[None, :]
    if loss == "mae":
        # (num,) 후보별 MAE. axis=1은 관측값 방향 평균
        scores = np.mean(np.abs(diff), axis=1)
    elif loss == "mse":
        # (num,) 후보별 MSE
        scores = np.mean(diff**2, axis=1)
    else:
        raise ValueError(loss)
    # argmin은 같은 값이 여럿이면 가장 앞 인덱스를 고른다. 짝수 n의 MAE는 가운데 두 값 사이가 평평해 구간의 왼쪽 끝이 나온다
    return float(grid[np.argmin(scores)]), step


def rmse_bounds_hold(e):
    """MAE <= RMSE <= sqrt(n) * MAE 이 성립하는지(2장 예고). 부동소수 여유 1e-12."""
    e = np.asarray(e, dtype=float)
    # 두 부등식의 근거: MAE <= RMSE는 코시-슈바르츠(Jensen) 부등식, RMSE <= sqrt(n)*MAE는 오차 하나가 전부일 때 등호(모든 e가 한 점에 몰린 극단)
    m, r, n = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), e.size
    return m <= r + 1e-12 and r <= np.sqrt(n) * m + 1e-12
