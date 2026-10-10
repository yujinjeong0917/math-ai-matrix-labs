"""모델 평가 7장. 판독 사례집이 쓰는 회귀 지표 여섯 개와 진단값의 NumPy 최소 구현.

장끼리 동시에 작성되므로 import하지 않고, 앞 장의 정의를 필요한 만큼 복사했다.
  mae, rmse            : 1장 (eval/ch01_mae_rmse/eval_ch01_metrics.py)
  r2                   : 3장 (eval/ch03_r_squared/eval_ch03_metrics.py). 기준선은 '평가 데이터 자신의 평균'.
  mape                 : 4장 (eval/ch04_mape/eval_ch04_metrics.py). 0이 있으면 ValueError.
  naive_mae_in_sample, mase : 5장 (eval/ch05_mase/eval_ch05_metrics.py). Hyndman·Koehler(2006) 1스텝 naive.
  nmbe, cv_rmse        : 6장 (eval/ch06_normalized_bias/eval_ch06_metrics.py). 이 장은 분모를 n으로(p=0) 쓴다.

오차의 부호(이 장 고정): e = y - yhat (실제 - 예측). ASHRAE Guideline 14-2002 식 5.5와 같은 방향이라
NMBE > 0 은 '모자라게 찍음(과소예측)', NMBE < 0 은 '넘치게 찍음(과대예측)'이다.
"""

import numpy as np

METRICS = ("mae", "rmse", "r2", "mape", "mase", "nmbe")
# 순위를 매길 때 작을수록 좋은 지표와 클수록 좋은 지표. NMBE는 0에 가까울수록 좋다고 보고 |NMBE|로 줄 세운다.
HIGHER_IS_BETTER = {"r2"}


def _arr(a):
    return np.asarray(a, dtype=float)


def errors(y, yhat):
    """e = y - yhat."""
    return _arr(y) - _arr(yhat)


def mae(y, yhat):
    return float(np.mean(np.abs(errors(y, yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean(errors(y, yhat) ** 2)))


def r2(y, yhat):
    """R^2 = 1 - SSE/SST. 정답이 모두 같으면 정의되지 않는다(3장)."""
    y = _arr(y)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0.0:
        raise ValueError("정답이 모두 같아 R^2가 정의되지 않는다")
    return 1.0 - float(np.sum(errors(y, yhat) ** 2)) / sst


def mape(y, yhat):
    """100 * mean(|y - yhat| / |y|) (%). 정답에 0이 있으면 ValueError(4장의 zero='raise')."""
    y = _arr(y)
    if np.any(y == 0.0):
        raise ValueError(f"정답에 0이 {int(np.sum(y == 0.0))}개 있어 MAPE가 정의되지 않는다")
    return float(100.0 * np.mean(np.abs(errors(y, yhat)) / np.abs(y)))


def naive_mae_in_sample(y_train, lag=1):
    """학습 구간에서 'lag 시점 전 값 그대로 찍기'의 MAE. MASE의 분모(5장)."""
    y = _arr(y_train)
    if len(y) <= lag:
        raise ValueError("학습 구간이 lag보다 길어야 한다")
    d = float(np.mean(np.abs(y[lag:] - y[:-lag])))
    if d == 0.0:
        raise ValueError("학습 구간의 naive 오차가 0이라 MASE가 정의되지 않는다")
    return d


def mase(y_train, y_test, yhat, lag=1):
    """MAE(평가) / 학습 구간 1스텝 naive의 MAE. 1보다 크면 학습 구간의 naive보다 평균적으로 못하다."""
    return mae(y_test, yhat) / naive_mae_in_sample(y_train, lag)


def nmbe(y, yhat):
    """100 * sum(y - yhat) / (n * mean(y)) (%). 6장 식에서 p = 0. 양수 = 과소예측."""
    y = _arr(y)
    return float(100.0 * np.sum(errors(y, yhat)) / (len(y) * np.mean(y)))


def cv_rmse(y, yhat):
    """100 * RMSE / mean(y) (%). 6장 식에서 p = 0. 같은 평가 데이터에서는 RMSE와 순위가 같다."""
    return float(100.0 * rmse(y, yhat) / np.mean(_arr(y)))


def metric_row(y_train, y_test, yhat):
    """지표 여섯 개와 판독에 쓰는 진단값."""
    e = errors(y_test, yhat)
    m, r = mae(y_test, yhat), rmse(y_test, yhat)
    row = {"mae": m, "rmse": r, "r2": r2(y_test, yhat), "mape": mape(y_test, yhat),
           "mase": mase(y_train, y_test, yhat), "nmbe": nmbe(y_test, yhat), "cv_rmse": cv_rmse(y_test, yhat),
           "rmse_mae_ratio": r / m if m > 0 else float("nan"),
           # MSE 가운데 '평균 편향' 조각의 몫(6장 분해). 1에 가까우면 상수 하나로 거의 고쳐진다.
           "bias_share": float(np.mean(e) ** 2 / np.mean(e ** 2)) if np.any(e) else 0.0,
           # MAPE가 'MAE를 평균 수준으로 나눈 %'보다 몇 배 큰가. 크면 작은 정답값이 MAPE를 끌어올린 것이다.
           "mape_inflation": mape(y_test, yhat) / (100.0 * m / float(np.mean(np.abs(y_test)))) if m > 0 else float("nan")}
    return row


def better(metric, a, b):
    """지표 하나로 A와 B 중 고른다. 'A' 또는 'B'(같으면 'tie')."""
    if metric == "nmbe":
        a, b = abs(a), abs(b)
    if a == b:
        return "tie"
    if metric in HIGHER_IS_BETTER:
        return "A" if a > b else "B"
    return "A" if a < b else "B"
