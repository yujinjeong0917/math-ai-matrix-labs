"""모델 평가 6장. 크기를 정규화한 지표(NMAE, CV(RMSE))와 방향을 남기는 지표(MBE, NMBE)의 NumPy 최소 구현.

부호 규약(이 장 고정): ASHRAE Guideline 14-2002 식 5.4, 5.5를 따른다.
  e_i  = y_i - yhat_i                       (실제 - 예측)
  CV(RMSE) = 100 * sqrt( sum e_i^2 / (n - p) ) / mean(y)        (식 5.4)
  NMBE     = 100 * sum e_i / ( (n - p) * mean(y) )              (식 5.5)
  보정 시뮬레이션은 p = 1 (5.2.11.3절).
그래서 NMBE > 0 은 '모자라게 찍음(과소예측)', NMBE < 0 은 '넘치게 찍음(과대예측)'이다.

다른 규약 두 가지도 같은 함수에서 고를 수 있게 했다.
  femp : FEMP M&V Guidelines 4.0 식 4-1, MBE(%) = 100 * sum(S - M) / sum(M). 분모가 n, 부호가 반대.
  sfa  : Solar Forecast Arbiter, NMBE = 100/norm * mean(F - O). norm 은 정격 용량. 부호는 FEMP와 같다.
mae, mse, rmse 는 1장(eval/ch01_mae_rmse)과 같은 정의라 여기서 다시 쓴다. 장끼리 동시에 작성되므로 import하지 않는다.
"""

import numpy as np

# ASHRAE Guideline 14-2002 5.3.2.4 f: 월별 NMBE 5%, CV(RMSE) 15% / 시간별 10%, 30%.
# 원문은 "NMBE of 5%"라고만 쓴다. 절댓값으로 비교하는 건 이 코드의 해석이다(FEMP 표 4-2는 ±5%로 적는다).
ASHRAE_2002_LIMITS = {"monthly": {"nmbe": 5.0, "cv_rmse": 15.0}, "hourly": {"nmbe": 10.0, "cv_rmse": 30.0}}


def _arr(a):
    return np.asarray(a, dtype=float)


def errors(y, yhat):
    """e = y - yhat (실제 - 예측). 이 장의 모든 부호는 여기서 정해진다."""
    return _arr(y) - _arr(yhat)


def mae(y, yhat):
    return float(np.mean(np.abs(errors(y, yhat))))


def mse(y, yhat):
    return float(np.mean(errors(y, yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mbe(y, yhat):
    """평균 편향 오차(단위 그대로). 양수면 과소예측."""
    return float(np.mean(errors(y, yhat)))


def _check_dof(n, p):
    if n - p <= 0:
        raise ValueError(f"데이터 수 n={n}가 모델 변수 수 p={p}보다 커야 한다 (n - p > 0)")


def nmbe(y, yhat, p=1, convention="ashrae2002", norm=None):
    """정규화 평균 편향 오차(%).

    ashrae2002: 100 * sum(y - yhat) / ((n - p) * mean(y)). 양수 = 과소예측.
    femp      : 100 * sum(yhat - y) / sum(y). p를 쓰지 않는다(분모 n). 양수 = 과대예측.
    sfa       : 100 / norm * mean(yhat - y). norm 필수(정격 용량). 양수 = 과대예측.
    """
    y, yhat = _arr(y), _arr(yhat)
    n = len(y)
    if convention == "ashrae2002":
        _check_dof(n, p)
        return float(100.0 * np.sum(y - yhat) / ((n - p) * np.mean(y)))
    if convention == "femp":
        return float(100.0 * np.sum(yhat - y) / np.sum(y))
    if convention == "sfa":
        if norm is None:
            raise ValueError("sfa 규약은 norm(정격 용량)이 필요하다")
        return float(100.0 / norm * np.mean(yhat - y))
    raise ValueError(f"알 수 없는 규약: {convention}")


def cv_rmse(y, yhat, p=1):
    """ASHRAE 2002 식 5.4: 100 * sqrt(sum e^2 / (n - p)) / mean(y). p=0이면 FEMP 식 4-2~4-4와 같다."""
    y = _arr(y)
    n = len(y)
    _check_dof(n, p)
    return float(100.0 * np.sqrt(np.sum(errors(y, yhat) ** 2) / (n - p)) / np.mean(y))


def norm_value(y, kind, capacity=None):
    """NMAE 분모. capacity(정격 용량, 없으면 관측 최댓값), mean(관측 평균), range(관측 최댓값 - 최솟값)."""
    y = _arr(y)
    if kind == "capacity":
        return float(capacity if capacity is not None else np.max(y))
    if kind == "mean":
        return float(np.mean(y))
    if kind == "range":
        return float(np.max(y) - np.min(y))
    raise ValueError(f"알 수 없는 분모: {kind}")


def nmae(y, yhat, norm):
    """정규화 평균 절대 오차(%). norm은 숫자 하나. 단일 표준식이 없어서 분모를 꼭 같이 적는다."""
    return float(100.0 * mae(y, yhat) / float(norm))


def mse_decomposition(y, yhat):
    """MSE = mean(e)^2 + Var(e). 분모는 n(모집단 분산). p를 쓰면 이 등식이 성립하지 않는다."""
    e = errors(y, yhat)
    bias = float(np.mean(e))
    var = float(np.mean((e - bias) ** 2))
    return {"mse": float(np.mean(e ** 2)), "bias": bias, "bias_sq": bias ** 2, "var": var,
            "bias_share": (bias ** 2) / float(np.mean(e ** 2)) if np.any(e) else 0.0}


def offset_correct(y, yhat):
    """예측 전체에 평균 오차를 더한다. 편향 조각만 지우고 흔들림 조각은 그대로 둔다."""
    return _arr(yhat) + mbe(y, yhat)


def ashrae_check(y, yhat, resolution, p=1):
    """ASHRAE Guideline 14-2002 5.3.2.4 f의 기준으로 통과 여부. |NMBE|로 비교하는 건 [해석]."""
    lim = ASHRAE_2002_LIMITS[resolution]
    nb, cv = nmbe(y, yhat, p=p), cv_rmse(y, yhat, p=p)
    ok_b, ok_cv = abs(nb) <= lim["nmbe"], cv <= lim["cv_rmse"]
    return {"nmbe": nb, "cv_rmse": cv, "nmbe_ok": bool(ok_b), "cv_ok": bool(ok_cv), "pass": bool(ok_b and ok_cv)}
