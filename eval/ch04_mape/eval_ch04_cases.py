"""모델 평가 4장의 데이터. 모두 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다."""

import numpy as np


def first_screen():
    """첫 화면 손 계산 예. 세 가게의 하루 판매량과, 세 가게 모두 2개씩 틀린 예측."""
    y = np.array([100.0, 10.0, 1.0])
    return y, y + 2.0


def case_near_zero():
    """(a) 정답 0.01과 100, 두 점 모두 절대오차 1."""
    y = np.array([0.01, 100.0])
    return y, y + 1.0


def case_with_zero():
    """(b) 정답에 0이 하나. 설계도의 [0, 1] vs [1, 1]."""
    return np.array([0.0, 1.0]), np.array([1.0, 1.0])


def case_sklearn_doc_example():
    """scikit-learn 1.9.1 mean_absolute_percentage_error 문서 예제. 문서 출력은 112589990684262.48(비율)."""
    return np.array([1.0, 0.0, 2.4, 7.0]), np.array([1.2, 0.1, 2.4, 8.0])


def case_asymmetry_fixed_actual(y=100.0, err=50.0):
    """(c-1) 정답을 고정하고 같은 크기만큼 모자라게/넘치게."""
    return np.array([y, y]), np.array([y - err, y + err])


def case_asymmetry_fixed_forecast(f=100.0, err=50.0):
    """(c-2) 예측을 고정하고 정답이 같은 크기만큼 위/아래. (Makridakis 1993의 150/100 예는 정답과 예측을 맞바꾼 비교라 구조는 다르고, 정답이 작은 쪽이 더 큰 %를 받는 방향만 같다)"""
    return np.array([f + err, f - err]), np.array([f, f])


def case_temperature():
    """기온 5일(섭씨). 예측은 날마다 1~2도 틀린다. 단위만 바꿔 MAPE를 다시 잰다."""
    c = np.array([3.0, -1.0, 0.5, 6.0, 2.0])
    fc = np.array([2.0, 1.0, 1.5, 5.0, 3.0])
    return c, fc


def to_fahrenheit(c):
    return np.asarray(c, float) * 9.0 / 5.0 + 32.0


def to_kelvin(c):
    return np.asarray(c, float) + 273.15


def case_intermittent_demand(p_zero, n=60, lam=4.0, seed=0):
    """간헐 수요. 판매가 있는 날은 1 + Poisson(lam)개, 그날이 '판매 없음'일 확률은 p_zero.

    p_zero = 0이면 0이 한 번도 나오지 않는다(1 + Poisson은 늘 1 이상).
    """
    rng = np.random.default_rng(seed)
    y = 1.0 + rng.poisson(lam, n)
    y[rng.random(n) < p_zero] = 0.0
    return y


def intermittent_models(p_zero, lam=4.0):
    """같은 계열에 내는 상수 예측 세 개. 모두 '내일 판매량' 한 숫자를 매일 똑같이 낸다.

    A: 판매량의 기댓값 (1 - p) * (1 + lam)
    B: A의 60% (일부러 낮게 잡은 예측)
    Z: 늘 0 (아무것도 안 팔린다고 찍기)
    """
    a = (1.0 - p_zero) * (1.0 + lam)
    return {"A": a, "B": 0.6 * a, "Z": 0.0}


def case_level_shift(seed=0, n=50):
    """정답 ~ 10 + N(0, 3^2)을 1 아래로는 자르고, 예측 = 정답 + N(0, 1). 둘에 같은 c를 더해 수준만 옮긴다."""
    rng = np.random.default_rng(seed)
    y = np.maximum(10.0 + rng.normal(0.0, 3.0, n), 1.0)
    return y, y + rng.normal(0.0, 1.0, n)


def case_skewed_sales(n=60, seed=0):
    """오른쪽으로 긴 판매액(로그정규, 로그의 평균 2, 표준편차 0.8). 작은 날이 많고 큰 날이 가끔 있다."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(2.0, 0.8, n)
