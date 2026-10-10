"""모델 평가 6장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다."""

import numpy as np

HOURS_PER_YEAR = 8760
# 윤년이 아닌 해의 달력 월 길이(일). 월 합계는 이 경계로 자른다(같은 길이 730시간 블록이 아님).
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def first_screen():
    """첫 화면 손 계산 예. 4시간 동안의 전력 사용량(kWh)과 두 모델의 예측.
    A는 2만큼 위아래로 번갈아 틀리고(오차 +2, -2, +2, -2), B는 늘 2만큼 모자라게 찍는다(오차 +2 네 번)."""
    y = np.array([8.0, 12.0, 10.0, 10.0])
    a = np.array([6.0, 14.0, 8.0, 12.0])
    b = np.array([6.0, 10.0, 8.0, 8.0])
    return y, a, b


def fixture_ashrae():
    """ASHRAE 식(p=1) 손 계산 fixture, 길이 5.
    y = 10, 12, 9, 11, 8 (합 50, 평균 10), yhat = 9, 12, 10, 9, 8 -> e = 1, 0, -1, 2, 0.
    sum e = 2, sum e^2 = 6, n - p = 4.
    NMBE = 100 * 2 / (4 * 10) = 5%,  CV(RMSE) = 100 * sqrt(6/4) / 10 = 10*sqrt(1.5) ~ 12.247%.
    FEMP: MBE = 100 * sum(S - M) / sum(M) = 100 * (-2) / 50 = -4%,  Cv(RMSE) = 100 * sqrt(6/5) / 10 ~ 10.954%."""
    return np.array([10.0, 12.0, 9.0, 11.0, 8.0]), np.array([9.0, 12.0, 10.0, 9.0, 8.0])


def base_load(seed=0, n=HOURS_PER_YEAR, noise_sd=5.0):
    """시간 단위 합성 부하 y_t = 100 + 30 sin(2 pi t / 24) + eps_t, eps_t ~ N(0, noise_sd^2). 기본 1년(8760시간)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100.0 + 30.0 * np.sin(2.0 * np.pi * t / 24.0) + rng.normal(0.0, noise_sd, n)


def case_unbiased_noisy(y, sd=25.0, seed=0):
    """모델 A: 편향 없음, 시점별 잡음 큼. yhat = y + N(0, sd^2) (sd는 표준편차)."""
    rng = np.random.default_rng(10_000 + seed)
    return np.asarray(y, float) + rng.normal(0.0, sd, len(y))


def case_biased_steady(y, bias=0.08, sd=5.0, seed=0):
    """모델 B: 꾸준히 bias 비율만큼 모자람. yhat = (1 - bias) * y + N(0, sd^2)."""
    rng = np.random.default_rng(20_000 + seed)
    return (1.0 - bias) * np.asarray(y, float) + rng.normal(0.0, sd, len(y))


def month_edges(n_hours=HOURS_PER_YEAR):
    edges = np.concatenate([[0], np.cumsum(np.array(MONTH_DAYS) * 24)])
    if n_hours != edges[-1]:
        raise ValueError(f"월 합계는 {edges[-1]}시간(1년)짜리 기록에만 쓴다. 받은 길이 {n_hours}")
    return edges


def aggregate_monthly(y, yhat):
    """시간 단위 실제·예측을 달력 월 12개의 합계로 묶는다(전기 요금 고지서 12장에 해당)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = month_edges(len(y))
    ym = np.array([y[e[i]:e[i + 1]].sum() for i in range(12)])
    pm = np.array([yhat[e[i]:e[i + 1]].sum() for i in range(12)])
    return ym, pm


def solar_site(cloud_low, cloud_high, capacity=100.0, days=365, seed=0):
    """태양광 발전소의 시간별 발전량. 낮 6~18시에 사인 모양, 날마다 맑음 정도 U(cloud_low, cloud_high)를 곱한다. 밤은 0."""
    rng = np.random.default_rng(seed)
    h = np.arange(24)
    shape = np.clip(np.sin(np.pi * (h - 6) / 12.0), 0.0, None)
    shape[(h < 6) | (h > 18)] = 0.0
    k = rng.uniform(cloud_low, cloud_high, days)
    return (capacity * k[:, None] * shape[None, :]).ravel()


def solar_forecast(y, rel_sd, capacity=100.0, seed=0):
    """발전량에 비례하는 잡음을 넣은 예측. 0 아래와 정격 용량 위는 자른다. 밤(실제 0)은 0으로 맞힌다."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    return np.clip(y * (1.0 + rng.normal(0.0, rel_sd, len(y))), 0.0, capacity)
