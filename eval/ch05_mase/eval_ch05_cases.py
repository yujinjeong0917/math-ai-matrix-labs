"""모델 평가 5장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다.

시계열은 (학습 구간, 평가 구간) 두 1차원 배열로 돌려준다(fixture_small, case_nearly_flat은 예측까지 3개).
MASE의 분모는 학습 구간만으로 정해지므로 학습과 평가를 처음부터 나눠서 만든다.
"""

import numpy as np


def first_screen():
    """첫 화면 손 계산 예. 도서관 하루 방문자 수 5일(학습)과 다음 2일(평가)."""
    # 하루 변화 |2|, |1|, |2|, |1|의 평균 1.5가 MASE의 분모가 된다
    train = np.array([1000.0, 1002.0, 1001.0, 1003.0, 1002.0])
    test = np.array([1003.0, 1005.0])
    return train, test


def fixture_small():
    """손 계산 fixture. 학습 6개, 평가 2개. 분모 = (2+1+4+2+1)/5 = 2, 오차 1, -2 -> MASE 3/4."""
    return np.array([3.0, 5.0, 4.0, 8.0, 6.0, 7.0]), np.array([9.0, 6.0]), np.array([8.0, 8.0])


def case_random_walk_level(level=1000.0, sigma=5.0, n_train=200, n_test=100, seed=0):
    """수준 약 1000, 하루 변화 ~ N(0, 5^2)(표준편차 5)인 랜덤워크. 앞 200개 학습, 뒤 100개 평가."""
    rng = np.random.default_rng(seed)
    # 누적합 = 랜덤워크. 마지막 값이 최선의 예측이라 naive가 강한 기준선이다. 학습·평가를 한 경로에서 자르므로 이어진다
    y = level + np.cumsum(rng.normal(0.0, sigma, n_train + n_test))
    return y[:n_train], y[n_train:]


def case_iid_level(level=20.0, sigma=5.0, n_train=200, n_test=100, seed=0):
    """평균 20 근처에서 날마다 독립적으로 흔들리는 계열(정규, 표준편차 5). 0 아래는 0으로 자른다."""
    rng = np.random.default_rng(seed)
    # 날마다 독립이라 평균 예측이 유리하고 naive는 불리한 구조. 0 아래로는 자른다(판매량이 음수가 될 수 없으므로)
    y = np.maximum(level + rng.normal(0.0, sigma, n_train + n_test), 0.0)
    return y[:n_train], y[n_train:]


def case_intermittent(p_zero=0.2, n=120, lam=4.0, seed=0):
    """간헐 수요. 4장(eval/ch04_mape/eval_ch04_cases.py case_intermittent_demand)에서 복사.
    판매가 있는 날은 1 + Poisson(lam)개, 그날이 '판매 없음'일 확률은 p_zero."""
    rng = np.random.default_rng(seed)
    y = 1.0 + rng.poisson(lam, n)
    y[rng.random(n) < p_zero] = 0.0
    return y


def case_nearly_flat(n_train=100, jump=1.0, n_test=10, err=1.0):
    """학습 구간이 거의 평평: 100일 중 딱 한 번 jump만큼 오르고 나머지는 그대로. 평가 오차는 모두 err."""
    train = np.full(n_train, 50.0)
    # 중간(인덱스 50)부터 한 번만 jump만큼 오른다. 그래서 일별 변화 99개 중 하나만 0이 아니다(분모 = jump/99)
    train[n_train // 2:] += jump
    # 평가 정답은 마지막 학습 값 그대로. 예측은 여기에 err를 더해 오차를 모두 err로 만든다
    test = np.full(n_test, train[-1])
    return train, test, test + err


def case_weekly(n_weeks_train=20, n_weeks_test=4, sigma=3.0, level=100.0, seed=0):
    """요일 효과가 큰 하루 판매량(주기 7). 월~일 요일 효과 -10, -10, -5, 0, 5, 30, 30, 잡음은 표준편차 sigma."""
    rng = np.random.default_rng(seed)
    # 요일 효과(월~일 순, 합 40이라 평균 수준은 level보다 약 5.7 높다). 인덱스 i가 i번째 요일
    pattern = np.array([-10.0, -10.0, -5.0, 0.0, 5.0, 30.0, 30.0])
    n = 7 * (n_weeks_train + n_weeks_test)
    # tile로 주간 패턴을 주 수만큼 반복. 학습 20주 = 140일 뒤에 평가 4주 = 28일이 이어진다
    y = level + np.tile(pattern, n_weeks_train + n_weeks_test) + rng.normal(0.0, sigma, n)
    return y[: 7 * n_weeks_train], y[7 * n_weeks_train:]
