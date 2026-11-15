"""모델 평가 1장의 합성 데이터. 정답 y는 0으로 두고 예측값 = 오차로 쓴다(오차만 중요하므로).

각 함수는 (y, yhat) 쌍을 돌려준다. 둘 다 모양 (n,)의 float 배열이고, y는 전부 0이다.
모델 A(정규 오차)와 모델 B(드문 큰 실수)를 비교해 MAE와 RMSE의 순위가 갈리는 지점을 만든다.
"""

import numpy as np


def case_gaussian_errors(n, sigma=1.0, seed=0):
    """모델 A: 오차가 평균 0, 표준편차 sigma인 정규분포. 작은 실수가 고르게 퍼져 있다."""
    # 시드를 인자로 받아 호출마다 같은 표본을 재현한다(함수마다 독립된 생성기)
    rng = np.random.default_rng(seed)
    return np.zeros(n), rng.normal(0.0, sigma, n)


def case_rare_large_errors(n, n_big, big=20.0, small=0.5, seed=0):
    """모델 B: n_big개는 크기 big, 나머지는 크기 small. 부호와 위치만 무작위.

    크기가 고정이라 MAE·RMSE는 시드와 무관하게 정해진다(손 계산과 정확히 일치).
    """
    rng = np.random.default_rng(seed)
    # (n,) 오차 크기. 기본은 작은 실수 small(0.5)
    mag = np.full(n, small)
    # 중복 없이 n_big개 위치를 골라 큰 실수(big)로 바꾼다. 개수가 정확히 n_big로 고정된다
    mag[rng.choice(n, size=n_big, replace=False)] = big
    # 부호는 MAE·RMSE에 영향이 없다(절댓값·제곱만 쓰므로). 편향 지표(6장)에서만 의미가 있다
    sign = rng.choice([-1.0, 1.0], size=n)
    return np.zeros(n), sign * mag


def case_random_rare_large_errors(n, p_big=0.01, big=20.0, small=0.5, seed=0):
    """모델 B의 무작위판: 각 오차가 확률 p_big로 크기 big, 아니면 small.

    큰 실수의 개수 자체가 테스트 세트마다 달라진다. 순위 안정성 실험에 쓴다.
    """
    rng = np.random.default_rng(seed)
    # 각 원소가 독립적으로 확률 p_big로 큰 실수. 큰 실수 개수는 이항분포 Bin(n, p_big)를 따른다
    mag = np.where(rng.random(n) < p_big, big, small)
    sign = rng.choice([-1.0, 1.0], size=n)
    return np.zeros(n), sign * mag
