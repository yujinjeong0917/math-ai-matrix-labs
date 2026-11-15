"""모델 평가 4장 검증. `uv run pytest -q eval/ch04_mape` 로 실행한다."""

import numpy as np
import pytest
import torch

import eval_ch04_cases as cases
import eval_ch04_metrics as m
import eval_ch04_torch as tm


def test_first_screen_hand_numbers():
    y, p = cases.first_screen()
    # 정답 100, 10, 1에 모두 +2 오차. MAE는 2로 같아도 MAPE는 74%
    assert m.ape(y, p).tolist() == [2.0, 20.0, 200.0]  # 2/100, 2/10, 2/1
    assert abs(m.mape(y, p) - 74.0) < 1e-12  # (2 + 20 + 200) / 3
    assert m.mae(y, p) == 2.0


def test_near_zero_case():
    y, p = cases.case_near_zero()
    # 정답 0.01은 1/0.01 = 100배 = 10000%, 100은 1%. 평균 5000.5%가 한 점에 끌려간 값
    assert np.allclose(m.ape(y, p), [10000.0, 1.0])
    assert abs(m.mape(y, p) - 5000.5) < 1e-9


def test_zero_policies():
    y, p = cases.case_with_zero()
    with pytest.raises(ValueError):
        m.mape(y, p, zero="raise")
    # [0,1] vs [1,1]에서 0인 시점을 빼면 남는 점은 오차 0이라 0%
    assert m.mape(y, p, zero="drop") == 0.0
    # sklearn 1.9.1 식: |1-0| / max(0, eps) = 1/eps, 둘째 점 0 -> 평균 1/(2 eps). 우리는 % 단위라 100배.
    assert m.mape(y, p, zero="eps") / 100.0 == 1.0 / (2.0 * m.EPS)
    assert np.isinf(m.ape(y, p)[0])


# 정답 [1, 0, 2.4, 7]의 0이 0.1/EPS = 4.5e14를 만들어 평균 1.1259e14를 지배한다. 부동소수 ==로 비교하는 이유는 같은 연산을 같은 순서로 쓰기 때문
def test_eps_policy_reproduces_sklearn_doc_example_exactly():
    yd, pd = cases.case_sklearn_doc_example()
    assert m.mape(yd, pd, zero="eps") / 100.0 == 112589990684262.48  # sklearn 1.9.1 문서 출력
    assert tm.mape(torch.tensor(yd), torch.tensor(pd), zero="eps") / 100.0 == 112589990684262.48


# 정답에 0이 없으면 세 정책이 같은 값을 내야 한다
def test_eps_matches_raise_when_no_zero():
    rng = np.random.default_rng(2)
    y = rng.uniform(1, 50, 100)
    p = y + rng.normal(0, 2, 100)
    assert abs(m.mape(y, p, "eps") - m.mape(y, p, "raise")) < 1e-12
    assert abs(m.mape(y, p, "drop") - m.mape(y, p, "raise")) < 1e-12


# eps 정책의 함정: 정답에 0이 하나라도 있으면 늘 0으로 찍는 예측이 다른 상수 예측보다 좋아진다
def test_zero_forecast_beats_everything_under_eps_once_a_zero_appears():
    y = np.array([0.0, 4.0, 5.0, 6.0])
    z = m.mape(y, np.zeros(4), "eps")
    assert z == 75.0  # 0/eps = 0, 나머지 셋은 100%
    assert m.mape(y, np.full(4, 5.0), "eps") > 1e15


# 정답을 고정하면 대칭(50%, 50%), 예측을 고정하면 정답이 작은 쪽의 %가 크다. 0 이상 예측이 모자랄 때 APE <= 100%
def test_asymmetry_two_views_and_bounds():
    ya, pa = cases.case_asymmetry_fixed_actual()
    assert m.ape(ya, pa).tolist() == [50.0, 50.0]  # 정답 고정: 대칭
    yf, pf = cases.case_asymmetry_fixed_forecast()
    assert np.allclose(m.ape(yf, pf), [100 / 3, 100.0])  # 예측 고정: 정답이 작은 쪽이 더 큰 %
    rng = np.random.default_rng(3)
    y = rng.uniform(0.1, 100, 10_000)
    under = y * rng.uniform(0, 1, 10_000)  # 0 <= 예측 < 정답
    over = y * rng.uniform(1, 50, 10_000)
    assert m.ape(y, under).max() <= 100.0 + 1e-12
    assert m.ape(y, over).max() > 1000.0
    assert m.ape_bounds_demo()[0.0] == 100.0


# 정답과 예측에 같은 100을 더하면 MAE는 그대로이고 MAPE는 10배 넘게 줄어든다(분모만 커짐)
def test_level_shift_keeps_mae_but_moves_mape():
    y, p = cases.case_level_shift()
    a, b = m.mape(y, p), m.mape(y + 100, p + 100)
    assert abs(m.mae(y, p) - m.mae(y + 100, p + 100)) < 1e-9
    assert b < a / 10


def test_temperature_units_change_mape_not_relative_mae():
    c, f = cases.case_temperature()
    assert abs(m.mae(c, f) - 1.2) < 1e-12
    assert abs(m.mae(cases.to_fahrenheit(c), cases.to_fahrenheit(f)) - 1.2 * 1.8) < 1e-12
    vals = [m.mape(c, f), m.mape(cases.to_fahrenheit(c), cases.to_fahrenheit(f)), m.mape(cases.to_kelvin(c), cases.to_kelvin(f))]
    # 섭씨 MAPE는 100%(APE 33.3, 200, 200, 16.7, 50의 평균). 같은 예보인데 단위만 바꿔 화씨, 켈빈으로 가면 값이 줄어든다
    assert abs(vals[0] - 100.0) < 1e-9 and vals[0] > vals[1] > vals[2]


# 가중 중앙값 해가 격자 탐색 최솟값(20001점)보다 나쁘지 않아야 한다. 오른쪽 꼬리가 긴 자료에서는 MAPE 최적 상수 < 중앙값 < 평균
def test_best_constant_mape_is_weighted_median_and_matches_grid():
    for y in (np.array([50.0, 150.0]), cases.case_skewed_sales(seed=0), cases.case_intermittent_demand(0.0, seed=0)):
        c = m.best_constant_mape(y)
        grid = np.linspace(y.min(), y.max(), 20001)
        vals = [m.mape(y, np.full(len(y), g)) for g in grid]
        assert m.mape(y, np.full(len(y), c)) <= min(vals) + 1e-9
    y = cases.case_skewed_sales(seed=0)
    assert m.best_constant_mape(y) < np.median(y) < y.mean()


def test_numpy_matches_torch():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.5, 20.0, 257)
    p = y + rng.normal(0.0, 1.0, 257)
    # 10개마다 하나를 0으로 만들어 0 처리 정책을 시험
    yz = y.copy(); yz[::10] = 0.0
    assert abs(m.mape(y, p) - tm.mape(torch.tensor(y), torch.tensor(p))) < 1e-12
    assert abs(m.mape(yz, p, "drop") - tm.mape(torch.tensor(yz), torch.tensor(p), "drop")) < 1e-12
    e1, e2 = m.mape(yz, p, "eps"), tm.mape(torch.tensor(yz), torch.tensor(p), "eps")
    # eps 정책 값은 1e14 이상으로 커서 절대가 아니라 상대 오차로 비교
    assert abs(e1 - e2) / e1 < 1e-12
    assert abs(m.mae(y, p) - tm.mae(torch.tensor(y), torch.tensor(p))) < 1e-12
    with pytest.raises(ValueError):
        tm.mape(torch.tensor(yz), torch.tensor(p), "raise")


def test_intermittent_series_has_no_zero_when_p_is_zero():
    for s in range(20):
        assert (cases.case_intermittent_demand(0.0, seed=s) >= 1).all()
