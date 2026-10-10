"""모델 평가 5장 검증. `uv run pytest -q eval/ch05_mase` 로 실행한다."""

from fractions import Fraction

import numpy as np
import pytest
import torch

import eval_ch05_cases as cases
import eval_ch05_metrics as m
import eval_ch05_torch as tm


def test_fixture_exact_fraction():
    tr, te, p = cases.fixture_small()
    assert m.naive_mae_in_sample(tr) == 2.0  # (2+1+4+2+1)/5
    assert m.scaled_errors(tr, te, p).tolist() == [0.5, -1.0]  # 부호가 남는다
    assert Fraction(m.mase(tr, te, p)).limit_denominator(100) == Fraction(3, 4)


def test_first_screen_hand_numbers():
    tr, te = cases.first_screen()
    assert m.naive_mae_in_sample(tr) == 1.5  # (2+1+2+1)/4
    a = m.mean_forecast(tr, 2)
    assert abs(a[0] - 1001.6) < 1e-9
    assert abs(m.mase(tr, te, a) - 1.6) < 1e-9  # 평균 오차 2.4 / 1.5
    assert abs(m.mase(tr, te, m.naive_forecast(tr, 2)) - 4 / 3) < 1e-12  # (1+3)/2 / 1.5
    assert m.mape(te, a) < 0.25


def test_in_sample_naive_is_exactly_one():
    for s in range(10):
        tr, _ = cases.case_random_walk_level(seed=s)
        assert abs(m.mase(tr, tr[1:], tr[:-1]) - 1.0) < 1e-12
        tr7, _ = cases.case_weekly(seed=s)
        assert abs(m.mase(tr7, tr7[7:], tr7[:-7], lag=7) - 1.0) < 1e-12


def test_scale_and_shift_invariance():
    tr, te = cases.case_random_walk_level(seed=3)
    p = m.mean_forecast(tr, len(te))
    base = m.mase(tr, te, p)
    assert abs(m.mase(10 * tr, 10 * te, 10 * p) - base) < 1e-9
    assert abs(m.mase(tr - 1000, te - 1000, p - 1000) - base) < 1e-9  # MAPE와 달리 수준 이동에도 그대로
    assert abs(m.mape(tr, tr + 1) - m.mape(tr - 900, tr - 899)) > 0.1


def test_constant_training_raises_numpy_and_torch():
    with pytest.raises(ValueError):
        m.mase(np.full(10, 7.0), [7.0, 8.0], [7.0, 7.0])
    with pytest.raises(ValueError):
        tm.mase(torch.full((10,), 7.0, dtype=torch.float64), torch.tensor([7.0, 8.0], dtype=torch.float64),
                torch.tensor([7.0, 7.0], dtype=torch.float64))


def test_zeros_in_data_are_fine():
    y = cases.case_intermittent(0.2, seed=5000)
    tr, te = y[:80], y[80:]
    assert (tr == 0).any() and (te == 0).any()
    with pytest.raises(ValueError):
        m.mape(te, np.full(len(te), 3.0))
    assert np.isfinite(m.mase(tr, te, np.full(len(te), 3.0)))
    assert abs(m.mase([0, 2, 1, 0, 3], [0, 2], [1, 1]) - 4 / 7) < 1e-12


def test_within_one_series_mase_ranks_like_mae():
    rng = np.random.default_rng(0)
    for s in range(50):
        tr, te = cases.case_iid_level(seed=s)
        p1, p2 = te + rng.normal(0, 3, len(te)), te + rng.normal(0, 3, len(te))
        assert (m.mase(tr, te, p1) < m.mase(tr, te, p2)) == (m.mae(te, p1) < m.mae(te, p2))
        assert abs(m.mase(tr, te, p1) - m.mae(te, p1) / m.naive_mae_in_sample(tr)) < 1e-12


def test_horizon_naive_mase_close_to_sqrt_h():
    for h in (1, 5, 20):
        one, hs = [], []
        for s in range(100):
            tr, te = cases.case_random_walk_level(seed=2000 + s)
            p = m.naive_rolling(tr, te, lag=h)
            one.append(m.mase(tr, te, p)); hs.append(m.mase(tr, te, p, lag=h))
        assert abs(np.mean(one) / np.sqrt(h) - 1) < 0.08
        assert abs(np.mean(hs) - 1) < 0.1


def test_naive_rolling_and_fixed():
    tr, te = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])
    assert m.naive_rolling(tr, te).tolist() == [3.0, 4.0]
    assert m.naive_rolling(tr, te, lag=2).tolist() == [2.0, 3.0]
    assert m.naive_forecast(tr, 2).tolist() == [3.0, 3.0]


def test_seasonal_denominator_matches_fpp3_formula():
    tr, _ = cases.case_weekly(seed=1)
    T, mm = len(tr), 7
    manual = sum(abs(tr[t] - tr[t - mm]) for t in range(mm, T)) / (T - mm)
    assert abs(m.naive_mae_in_sample(tr, lag=7) - manual) < 1e-12


def test_nearly_flat_training_blows_up():
    tr, te, p = cases.case_nearly_flat(jump=1.0)
    assert abs(m.mase(tr, te, p) - 99.0) < 1e-9  # 분모 1/99


def test_numpy_matches_torch():
    rng = np.random.default_rng(7)
    tr = 50 + np.cumsum(rng.normal(0, 2, 257)); te = tr[-1] + np.cumsum(rng.normal(0, 2, 64)); p = te + rng.normal(0, 3, 64)
    for lag in (1, 7):
        a = m.mase(tr, te, p, lag)
        b = tm.mase(torch.tensor(tr), torch.tensor(te), torch.tensor(p), lag)
        assert abs(a - b) < 1e-12


def test_web_chapter_exercise_answers():
    # 연습 1: 분모 (3+2+1)/3 = 2, 오차 평균 2.5 -> 1.25
    tr, te, p = np.array([10.0, 13, 11, 12]), np.array([14.0, 15]), np.array([12.0, 12])
    assert m.mase(tr, te, p) == 1.25
    # 연습 2: 1000을 더해도 MASE 그대로, MAPE 17.1% -> 0.25%
    assert m.mase(tr + 1000, te + 1000, p + 1000) == 1.25
    assert round(m.mape(te, p), 1) == 17.1 and round(m.mape(te + 1000, p + 1000), 2) == 0.25
    # 연습 3: lag 1 분모 65/13 = 5, lag 7 분모 5/7
    w = np.array([10.0, 10, 10, 10, 10, 30, 30, 11, 9, 10, 10, 11, 31, 29])
    assert m.naive_mae_in_sample(w, 1) == 5.0
    assert abs(m.naive_mae_in_sample(w, 7) - 5 / 7) < 1e-12
