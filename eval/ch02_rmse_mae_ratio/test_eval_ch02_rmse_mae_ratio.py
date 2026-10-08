"""모델 평가 2장 검증. `uv run pytest -q eval/ch02_rmse_mae_ratio` 로 실행한다."""

import math

import numpy as np
import pytest
import torch

import eval_ch02_cases as cases
import eval_ch02_metrics as m
import eval_ch02_torch as tm


def test_first_screen_hand_numbers():
    z = np.zeros(5)
    a, c, b = np.full(5, 2.0), np.array([1.0, 1, 2, 3, 3]), np.array([0.0, 0, 0, 0, 9])
    assert m.rmse_mae_ratio(z, a) == 1.0
    assert abs(m.rmse(z, c) - math.sqrt(24 / 5)) < 1e-12  # 1+1+4+9+9 = 24
    assert abs(m.rmse_mae_ratio(z, c) - math.sqrt(24 / 5) / 2) < 1e-12  # sqrt(4.8) / 2 = sqrt(1.2) = 1.10
    assert abs(m.rmse_mae_ratio(z, b) - math.sqrt(5)) < 1e-12  # sqrt(16.2) / 1.8 = sqrt(16.2 / 3.24) = sqrt(5) = 2.24, n=5의 상한


def test_ratio_always_within_bounds():
    rng = np.random.default_rng(0)
    for n in (2, 5, 30, 300):
        lo, hi = m.ratio_bounds(n)
        for _ in range(200):
            e = rng.standard_t(1.5, n) * rng.random(n)  # 아주 두꺼운 꼬리까지 섞는다
            r = m.rmse_mae_ratio(np.zeros(n), e)
            assert lo - 1e-12 <= r <= hi + 1e-12


def test_equal_sizes_give_one_and_single_spike_gives_sqrt_n():
    for n in (4, 100, 1000):
        y, p = cases.case_uniform_errors(n, size=0.7, seed=n)
        assert abs(m.rmse_mae_ratio(y, p) - 1.0) < 1e-12
        y, p = cases.case_single_spike(n, seed=n)
        assert abs(m.mae(y, p) - 1.0) < 1e-12
        assert abs(m.rmse_mae_ratio(y, p) - math.sqrt(n)) < 1e-9


def test_large_sample_ratios_near_reference():
    for dist in ("normal", "uniform", "laplace"):
        e = cases.sample_errors(dist, 1_000_000, seed=0)
        assert abs(m.rmse_mae_ratio(np.zeros_like(e), e) - m.ratio_reference(dist)) < 1e-2


def test_reference_values_analytic():
    assert abs(m.ratio_reference("normal") - 1.2533141373155) < 1e-12
    assert abs(m.ratio_reference("uniform") - 2 / math.sqrt(3)) < 1e-15
    assert abs(m.ratio_reference("laplace") - math.sqrt(2)) < 1e-15
    assert abs(m.ratio_reference("t", df=3) - math.pi / 2) < 1e-12  # t(3): sqrt(3) / (2 sqrt(3)/pi)
    assert m.ratio_reference("t", df=2) == math.inf
    assert abs(m.ratio_reference("t", df=1e6) - m.ratio_reference("normal")) < 1e-5  # 자유도가 크면 정규로 간다


def test_unit_mae_scaling_is_right():
    for dist, df in (("normal", None), ("laplace", None), ("uniform", None), ("t", 5)):
        e = cases.sample_errors(dist, 1_000_000, seed=1, df=df)
        assert abs(np.mean(np.abs(e)) - 1.0) < 1e-2


def test_variance_decomposition_matches_ratio():
    rng = np.random.default_rng(3)
    for _ in range(20):
        e = rng.standard_t(3, 77)
        assert abs(m.ratio_via_variance(e) - m.rmse_mae_ratio(np.zeros(77), e)) < 1e-12


def test_numpy_matches_torch_builtin_losses():
    rng = np.random.default_rng(1)
    y, yhat = rng.normal(size=257), rng.normal(size=257)
    assert abs(m.rmse_mae_ratio(y, yhat) - tm.rmse_mae_ratio(torch.tensor(y), torch.tensor(yhat))) < 1e-12


def test_two_level_errors_mimic_normal_ratio():
    y, p = cases.case_two_level(1000, 2 / math.pi, seed=0)
    assert abs(m.rmse_mae_ratio(y, p) - math.sqrt(1000 / 637)) < 1e-12
    assert abs(m.rmse_mae_ratio(y, p) - m.ratio_reference("normal")) < 1e-3  # 정규가 아닌데도 거의 같다


def test_zero_mae_raises():
    with pytest.raises(ValueError):
        m.rmse_mae_ratio(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        tm.rmse_mae_ratio(torch.ones(3), torch.ones(3))


def test_signed_mean_denominator_is_a_trap():
    # 분모에 mean(e)를 쓰면 +/-가 상쇄돼 0에 가까워지고 '비율'이 폭주한다.
    e = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0 + 1e-6])
    wrong = np.sqrt(np.mean(e**2)) / abs(np.mean(e))
    assert m.rmse_mae_ratio(np.zeros(6), e) < 1.0 + 1e-6
    assert wrong > 1e6
