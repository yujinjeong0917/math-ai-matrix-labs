"""모델 평가 1장 검증. `uv run pytest -q eval/ch01_mae_rmse` 로 실행한다."""

import numpy as np
import torch

import eval_ch01_cases as cases
import eval_ch01_metrics as m
import eval_ch01_torch as tm


def test_first_screen_hand_numbers():
    z = np.zeros(5)
    a, b = np.full(5, 2.0), np.array([0.0, 0, 0, 0, 9])
    assert m.mae(z, a) == 2.0 and m.rmse(z, a) == 2.0
    assert abs(m.mae(z, b) - 1.8) < 1e-12  # 9 / 5
    assert abs(m.rmse(z, b) - np.sqrt(81 / 5)) < 1e-12  # sqrt(16.2) = 4.02
    assert m.mae(z, b) < m.mae(z, a) and m.rmse(z, b) > m.rmse(z, a)  # 두 지표가 반대 모델을 고른다


def test_rare_large_errors_match_hand_calculation():
    y, p = cases.case_rare_large_errors(1000, 10, seed=3)
    assert abs(m.mae(y, p) - 0.695) < 1e-12  # (990*0.5 + 10*20)/1000
    assert abs(m.rmse(y, p) - np.sqrt((990 * 0.25 + 10 * 400) / 1000)) < 1e-12  # 2.061


def test_gaussian_case_is_near_expected_values():
    y, p = cases.case_gaussian_errors(1_000_000, seed=0)
    assert abs(m.mae(y, p) - np.sqrt(2 / np.pi)) < 5e-3
    assert abs(m.rmse(y, p) - 1.0) < 5e-3


def test_numpy_matches_torch_builtin_losses():
    rng = np.random.default_rng(1)
    y, yhat = rng.normal(size=257), rng.normal(size=257)
    yt, pt = torch.tensor(y), torch.tensor(yhat)
    assert abs(m.mae(y, yhat) - tm.mae(yt, pt)) < 1e-12
    assert abs(m.rmse(y, yhat) - tm.rmse(yt, pt)) < 1e-12


def test_equal_error_sizes_give_mae_equal_rmse_and_bounds_hold():
    e = np.array([1.5, -1.5, 1.5, -1.5])
    assert abs(m.mae(np.zeros(4), e) - m.rmse(np.zeros(4), e)) < 1e-12
    rng = np.random.default_rng(2)
    for _ in range(50):
        assert m.rmse_bounds_hold(rng.standard_t(2, size=30))


def test_best_constant_is_median_for_mae_and_mean_for_mse():
    y = np.array([1.0, 2, 3, 4, 10])
    c_mae, step = m.best_constant(y, "mae")
    c_mse, _ = m.best_constant(y, "mse")
    assert abs(c_mae - 3.0) <= step and abs(c_mse - 4.0) <= step
    assert abs(tm.fit_constant(y, "l1") - 3.0) < 1e-3
    assert abs(tm.fit_constant(y, "l2") - 4.0) < 1e-6


def test_even_n_mae_minimizer_is_anywhere_between_middle_values():
    # 짝수 개면 가운데 두 값 사이 어디든 MAE가 같다. 격자 답은 그 구간 안에만 있으면 된다.
    y = np.array([1.0, 2, 3, 4])
    c, step = m.best_constant(y, "mae")
    assert 2.0 - step <= c <= 3.0 + step
    assert torch.median(torch.tensor(y)).item() == 2.0  # torch.median은 아래쪽 가운데 값을 돌려준다


def test_boundary_where_rankings_flip():
    """n=1000에서 A의 기대값(MAE sqrt(2/pi), RMSE 1)과 비교한 경계: RMSE는 k=1|2, MAE는 k=15|16."""
    n, a_mae, a_rmse = 1000, np.sqrt(2 / np.pi), 1.0
    winners = {}
    for k in (1, 2, 15, 16):
        y, p = cases.case_rare_large_errors(n, k)
        winners[k] = (m.mae(y, p) < a_mae, m.rmse(y, p) < a_rmse)
    assert winners[1] == (True, True)
    assert winners[2] == (True, False)
    assert winners[15] == (True, False)
    assert winners[16] == (False, False)
