"""모델 평가 3장 검증. `uv run pytest -q eval/ch03_r_squared` 로 실행한다."""

import numpy as np
import pytest
import torch

import eval_ch03_cases as cases
import eval_ch03_metrics as m
import eval_ch03_torch as tm


def test_first_screen_hand_numbers():
    xtr, ytr, xte, yte = cases.first_screen()
    b0, b1 = m.fit_line(xtr, ytr)
    assert (b0, b1) == (-2.0, 4.0)  # 기울기 40/10, 절편 6 - 4*2
    p = m.predict_line((b0, b1), xtr)
    assert (ytr - p).tolist() == [2.0, -1.0, -2.0, -1.0, 2.0]
    assert m.sse(ytr, p) == 14.0 and m.sst(ytr) == 174.0  # 36+25+4+9+100
    assert abs(m.r2(ytr, p) - (1 - 14 / 174)) < 1e-15
    q = m.predict_line((b0, b1), xte)
    assert m.sse(yte, q) == 1930.0  # 7^2+14^2+23^2+34^2
    assert m.sst(yte) == 849.0  # 평균 43.5, 18.5^2+7.5^2+5.5^2+20.5^2
    assert m.r2(yte, q) < 0


def test_anscombe_matches_papers_printed_summary():
    """Anscombe(1973) p.19: n=11, mean x 9.0, mean y 7.5, b1 0.5, y = 3 + 0.5x, 회귀 SS 27.50, 잔차 SS 13.75, R^2 0.667."""
    for x, y in cases.anscombe().values():
        assert len(x) == 11
        assert abs(x.mean() - 9.0) < 1e-12
        assert abs(np.sum((x - 9.0) ** 2) - 110.0) < 1e-9
        assert abs(y.mean() - 7.5) < 0.005
        b0, b1 = m.fit_line(x, y)
        assert abs(b1 - 0.5) < 0.0005 and abs(b0 - 3.0) < 0.005
        p = m.predict_line((b0, b1), x)
        assert abs(np.sum((p - y.mean()) ** 2) - 27.50) < 0.03
        assert abs(m.sse(y, p) - 13.75) < 0.03
        assert 0.666 <= m.r2(y, p) <= 0.667


def test_anscombe_set3_line_without_outlier():
    """원문: 관측 하나를 빼면 나머지는 y = 4 + 0.346x 가까이에 놓인다. 원 표 수치로 맞추면 기울기 0.3454."""
    x, y = cases.anscombe()["3"]
    keep = np.arange(11) != 2
    b0, b1 = m.fit_line(x[keep], y[keep])
    assert abs(b0 - 4.0) < 0.01 and abs(b1 - 0.346) < 0.001


def test_mean_prediction_scores_zero():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    assert abs(m.r2(y, np.full(40, y.mean()))) < 1e-15


def test_offset_model_hand_formula_and_perfect_corr():
    y, p = cases.case_offset(200, 5.0, seed=0)
    assert abs(m.squared_corr(y, p) - 1.0) < 1e-12
    assert abs(m.r2(y, p) - (1 - 200 * 25.0 / m.sst(y))) < 1e-9
    assert m.r2(y, p) < -20


def test_ols_with_intercept_r2_equals_r2corr_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.uniform(-3, 3, 30)
        y = rng.normal(0, 1) * x + rng.standard_t(3, 30)
        p = m.predict_line(m.fit_line(x, y), x)
        a = m.r2(y, p)
        assert abs(a - m.squared_corr(y, p)) < 1e-10
        assert -1e-12 <= a <= 1 + 1e-12


def test_no_intercept_counterexample_from_concept_card():
    """개념 카드 eval-r-squared: x=1,2,3, y=10,11,10, 원점을 지나는 직선이면 R^2 ≈ -68.6."""
    x, y = np.array([1.0, 2, 3]), np.array([10.0, 11, 10])
    p = m.predict_line(m.fit_line(x, y, intercept=False), x)
    assert abs(m.r2(y, p) - (-68.6429)) < 1e-4
    assert abs(m.squared_corr(y, p)) < 1e-12  # 정의가 갈린다: r^2 = 0
    assert abs(1 - m.sse(y, p) / np.sum(y**2) - 0.8554) < 1e-4  # 원점 기준 R^2


def test_extrapolation_high_train_negative_test():
    xtr, ytr, xte, yte = cases.case_extrapolation(seed=0)
    c = m.fit_line(xtr, ytr)
    assert m.r2(ytr, m.predict_line(c, xtr)) > 0.9
    assert m.r2(yte, m.predict_line(c, xte)) < 0
    assert m.squared_corr(yte, m.predict_line(c, xte)) > 0.98


def test_residual_diagnostics_catch_curvature_but_corr_does_not():
    xtr, ytr, _, _ = cases.case_extrapolation(seed=0)
    d = m.residual_diagnostics(xtr, ytr, m.predict_line(m.fit_line(xtr, ytr), xtr), bins=5)
    assert abs(d["resid_mean"]) < 1e-10 and abs(d["resid_x_corr"]) < 1e-10  # OLS 학습 데이터에서는 늘 0
    b = d["bin_resid_mean"]
    assert b[0] > 0 and b[2] < 0 and b[-1] > 0  # +, -, + : 곡선을 직선으로 맞췄다


def test_r2_and_rmse_rank_always_agree_on_same_test_set():
    rng = np.random.default_rng(5)
    for _ in range(300):
        y = rng.normal(size=20)
        a, b = y + rng.normal(size=20), y + rng.normal(size=20)
        assert (m.r2(y, a) > m.r2(y, b)) == (m.rmse(y, a) < m.rmse(y, b))


def test_constant_truth_raises():
    with pytest.raises(ValueError):
        m.r2(np.ones(5), np.arange(5.0))
    with pytest.raises(ValueError):
        tm.r2(torch.ones(5, dtype=torch.float64), torch.arange(5.0, dtype=torch.float64))


def test_numpy_matches_torch():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=257), rng.normal(size=257)
    p = 0.3 * y + rng.normal(size=257)
    assert abs(m.r2(y, p) - tm.r2(torch.tensor(y), torch.tensor(p))) < 1e-12
    assert abs(m.squared_corr(y, p) - tm.squared_corr(torch.tensor(y), torch.tensor(p))) < 1e-12
    yl = 2 * x + rng.normal(size=257)
    a, b = m.fit_line(x, yl), tm.fit_line(torch.tensor(x), torch.tensor(yl))
    assert abs(a[0] - b[0]) < 1e-10 and abs(a[1] - b[1]) < 1e-10
