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
    # (x=0..4, y=x^2) 평균 x=2, 평균 y=6, Sxy=40, Sxx=10. 값이 정수라 정확 비교가 가능하다
    assert (b0, b1) == (-2.0, 4.0)  # 기울기 40/10, 절편 6 - 4*2
    p = m.predict_line((b0, b1), xtr)
    assert (ytr - p).tolist() == [2.0, -1.0, -2.0, -1.0, 2.0]
    assert m.sse(ytr, p) == 14.0 and m.sst(ytr) == 174.0  # 36+25+4+9+100
    assert abs(m.r2(ytr, p) - (1 - 14 / 174)) < 1e-15
    q = m.predict_line((b0, b1), xte)
    # 테스트 x=5..8: 직선 예측 18,22,26,30 vs 정답 25,36,49,64. 잔차 제곱합 49+196+529+1156 = 1930 > SST 849이므로 R^2 < 0
    assert m.sse(yte, q) == 1930.0  # 7^2+14^2+23^2+34^2
    assert m.sst(yte) == 849.0  # 평균 43.5, 18.5^2+7.5^2+5.5^2+20.5^2
    assert m.r2(yte, q) < 0


# 표의 수치가 논문 요약과 맞는지 확인. 표의 y가 소수점 둘째 자리로 반올림돼 있어 허용 오차를 둔다
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


# 평가 데이터의 평균으로 찍으면 정의상 SSE = SST이므로 R^2 = 0
def test_mean_prediction_scores_zero():
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    assert abs(m.r2(y, np.full(40, y.mean()))) < 1e-15


# 오프셋 모델: r^2 = 1인데 R^2 = 1 - n shift^2 / SST (SST는 약 n = 200이므로 약 -24)
def test_offset_model_hand_formula_and_perfect_corr():
    y, p = cases.case_offset(200, 5.0, seed=0)
    assert abs(m.squared_corr(y, p) - 1.0) < 1e-12
    assert abs(m.r2(y, p) - (1 - 200 * 25.0 / m.sst(y))) < 1e-9
    assert m.r2(y, p) < -20


# 절편 있는 OLS를 학습 데이터에서 재면 R^2 = r^2이고 [0, 1] 안이다. 기울기 부호는 상관 제곱에서 사라진다
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


# [0,5]에서 학습한 직선은 학습 R^2가 높지만 [5,10]에서는 R^2 < 0. 그래도 상관 제곱은 0.98 넘게 높아 둘이 갈린다
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
    # 5개 구간 중 처음·가운데·끝을 본다. y=x^2는 아래로 볼록이라 양끝 잔차가 +
    assert b[0] > 0 and b[2] < 0 and b[-1] > 0  # +, -, + : 곡선을 직선으로 맞췄다


# 같은 테스트 세트에서는 SST가 공통이라 R^2 순위와 RMSE 순위가 항상 같다(다른 테스트 세트끼리는 비교 불가)
def test_r2_and_rmse_rank_always_agree_on_same_test_set():
    rng = np.random.default_rng(5)
    for _ in range(300):
        y = rng.normal(size=20)
        a, b = y + rng.normal(size=20), y + rng.normal(size=20)
        assert (m.r2(y, a) > m.r2(y, b)) == (m.rmse(y, a) < m.rmse(y, b))


# SST = 0이면 NumPy·torch 모두 ValueError
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
