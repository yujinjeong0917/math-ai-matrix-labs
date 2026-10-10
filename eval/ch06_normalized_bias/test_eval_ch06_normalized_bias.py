"""모델 평가 6장 검증. `uv run pytest -q eval/ch06_normalized_bias` 로 실행한다."""

import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import eval_ch06_cases as cases
import eval_ch06_metrics as m
import eval_ch06_torch as tm


def test_first_screen_hand_numbers():
    y, a, b = cases.first_screen()
    # MAE, RMSE가 둘 다 2로 같다
    assert m.mae(y, a) == m.mae(y, b) == 2.0
    assert m.rmse(y, a) == m.rmse(y, b) == 2.0
    # 분모 n: NMBE는 0%와 20%, CV(RMSE)는 둘 다 20%
    assert m.nmbe(y, a, p=0) == 0.0 and abs(m.nmbe(y, b, p=0) - 20.0) < 1e-12
    assert abs(m.cv_rmse(y, a, p=0) - 20.0) < 1e-12 and abs(m.cv_rmse(y, b, p=0) - 20.0) < 1e-12
    # B에 2를 더하면 오차가 모두 0, A는 평균 오차가 0이라 더할 것이 없다
    assert m.rmse(y, m.offset_correct(y, b)) == 0.0
    assert m.rmse(y, m.offset_correct(y, a)) == 2.0


def test_ashrae_fixture_exact():
    y, p = cases.fixture_ashrae()
    assert m.errors(y, p).tolist() == [1.0, 0.0, -1.0, 2.0, 0.0]
    assert abs(m.nmbe(y, p, p=1) - 5.0) < 1e-12  # 100*2/(4*10)
    assert abs(m.cv_rmse(y, p, p=1) - 10.0 * math.sqrt(1.5)) < 1e-12  # 100*sqrt(6/4)/10
    assert Fraction(m.cv_rmse(y, p, p=1) ** 2).limit_denominator(1000) == 150


def test_femp_fixture_sign_and_denominator():
    y, p = cases.fixture_ashrae()
    assert abs(m.nmbe(y, p, convention="femp") - (-4.0)) < 1e-12  # 100*sum(S-M)/sum(M) = -200/50
    assert abs(m.cv_rmse(y, p, p=0) - 100 * math.sqrt(6 / 5) / 10) < 1e-12


def test_conventions_have_opposite_signs():
    y = cases.base_load(seed=1)
    for pred in (cases.case_biased_steady(y, seed=1), cases.case_unbiased_noisy(y, seed=1) + 3.0):
        a = m.nmbe(y, pred, p=0)
        f = m.nmbe(y, pred, convention="femp")
        s = m.nmbe(y, pred, convention="sfa", norm=float(np.max(y)))
        assert abs(a + f) < 1e-9  # FEMP = -ASHRAE(p=0) 정확히
        assert np.sign(a) == -np.sign(f) == -np.sign(s) != 0


def test_mse_decomposition_identity_and_builtin():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(50, 10, 200)
        p = y + rng.normal(rng.normal(0, 3), rng.uniform(0.5, 5), 200)
        d = m.mse_decomposition(y, p)
        assert abs(d["bias_sq"] + d["var"] - d["mse"]) < 1e-12 * max(1.0, d["mse"])
        builtin = F.mse_loss(torch.tensor(p), torch.tensor(y)).item()
        assert abs(builtin - d["mse"]) < 1e-12 * max(1.0, d["mse"])


def test_cv_nmbe_pythagoras_only_with_n_denominator():
    y = cases.base_load(seed=2)
    p = cases.case_biased_steady(y, seed=2)
    d = m.mse_decomposition(y, p)
    shake = 100 * math.sqrt(d["var"]) / y.mean()
    assert abs(m.cv_rmse(y, p, p=0) ** 2 - (m.nmbe(y, p, p=0) ** 2 + shake ** 2)) < 1e-9
    # 오차가 모두 같으면 분모 n에서는 |NMBE| = CV, p=1에서는 |NMBE| > CV
    ye = np.full(12, 100.0)
    pe = ye - 1.0
    assert abs(m.nmbe(ye, pe, p=0) - m.cv_rmse(ye, pe, p=0)) < 1e-12
    assert m.nmbe(ye, pe, p=1) > m.cv_rmse(ye, pe, p=1)
    assert abs(m.nmbe(ye, pe, p=1) - 12 / 11) < 1e-12 and abs(m.cv_rmse(ye, pe, p=1) - math.sqrt(12 / 11)) < 1e-12


def test_offset_correction_zeroes_nmbe_and_lowers_cv():
    y = cases.base_load(seed=3)
    p = cases.case_biased_steady(y, seed=3)
    pc = m.offset_correct(y, p)
    assert abs(m.nmbe(y, pc)) < 1e-9
    assert m.cv_rmse(y, pc) < m.cv_rmse(y, p)
    # 흔들림 조각은 그대로
    assert abs(m.mse_decomposition(y, pc)["var"] - m.mse_decomposition(y, p)["var"]) < 1e-9
    # 곱하기 꼴 편향이라 잡음 바닥(5/100 = 5%)까지는 못 내려가고 하루 주기 잔차가 남는다
    assert m.cv_rmse(y, pc) > 100 * 5.0 / y.mean()


def test_monthly_aggregation():
    y = cases.base_load(seed=4)
    p = cases.case_unbiased_noisy(y, seed=4)
    ym, pm = cases.aggregate_monthly(y, p)
    assert len(ym) == 12 and abs(ym.sum() - y.sum()) < 1e-6 and abs(pm.sum() - p.sum()) < 1e-6
    assert cases.month_edges()[-1] == 8760 and cases.month_edges()[2] == (31 + 28) * 24
    with pytest.raises(ValueError):
        cases.aggregate_monthly(y[:720], p[:720])
    # 잡음은 월 합계에서 상쇄되고, 편향은 그대로 남는다
    b = cases.case_biased_steady(y, seed=4)
    bm = cases.aggregate_monthly(y, b)[1]
    assert m.cv_rmse(ym, pm) < m.cv_rmse(y, p) / 10
    assert abs(m.nmbe(ym, bm, p=0) - m.nmbe(y, b, p=0)) < 1e-9


def test_dof_guard():
    with pytest.raises(ValueError):
        m.nmbe([100.0], [90.0], p=1)
    with pytest.raises(ValueError):
        m.cv_rmse([100.0], [90.0], p=1)


def test_ashrae_check_limits():
    y = np.full(12, 100.0)
    assert m.ashrae_check(y, y - 4.0, "monthly")["pass"]  # NMBE 100*48/(11*100)=4.36%
    assert not m.ashrae_check(y, y - 4.7, "monthly")["pass"]  # 5.13% > 5
    assert not m.ashrae_check(y, y + 4.7, "monthly")["pass"]  # 과대예측도 같은 크기면 떨어진다(|NMBE|, [해석])
    assert m.ASHRAE_2002_LIMITS["hourly"] == {"nmbe": 10.0, "cv_rmse": 30.0}


def test_nmae_ranks_never_flip_on_same_series():
    for s in range(30):
        y = cases.base_load(seed=s)
        a, b = cases.case_unbiased_noisy(y, seed=s), cases.case_biased_steady(y, seed=s)
        picks = {k: m.nmae(y, a, m.norm_value(y, k)) < m.nmae(y, b, m.norm_value(y, k)) for k in ("capacity", "mean", "range")}
        assert len(set(picks.values())) == 1
        assert picks["mean"] == (m.mae(y, a) < m.mae(y, b))


def test_nmae_scale_invariance_and_norm_values():
    y, a, _ = cases.first_screen()
    assert m.norm_value(y, "capacity") == 12.0 and m.norm_value(y, "mean") == 10.0 and m.norm_value(y, "range") == 4.0
    assert m.nmae(y, a, 10.0) == 20.0
    assert abs(m.nmae(1000 * y, 1000 * a, m.norm_value(1000 * y, "range")) - m.nmae(y, a, 4.0)) < 1e-9


def test_numpy_torch_agree():
    y = cases.base_load(seed=7)
    p = cases.case_biased_steady(y, seed=7)
    ty, tp = torch.tensor(y), torch.tensor(p)
    assert abs(m.nmbe(y, p) - tm.nmbe_ashrae(ty, tp)) < 1e-10
    assert abs(m.cv_rmse(y, p) - tm.cv_rmse(ty, tp)) < 1e-10
    assert abs(m.nmae(y, p, y.mean()) - tm.nmae(ty, tp, float(y.mean()))) < 1e-10
    d, dt = m.mse_decomposition(y, p), tm.mse_decomposition(ty, tp)
    assert abs(d["mse"] - dt["mse_builtin"]) < 1e-10 and abs(d["var"] - dt["var"]) < 1e-10


def test_exercises():
    # 연습 1: 실제 20, 20, 20, 20, 예측 18, 23, 17, 22 -> 오차 2, -3, 3, -2
    y = np.array([20.0, 20, 20, 20]); p = np.array([18.0, 23, 17, 22])
    assert m.nmbe(y, p, p=0) == 0.0 and abs(m.cv_rmse(y, p, p=0) - 100 * math.sqrt(6.5) / 20) < 1e-12
    # 연습 2: 월별 NMBE(n 분모) 4.6%면 p=1에서 4.6*12/11
    assert abs(4.6 * 12 / 11 - 5.018) < 1e-3
    # 연습 3: NMBE 6, CV 10 (분모 n) -> 흔들림 조각 8
    assert math.isclose(math.sqrt(10 ** 2 - 6 ** 2), 8.0)
