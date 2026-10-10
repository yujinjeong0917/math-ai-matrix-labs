"""강화학습 8장 검증. `uv run pytest -q rl/ch08_function_approx`"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch08_np as F  # noqa: E402
import rl_ch08_torch as FT  # noqa: E402

torch.set_num_threads(2)
PHI, NXT, R = F.baird_star()
P_T = F.transition_matrix(NXT)
P_B = F.behavior_chain()
D_U = np.full(6, 1 / 6)
W0 = np.array([1, 1, 1, 1, 1, 1, 10.0])


def test_star_features_match_baird_figure_1():
    w = np.arange(7, dtype=float) + 1          # w0..w6 = 1..7
    v = PHI @ w
    assert np.allclose(v[:5], w[0] + 2 * w[1:6])
    assert np.isclose(v[5], 2 * w[0] + w[6])
    assert np.linalg.matrix_rank(PHI) == 6     # 가중치 7개, 상태 6개: 열이 일차독립이 아니다


def test_hand_calculation_one_sweep():
    w1, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, W0, 0.01, 0.99, 1)
    assert np.isclose(w1[0] - 1, 0.01 * (5 * 8.88 * 1 + (-0.12) * 2))     # 0.4416
    assert np.isclose(PHI[5] @ w1, 12.882)                                # V(6)은 오른다
    v_tab, _, _ = F.semi_gradient_td_sweeps(np.eye(6), NXT, R, PHI @ W0, 0.01, 0.99, 1)
    assert np.isclose(v_tab[5], 11.9988)                                  # 표에서는 내려간다


def test_uniform_updates_diverge_at_gamma_099():
    w, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, W0, 0.01, 0.99, 3000, blowup=1e300)
    assert F.value_norm(w, PHI) > 1e6
    A = F.expected_update_matrix(PHI, P_T, D_U, 0.99)
    assert np.linalg.eigvals(A).real.min() < -0.07


def test_real_part_formula_and_threshold():
    for g in (0.3, 0.5, 0.9, 0.95, 0.99):
        ev = np.linalg.eigvals(F.expected_update_matrix(PHI, P_T, D_U, g))
        cpx = ev[np.abs(ev.imag) > 1e-9]
        assert np.allclose(cpx.real, 7 / 6 - 5 * g / 4)
        assert np.isclose(np.trace(F.expected_update_matrix(PHI, P_T, D_U, g)), 5 - 2.5 * g)
    assert np.linalg.eigvals(F.expected_update_matrix(PHI, P_T, D_U, 14 / 15 - 1e-6)).real.min() > -1e-12
    assert np.linalg.eigvals(F.expected_update_matrix(PHI, P_T, D_U, 14 / 15 + 1e-3)).real.min() < -1e-4


def test_same_updates_converge_in_values_at_gamma_09():
    w, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, W0, 0.01, 0.9, 10000)
    assert F.value_norm(w, PHI) < 1e-6
    assert np.linalg.norm(w) > 1                # 값은 0으로 가도 가중치는 영공간 방향에 남는다


def test_removing_shared_weight_converges():
    Phi6 = PHI[:, 1:]                            # w0 제거: 각 상태가 자기 가중치만 가진 (배율이 붙은) 표
    A = F.expected_update_matrix(Phi6, P_T, D_U, 0.99)
    assert np.linalg.eigvals(A).real.min() > 0
    w, _, _ = F.semi_gradient_td_sweeps(Phi6, NXT, R, W0[1:], 0.05, 0.99, 30000)
    assert F.value_norm(w, Phi6) < 1e-2


def test_residual_gradient_bellman_error_decreases_monotonically():
    w = W0.copy()
    prev = F.bellman_error(w, PHI, P_T, R, 0.99, D_U)
    for _ in range(2000):
        w, _, _ = F.residual_sweeps(PHI, NXT, R, w, 0.01, 0.99, 1, mix=1.0)
        be = F.bellman_error(w, PHI, P_T, R, 0.99, D_U)
        assert be <= prev + 1e-12
        prev = be


def test_residual_mix_zero_equals_direct():
    a, _, _ = F.residual_sweeps(PHI, NXT, R, W0, 0.01, 0.95, 50, mix=0.0)
    b, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, W0, 0.01, 0.95, 50)
    assert np.allclose(a, b)
    M0 = F.residual_update_matrix(PHI, P_T, D_U, 0.95, 0.0)
    assert np.allclose(M0, F.expected_update_matrix(PHI, P_T, D_U, 0.95))


def test_torch_detach_is_semi_gradient_and_no_detach_is_residual_gradient():
    for mix, det in ((0.0, True), (1.0, False)):
        w = W0.copy()
        ws = [w.copy()]
        for _ in range(200):
            w, _, _ = F.residual_sweeps(PHI, NXT, R, w, 0.01, 0.99, 1, mix=mix)
            ws.append(w.copy())
        wt = FT.run_sweeps(PHI, NXT, R, W0, 0.01, 0.99, 200, detach_target=det).numpy()
        assert np.allclose(np.array(ws), wt, rtol=1e-9, atol=1e-12)


def test_on_policy_matrix_is_positive_semidefinite():
    d = F.stationary_distribution(P_B)
    assert np.allclose(d, 1 / 6)
    for g in (0.5, 0.9, 0.99):
        A = F.expected_update_matrix(PHI, P_B, d, g)
        assert np.linalg.eigvalsh((A + A.T) / 2).min() > -1e-12


def test_trajectory_on_policy_converges_off_policy_diverges():
    w_on, dv_on, _ = F.td0_trajectory(PHI, P_B, P_T, R, W0, 0.01, 0.99, 60000, seed=1000, off_policy=False)
    assert dv_on is None and F.value_norm(w_on, PHI) < 1e-2
    w_off, dv_off, _ = F.td0_trajectory(PHI, P_B, P_T, R, W0, 0.01, 0.99, 60000, seed=1000, off_policy=True)
    assert dv_off is not None                   # 노름이 1e8을 넘어 멈춘다


def test_importance_ratio_values():
    log = []
    F.td0_trajectory(PHI, P_B, P_T, R, W0, 0.01, 0.99, 300, seed=0, off_policy=True, log=log, log_steps=300)
    rhos = {row["info"]["rho"] for row in log}
    assert rhos <= {0.0, 6.0}
    for row in log:
        assert (row["info"]["rho"] == 6.0) == (row["s_next"] == 6)
