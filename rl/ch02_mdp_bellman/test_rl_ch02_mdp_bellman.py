"""강화학습 2장 검증. `uv run pytest -q rl/ch02_mdp_bellman`"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch02_gridworld as gw  # noqa: E402
import rl_ch02_mdp_np as mdp  # noqa: E402
import rl_ch02_mdp_torch as mdpt  # noqa: E402

S0 = gw.to_index(gw.START, gw.W)
TERM = {gw.to_index(t, gw.W) for t in gw.TERMINALS}


def test_transition_rows_sum_to_one():
    for slip in (0.0, 0.1, 0.3):
        P, R = gw.chapter_grid(slip)
        np.testing.assert_allclose(P.sum(axis=2), 1.0, atol=1e-12)
        assert np.all(P >= 0)


def test_terminal_values_are_zero():
    P, R = gw.chapter_grid(0.1)
    V, _ = mdp.optimal_values(P, R, 0.9)
    for t in TERM:
        assert V[t] == 0.0


def test_hand_computed_optimal_values():
    # 손계산: +10 바로 옆 칸 10, 두 칸 9, ... 시작 칸(6걸음) 10 * 0.9**5
    P, R = gw.chapter_grid(0.0)
    V, _ = mdp.optimal_values(P, R, 0.9)
    path = [(2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    for k, cell in enumerate(path):
        assert abs(V[gw.to_index(cell, gw.W)] - 10 * 0.9**k) < 1e-12
    assert abs(V[S0] - max(1.0, 10 * 0.9**5)) < 1e-12


def test_gamma_half_prefers_small_reward_at_start():
    P, R = gw.chapter_grid(0.0)
    V, _ = mdp.optimal_values(P, R, 0.5)
    Q = mdp.q_from_v(V, P, R, 0.5)
    assert int(np.argmax(Q[S0])) == 1  # 오른쪽(+1)
    assert abs(Q[S0, 2] - 10 * 0.5**5) < 1e-12


def test_numpy_matches_torch():
    rng = np.random.default_rng(1)
    for slip in (0.0, 0.3):
        P, R = gw.chapter_grid(slip)
        V = rng.standard_normal(P.shape[0])
        pi = rng.random((P.shape[0], 4))
        pi /= pi.sum(1, keepdims=True)
        Pt, Rt, Vt, pit = (torch.tensor(x, dtype=torch.float64) for x in (P, R, V, pi))
        np.testing.assert_allclose(mdpt.bellman_expectation_backup(Vt, pit, Pt, Rt, 0.9).numpy(),
                                   mdp.bellman_expectation_backup(V, pi, P, R, 0.9), atol=1e-12)
        np.testing.assert_allclose(mdpt.bellman_optimality_backup(Vt, Pt, Rt, 0.9).numpy(),
                                   mdp.bellman_optimality_backup(V, P, R, 0.9), atol=1e-12)


def test_solution_is_fixed_point():
    P, R = gw.chapter_grid(0.3)
    V, _ = mdp.optimal_values(P, R, 0.9)
    np.testing.assert_allclose(mdp.bellman_optimality_backup(V, P, R, 0.9), V, atol=1e-12)


def test_myopic_policy_fails_reproducibly():
    """실패 재현: 근시 정책은 시작 칸에서 늘 +1로 가고 리턴 1.0, 최적 정책은 10*0.9**5."""
    P, R = gw.chapter_grid(0.0)
    my, opt = mdp.myopic_policy(P, R), mdp.optimal_policy(P, R, 0.9)
    rng = np.random.default_rng(0)
    g_my = [mdp.rollout(my, P, R, 0.9, S0, TERM, rng)[0] for _ in range(50)]
    g_opt = [mdp.rollout(opt, P, R, 0.9, S0, TERM, rng)[0] for _ in range(50)]
    assert np.allclose(g_my, 1.0)
    assert np.allclose(g_opt, 10 * 0.9**5)
    assert np.mean(g_opt) > 5 * np.mean(g_my)


def test_monte_carlo_matches_bellman_expectation():
    P, R = gw.chapter_grid(0.1)
    pi = mdp.optimal_policy(P, R, 0.9)
    Vpi, _ = mdp.evaluate(pi, P, R, 0.9)
    rng = np.random.default_rng(0)
    g = np.array([mdp.rollout(pi, P, R, 0.9, S0, TERM, rng)[0] for _ in range(3000)])
    assert abs(g.mean() - Vpi[S0]) < 4 * g.std() / np.sqrt(len(g))


def test_rollout_is_seed_reproducible():
    P, R = gw.chapter_grid(0.3)
    pi = mdp.optimal_policy(P, R, 0.9)
    a = mdp.rollout(pi, P, R, 0.9, S0, TERM, np.random.default_rng(7))
    b = mdp.rollout(pi, P, R, 0.9, S0, TERM, np.random.default_rng(7))
    assert a == b
