"""강화학습 4장 검증. `uv run pytest -q rl/ch04_policy_value_iteration`"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch04_grid as g  # noqa: E402
import rl_ch04_pivi_np as m  # noqa: E402
import rl_ch04_pivi_torch as mt  # noqa: E402

GAMMA = 0.99


def small():
    (P, R, term), _ = g.compare_grid(8)
    return P, R, term


def test_grid_rows_sum_to_one_and_terminals_absorb():
    P, R, term = small()
    np.testing.assert_allclose(P.sum(axis=2), 1.0, atol=1e-12)
    for s in np.flatnonzero(term):
        assert np.all(P[s, :, s] == 1.0) and np.all(R[s] == 0.0)


def test_corridor_hand_steps():
    # 첫 화면 손계산: 모두 왼쪽 -> B만 오른쪽 -> A도 오른쪽 -> 멈춤. V(A)=0.9, V(B)=1
    P, R, term = g.corridor()
    pi, V, hist, ok = m.policy_iteration(P, R, 0.9, pi0=np.full(3, 3), tie="keep")
    assert ok and [h["changed"] for h in hist] == [1, 1, 0]
    assert pi[0] == 1 and pi[1] == 1
    np.testing.assert_allclose(V[:2], [0.9, 1.0], atol=1e-12)
    # 가치 반복: (0,0) -> (0,1) -> (0.9,1) -> 그대로
    Vk = np.zeros(3)
    for expect in ([0.0, 1.0], [0.9, 1.0], [0.9, 1.0]):
        Vk = m.bellman_T(Vk, P, R, 0.9)
        np.testing.assert_allclose(Vk[:2], expect, atol=1e-12)


def test_pi_and_vi_agree():
    P, R, _ = small()
    pi, V, _, ok = m.policy_iteration(P, R, GAMMA)
    Vv, _ = m.value_iteration(P, R, GAMMA, theta=1e-12)
    assert ok
    np.testing.assert_allclose(V, Vv, atol=1e-9)
    # V*는 최적 방정식의 고정점
    np.testing.assert_allclose(m.bellman_T(V, P, R, GAMMA), V, atol=1e-12)
    for k in (0, 1, 5, 20):
        _, Vm, _, _ = m.modified_policy_iteration(P, R, GAMMA, k, theta=1e-12)
        np.testing.assert_allclose(Vm, V, atol=1e-9)


def test_mpi_k0_is_value_iteration():
    P, R, _ = small()
    _, Vm, it, b = m.modified_policy_iteration(P, R, GAMMA, 0, theta=1e-10)
    Vv, hist = m.value_iteration(P, R, GAMMA, theta=1e-10)
    assert it == len(hist) and b == it * P.shape[0] * P.shape[1]
    np.testing.assert_allclose(Vm, Vv, atol=1e-14)


def test_contraction_on_100_random_pairs():
    P, R, _ = small()
    rng = np.random.default_rng(1)
    for _ in range(100):
        V, U = rng.normal(0, 5, P.shape[0]), rng.normal(0, 5, P.shape[0])
        lhs = np.max(np.abs(m.bellman_T(V, P, R, GAMMA) - m.bellman_T(U, P, R, GAMMA)))
        assert lhs <= GAMMA * np.max(np.abs(V - U)) + 1e-12


def test_policy_improvement_never_hurts():
    P, R, _ = small()
    _, _, hist, _ = m.policy_iteration(P, R, GAMMA, tie="first")
    for a, b in zip(hist, hist[1:]):
        # 동률 허용폭 1e-9 때문에 생길 수 있는 감소는 1e-9/(1-gamma) 이하
        assert np.min(b["V"] - a["V"]) >= -1e-9 / (1 - GAMMA)


def test_tie_first_terminates_but_random_does_not():
    P, R, term = g.tie_grid(6)
    pi0 = np.zeros(36, dtype=np.int64)
    _, _, hist, ok = m.policy_iteration(P, R, 0.95, pi0=pi0, tie="first", max_iter=200)
    assert ok and len(hist) == 11
    _, V, hist, ok = m.policy_iteration(P, R, 0.95, pi0=pi0, tie="random", rng=np.random.default_rng(0), max_iter=200)
    assert not ok  # 실패 재현: 정책은 이미 최적인데 동률 칸이 계속 뒤집힌다
    V_star, _ = m.value_iteration(P, R, 0.95, theta=1e-13)
    assert np.max(np.abs(hist[-1]["V"] - V_star)) < 1e-9
    assert hist[-1]["changed"] > 0


def test_keep_rule_terminates_from_random_starts():
    P, R, _ = g.tie_grid(6)
    for sd in range(5):
        pi0 = np.random.default_rng(sd).integers(0, 4, 36)
        _, _, _, ok = m.policy_iteration(P, R, 0.95, pi0=pi0, tie="keep", max_iter=200)
        assert ok


def test_gamma_one_value_iteration_grows_linearly():
    P, R, _ = g.loop_grid()
    V = np.zeros(9)
    for k in range(1, 31):
        V = m.bellman_T(V, P, R, 1.0)
        assert abs(np.max(V) - k) < 1e-12  # 실패 2: ||V_k|| = k
    V = np.zeros(9)
    for _ in range(300):
        V = m.bellman_T(V, P, R, 0.9)
    assert np.max(V) <= 1 / (1 - 0.9) + 1e-9


def test_no_terminal_sweeps_match_theory():
    P, R, _ = g.loop_grid()
    gamma, eps = 0.9, 1e-6
    V_star, _ = m.value_iteration(P, R, gamma, theta=1e-14)
    _, hist = m.value_iteration(P, R, gamma, theta=1e-14, V_star=V_star)
    k = next(i + 1 for i, (_, e) in enumerate(hist) if e < eps)
    assert k == m.theory_vi_sweeps(eps, gamma) == 153


def test_torch_matches_numpy():
    P, R, _ = small()
    Pt, Rt = torch.tensor(P), torch.tensor(R)
    pi, V, hist, _ = m.policy_iteration(P, R, GAMMA)
    pit, Vt, it = mt.policy_iteration(Pt, Rt, GAMMA)
    assert np.array_equal(pi, pit.numpy()) and it == len(hist)
    np.testing.assert_allclose(Vt.numpy(), V, atol=1e-10)
    Vv, kv = m.value_iteration(P, R, GAMMA, theta=1e-10)
    Vvt, kt = mt.value_iteration(Pt, Rt, GAMMA, theta=1e-10)
    assert kt == len(kv)
    np.testing.assert_allclose(Vvt.numpy(), Vv, atol=1e-10)
    _, Vm, _, _ = m.modified_policy_iteration(P, R, GAMMA, 5, theta=1e-10)
    _, Vmt, _ = mt.modified_policy_iteration(Pt, Rt, GAMMA, 5, theta=1e-10)
    np.testing.assert_allclose(Vmt.numpy(), Vm, atol=1e-10)
