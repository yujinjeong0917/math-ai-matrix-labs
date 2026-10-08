"""강화학습 3장 검증. `uv run pytest -q rl/ch03_policy_evaluation`"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch03_eval_np as ev  # noqa: E402
import rl_ch03_eval_torch as evt  # noqa: E402
import rl_ch03_grid as g  # noqa: E402


def test_policy_rows_sum_to_one_and_match_full_build():
    m = g.chapter_grid(6)
    np.testing.assert_allclose(m["P_pi"].sum(axis=1), 1.0, atol=1e-12)
    P, R = g.make_grid_full(6, 6, {(5, 5): 1.0}, ((5, 5),))
    pi = np.full((36, 4), 0.25)
    np.testing.assert_allclose(g.policy_matrix(P, pi), m["P_pi"], atol=1e-15)
    np.testing.assert_allclose(g.policy_reward(P, R, pi), m["r_pi"], atol=1e-15)


def test_corridor_hand_solution():
    # V(A) = 0.45 V(A) + 0.45 V(B), V(B) = 0.45 V(A) + 0.5  ->  V(B) = 0.5 / (1 - 0.45 * 9/11)
    c = g.corridor()
    V = ev.evaluate_direct(c["P_pi"], c["r_pi"], 0.9)
    VB = 0.5 / (1 - 0.45 * 0.45 / 0.55)
    assert abs(V[1] - VB) < 1e-12
    assert abs(V[0] - 0.45 / 0.55 * VB) < 1e-12
    assert V[2] == 0.0
    # 손으로 적은 처음 세 스윕
    P, r, Vk = c["P_pi"], c["r_pi"], np.zeros(3)
    expect = [(0.0, 0.5), (0.225, 0.5), (0.32625, 0.60125)]
    for a, b in expect:
        Vk = r + 0.9 * P @ Vk
        assert abs(Vk[0] - a) < 1e-12 and abs(Vk[1] - b) < 1e-12


def test_direct_and_iterative_agree():
    m = g.chapter_grid(12)
    V = ev.evaluate_direct(m["P_pi"], m["r_pi"], 0.95)
    Vi, _, _ = ev.evaluate_iterative(m["P_pi"], m["r_pi"], 0.95, tol=1e-11, V_true=V)
    Vs, _, _ = ev.evaluate_iterative_sparse(m["nbr"], m["prob"], m["r_pi"], 0.95, tol=1e-11, V_true=V)
    Vg, _, _ = ev.evaluate_inplace(m["nbr"], m["prob"], m["r_pi"], 0.95, tol=1e-11, V_true=V)
    for X in (Vi, Vs, Vg):
        np.testing.assert_allclose(X, V, atol=1e-10)
    # 해는 백업의 고정점이다
    np.testing.assert_allclose(m["r_pi"] + 0.95 * m["P_pi"] @ V, V, atol=1e-12)


def test_error_shrinks_by_at_least_gamma_each_sweep():
    m = g.chapter_grid(10)
    gm = 0.9
    V = ev.evaluate_direct(m["P_pi"], m["r_pi"], gm)
    _, _, errs = ev.evaluate_iterative(m["P_pi"], m["r_pi"], gm, tol=1e-9, V_true=V)
    prev = np.max(np.abs(V))  # V_0 = 0
    for e in errs:
        assert e <= gm * prev + 1e-15
        prev = e


def test_numpy_torch_match():
    m = g.chapter_grid(15)
    P, r = m["P_pi"], m["r_pi"]
    Vn = ev.evaluate_direct(P, r, 0.99)
    Vt = evt.evaluate_direct(torch.tensor(P), torch.tensor(r), 0.99).numpy()
    np.testing.assert_allclose(Vt, Vn, atol=1e-10)
    Vni, kn, _ = ev.evaluate_iterative(P, r, 0.99, 1e-8, V_true=Vn)
    Vti, kt = evt.evaluate_iterative(torch.tensor(P), torch.tensor(r), 0.99, 1e-8, V_true=torch.tensor(Vn))
    assert kn == kt
    np.testing.assert_allclose(Vti.numpy(), Vni, atol=1e-10)


def test_gamma_one_with_terminal_is_singular():
    # 종료 칸은 자기 자신으로만 가니 I - P 의 그 행이 정확히 0이다
    m = g.chapter_grid(5)
    with pytest.raises(np.linalg.LinAlgError):
        ev.evaluate_direct(m["P_pi"], m["r_pi"], 1.0)


def test_gamma_one_without_terminal_returns_garbage_silently():
    # 종료 칸이 없는 확률 표에서는 I - P 가 특이행렬인데도 solve가 오류 없이 엉뚱한 값을 낼 수 있다(설계도 예상과 다름)
    nbr, _ = g.neighbors(5, 5, ())
    S = 25
    P = np.zeros((S, S))
    np.add.at(P, (np.repeat(np.arange(S), 4), nbr.ravel()), 0.25)
    A = np.eye(S) - P
    assert np.linalg.matrix_rank(A) == S - 1
    r = np.zeros(S)
    r[23] = 0.25
    try:
        V = np.linalg.solve(A, r)
    except np.linalg.LinAlgError:
        return  # 오류가 나면 그것도 괜찮다. 중요한 건 아래처럼 믿을 수 없는 값을 걸러 내는 것
    assert np.max(np.abs(A @ V - r)) > 1e-6 or np.max(np.abs(V)) > 1e6


def test_inplace_needs_fewer_sweeps():
    m = g.chapter_grid(10)
    V = ev.evaluate_direct(m["P_pi"], m["r_pi"], 0.99)
    _, kj, _ = ev.evaluate_iterative_sparse(m["nbr"], m["prob"], m["r_pi"], 0.99, 1e-6, V_true=V)
    _, kg, _ = ev.evaluate_inplace(m["nbr"], m["prob"], m["r_pi"], 0.99, 1e-6, V_true=V)
    assert kg < kj


def test_diff_stop_error_within_gamma_over_one_minus_gamma():
    m = g.chapter_grid(10)
    gm, tol = 0.99, 1e-6
    V = ev.evaluate_direct(m["P_pi"], m["r_pi"], gm)
    Vd, _, _ = ev.evaluate_iterative_sparse(m["nbr"], m["prob"], m["r_pi"], gm, tol, V_true=None)
    err = np.max(np.abs(Vd - V))
    assert err > tol  # 앞뒤 차이가 tol보다 작아도 정답과의 오차는 tol보다 크다
    assert err <= gm / (1 - gm) * tol + 1e-15


def test_nonterminal_block_is_symmetric():
    # 벽에 부딪히면 제자리인 균등 무작위 걷기: 이웃 사이 확률이 양방향 모두 0.25
    m = g.chapter_grid(8)
    keep = ~m["term"]
    Q = m["P_pi"][np.ix_(keep, keep)]
    np.testing.assert_array_equal(Q, Q.T)
