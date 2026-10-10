"""강화학습 7장 검증. `uv run pytest -q rl/ch07_nstep_td_lambda`"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch07_np as T  # noqa: E402
import rl_ch07_torch as TT  # noqa: E402
import rl_ch07_walk as W  # noqa: E402

torch.set_num_threads(2)
NS = W.N_WALK + 2


def _eps(seed=0, k=10):
    return W.walk_episodes(seed, k)


def test_walk_true_values_solve_bellman():
    v = W.walk_true_values()
    for i in range(1, W.N_WALK + 1):
        left = -1.0 if i == 1 else v[i - 1]
        right = 1.0 if i == W.N_WALK else v[i + 1]
        assert abs(v[i] - 0.5 * (left + right)) < 1e-12
    assert abs(W.rmse(np.zeros(NS)) - np.sqrt(2 * 91 / (13 * 49))) < 1e-12   # 처음 오차 0.5345


def test_hand_episode_first_screen():
    ep = W.hand_episode()
    z = np.zeros(7)
    V1 = T.n_step_td([ep], 7, 1, 0.5, v0=0.0)
    V2 = T.n_step_td([ep], 7, 2, 0.5, v0=0.0)
    Vinf = T.n_step_td([ep], 7, None, 0.5, v0=0.0)
    np.testing.assert_allclose(V1[1:6], [0, 0, 0, 0, 0.5])
    np.testing.assert_allclose(V2[1:6], [0, 0, 0, 0.5, 0.5])
    np.testing.assert_allclose(Vinf[1:6], [0, 0.5, 0.75, 0.5, 0.5])
    Va, _ = T.td_lambda([ep], 7, 0.5, 0.5, trace="accumulating")
    Vr, _ = T.td_lambda([ep], 7, 0.5, 0.5, trace="replacing")
    np.testing.assert_allclose(Va[1:6], [0, 0.0625, 0.15625, 0.25, 0.5])
    np.testing.assert_allclose(Vr[1:6], [0, 0.0625, 0.125, 0.25, 0.5])
    assert z.sum() == 0


def test_n1_and_lambda0_equal_td0():
    eps = _eps(1)
    ref = T.td0(eps, NS, 0.3)
    np.testing.assert_allclose(T.n_step_td(eps, NS, 1, 0.3), ref, atol=1e-14)
    for tr in ("accumulating", "replacing"):
        V, div = T.td_lambda(eps, NS, 0.0, 0.3, trace=tr)
        assert not div
        np.testing.assert_allclose(V, ref, atol=1e-14)


def test_n_infinity_equals_constant_alpha_mc():
    eps = _eps(2)
    ref = T.mc_constant_alpha_every_visit(eps, NS, 0.2)
    np.testing.assert_allclose(T.n_step_td(eps, NS, None, 0.2), ref, atol=1e-14)
    np.testing.assert_allclose(T.n_step_td(eps, NS, 10_000, 0.2), ref, atol=1e-14)


def test_lambda_return_recursion_matches_definition():
    rng = np.random.default_rng(0)
    V = rng.normal(size=NS)
    V[0] = V[-1] = 0.0
    ep = _eps(3, 1)[0]
    for lam in (0.0, 0.4, 0.9, 1.0):
        for gamma in (1.0, 0.95):
            np.testing.assert_allclose(T.lambda_returns(ep, V, lam, gamma),
                                       T.lambda_return_by_definition(ep, V, lam, gamma), atol=1e-12)


def test_forward_backward_offline_equivalence_accumulating():
    # 판 동안 V를 고정하는(오프라인) 표 기반에서만 정확히 같다
    eps = _eps(4)
    for lam in (0.0, 0.5, 0.9, 1.0):
        fwd = T.lambda_return_summed(eps, NS, lam, 0.1)
        bwd, div = T.td_lambda(eps, NS, lam, 0.1, trace="accumulating", online=False)
        assert not div
        np.testing.assert_allclose(fwd, bwd, atol=1e-12)


def test_online_and_replacing_are_not_exactly_forward_view():
    # 실패 재현: 같은 판에서 온라인 갱신이나 대체 흔적은 오프라인 λ-리턴과 어긋난다(다시 들른 칸이 있을 때)
    eps = _eps(4)
    fwd = T.lambda_return_summed(eps, NS, 0.9, 0.1)
    on, _ = T.td_lambda(eps, NS, 0.9, 0.1, trace="accumulating", online=True)
    rep, _ = T.td_lambda(eps, NS, 0.9, 0.1, trace="replacing", online=False)
    assert np.abs(on - fwd).max() > 1e-6
    assert np.abs(rep - fwd).max() > 1e-6
    seq = T.offline_lambda_return(eps, NS, 0.9, 0.1)       # 판 끝에 차례로 넣으면 합산과도 다르다
    assert np.abs(seq - fwd).max() > 1e-6


def test_offline_lambda_return_stays_between_targets():
    # 판 끝에 차례로 넣는 오프라인 λ-리턴은 alpha 1에서도 [-1, 1]을 벗어나지 않는다. 합산 판은 벗어난다
    eps = _eps(7, 10)
    seq = T.offline_lambda_return(eps, NS, 0.9, 1.0)
    summed = T.lambda_return_summed(eps, NS, 0.9, 1.0)
    assert np.abs(seq).max() <= 1.0 + 1e-12
    assert np.abs(summed).max() > 1.0


def test_lambda1_offline_accumulating_is_every_visit_replacing_is_first_visit():
    ep = W.hand_episode()                     # C를 두 번 지난다
    V0 = np.zeros(7)
    acc, _ = T.td_lambda([ep], 7, 1.0, 1.0, trace="accumulating", online=False)
    rep, _ = T.td_lambda([ep], 7, 1.0, 1.0, trace="replacing", online=False)
    # 보상은 마지막 +1뿐이라 모든 방문의 G = 1. 모든 방문 합: C는 2, 첫 방문만: C는 1
    np.testing.assert_allclose(acc[1:6], [0, 1, 2, 1, 1])
    np.testing.assert_allclose(rep[1:6], [0, 1, 1, 1, 1])
    assert V0.sum() == 0


def test_batch_matches_reference():
    eps_by_seed = [W.walk_episodes(s, 5) for s in range(4)]
    alphas = [0.1, 0.4, 0.9]
    for n in (1, 3, 8, None):
        hist = T.n_step_td_batch(eps_by_seed, NS, n, alphas)
        for b in range(4):
            for a, al in enumerate(alphas):
                rec = []
                T.n_step_td(eps_by_seed[b], NS, n, al, record=rec)
                np.testing.assert_allclose(hist[:, b * 3 + a], np.array(rec), atol=1e-12)
    for lam in (0.0, 0.6, 0.95):
        for tr in ("accumulating", "replacing"):
            for online in (True, False):
                hist, div, _ = T.td_lambda_batch(eps_by_seed, NS, lam, alphas, trace=tr, online=online)
                for b in range(4):
                    for a, al in enumerate(alphas):
                        rec = []
                        _, d = T.td_lambda(eps_by_seed[b], NS, lam, al, trace=tr, online=online, record=rec)
                        assert d == div[b * 3 + a]
                        if not d:
                            np.testing.assert_allclose(hist[:, b * 3 + a], np.array(rec), atol=1e-12)
        hist, div, _ = T.offline_lambda_return_batch(eps_by_seed, NS, lam, alphas)
        for b in range(4):
            for a, al in enumerate(alphas):
                rec = []
                T.offline_lambda_return(eps_by_seed[b], NS, lam, al, record=rec)
                np.testing.assert_allclose(hist[:, b * 3 + a], np.array(rec), atol=1e-12)


def test_torch_matches_numpy():
    rng = np.random.default_rng(1)
    V = rng.normal(size=NS)
    V[0] = V[-1] = 0.0
    for ep in _eps(5, 3):
        for lam in (0.0, 0.3, 0.8, 1.0):
            np.testing.assert_allclose(TT.lambda_returns_t(ep, V, lam, 1.0).numpy(),
                                       T.lambda_returns(ep, V, lam, 1.0), atol=1e-12)
        for n in (1, 2, 5, None):
            ref = np.array([T.n_step_return(ep, V, t, n, 0.9) for t in range(len(ep))])
            np.testing.assert_allclose(TT.n_step_returns_t(ep, V, n, 0.9).numpy(), ref, atol=1e-12)
    eps = _eps(6)
    np.testing.assert_allclose(TT.lambda_return_summed_t(eps, NS, 0.7, 0.2).numpy(),
                               T.lambda_return_summed(eps, NS, 0.7, 0.2), atol=1e-12)


def test_chain_n_step_needs_ceil_episodes():
    # 외길 10칸(결정 칸 9개), alpha 1, gamma 0.9: n스텝은 ceil(9/n)판, 끝까지(MC)는 1판
    ep = W.chain_episode(10)
    truth = W.chain_true_values(10, 0.9)
    for n, need in ((1, 9), (2, 5), (4, 3), (8, 2), (None, 1)):
        rec = []
        T.n_step_td([ep] * 12, 10, n, 1.0, gamma=0.9, record=rec)
        first = next(k + 1 for k, V in enumerate(rec) if np.allclose(V, truth, atol=1e-12))
        assert first == need
