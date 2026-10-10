"""강화학습 5장 검증. `uv run pytest -q rl/ch05_monte_carlo`"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch05_grid as g  # noqa: E402
import rl_ch05_mc_np as m  # noqa: E402
import rl_ch05_mc_torch as mt  # noqa: E402

torch.set_num_threads(2)


def corridor_env():
    P, R, term, pi = g.corridor()
    return P, R, term, pi, m.make_env(P, R, term)


def test_corridor_hand_episodes():
    # 첫 화면 손계산: 시드 0에서 A로 출발한 첫 네 판과 그 리턴 0.9^(걸음 수 - 1)
    P, R, term, pi, env = corridor_env()
    rng = np.random.default_rng(0)
    eps = [m.generate_episode(env, pi, 0, rng) for _ in range(4)]
    assert [len(e.states) for e in eps] == [11, 2, 2, 2]
    G0 = [m.returns(e.rewards, 0.9)[0] for e in eps]
    np.testing.assert_allclose(G0, [0.9**10, 0.9, 0.9, 0.9], atol=1e-15)
    V = g.exact_v(P, R, pi, 0.9, term)
    np.testing.assert_allclose(V[:2], [0.647482, 0.791367], atol=1e-6)  # 3장의 값과 같다


def test_returns_backward():
    np.testing.assert_allclose(m.returns([0.0, 0.0, 1.0], 0.9), [0.81, 0.9, 1.0], atol=1e-15)
    np.testing.assert_allclose(m.returns([1.0, 1.0, 0.0], 1.0), [2.0, 1.0, 0.0], atol=1e-15)


def test_first_visit_approaches_exact_with_more_episodes():
    P, R, term = g.corner_grid(4)
    env = m.make_env(P, R, term)
    pi = g.uniform_policy(16)
    V = g.exact_v(P, R, pi, 0.9, term)
    rng = np.random.default_rng(3)
    eps = [m.generate_episode(env, pi, 0, rng) for _ in range(5000)]
    keep = ~term
    errs = []
    for N in (50, 500, 5000):
        Vf, Nf, skipped = m.mc_first_visit(eps[:N], 0.9, 16)
        assert skipped == 0
        errs.append(np.sqrt(np.mean((Vf[keep] - V[keep]) ** 2)))
    assert errs[0] > errs[1] > errs[2]
    # 5,000판이면 칸마다 수천 번 이상 방문. 리턴의 표준편차가 0.25 이하라 4 x 0.25 / sqrt(1000) = 0.032 안
    Vf, Nf, _ = m.mc_first_visit(eps, 0.9, 16)
    assert Nf[keep].min() > 1000
    assert np.max(np.abs(Vf[keep] - V[keep])) < 0.032


def test_every_visit_equals_first_visit_when_no_revisits():
    P, R, term, pi, env = corridor_env()
    eps = [m.Episode([1], [1], [1.0], True), m.Episode([0, 1], [1, 1], [0.0, 1.0], True)]
    Vf, Nf, _ = m.mc_first_visit(eps, 0.9, 3)
    Ve, Ne, _ = m.mc_every_visit(eps, 0.9, 3)
    np.testing.assert_allclose(Vf, Ve)
    np.testing.assert_array_equal(Nf, Ne)
    # 다시 방문하면 다르다: A, A, B, 도착
    ep = m.Episode([0, 0, 1], [3, 1, 1], [0.0, 0.0, 1.0], True)
    assert m.mc_first_visit([ep], 0.9, 3)[0][0] == pytest.approx(0.81)
    assert m.mc_every_visit([ep], 0.9, 3)[0][0] == pytest.approx((0.81 + 0.9) / 2)


def test_truncated_episode_updates_nothing():
    # 2장 격자에서 모든 칸이 "위": (0,0)에서 벽에 부딪혀 제자리. 끝나지 않으니 MC는 아무것도 못 고친다
    P, R, term = g.ch02_grid()
    env = m.make_env(P, R, term)
    pi = np.zeros((25, 4))
    pi[:, 0] = 1.0
    ep = m.generate_episode(env, pi, 0, np.random.default_rng(0), max_steps=500)
    assert not ep.done and len(ep.states) == 500 and set(ep.states) == {0}
    V, N, skipped = m.mc_first_visit([ep], 0.9, 25)
    assert skipped == 1 and N.sum() == 0 and np.all(V == 0)


def test_greedy_control_from_zero_q_gets_stuck():
    P, R, term = g.ch02_grid()
    env = m.make_env(P, R, term)
    Q, _, lens, trunc = m.mc_control_eps_soft(env, 0, 0.9, 0.0, 3, np.random.default_rng(0), max_steps=300)
    assert trunc == 3 and lens == [300, 300, 300] and np.count_nonzero(Q) == 0


def test_exploring_start_forces_first_action():
    P, R, term = g.ch02_grid()
    env = m.make_env(P, R, term)
    pi = np.zeros((25, 4))
    pi[:, 1] = 1.0
    uni = m._Uniforms(np.random.default_rng(0))
    ep = m._first_forced(env, pi, 10, 0, np.random.default_rng(0), 100, uni)
    assert ep.states[0] == 10 and ep.actions[0] == 0 and ep.states[1] == 5
    assert all(a == 1 for a in ep.actions[1:])


def test_ratio_is_one_when_target_equals_behavior():
    P, R, term, b, env = corridor_env()
    rng = np.random.default_rng(0)
    eps = [m.generate_episode(env, b, 0, rng) for _ in range(200)]
    assert all(m.importance_ratio(e, b, b) == 1.0 for e in eps)
    o = m.off_policy_is(eps, b, b, 0.9, False)
    w = m.off_policy_is(eps, b, b, 0.9, True)
    assert o == pytest.approx(w) and o == pytest.approx(np.mean([0.9 ** (len(e.states) - 1) for e in eps]))


def test_ratio_does_not_need_transition_probs():
    # 비율은 정책 확률만으로 계산된다: 미끄러짐을 바꿔도 같은 (s, a) 기록이면 비율이 같다
    ep = m.Episode([0, 0, 1], [3, 1, 1], [0.0, 0.0, 1.0], True)
    b = np.tile([0.0, 0.5, 0.0, 0.5], (3, 1))
    pi = np.tile([0.0, 0.8, 0.0, 0.2], (3, 1))
    assert m.importance_ratio(ep, pi, b) == pytest.approx((0.2 / 0.5) * (0.8 / 0.5) * (0.8 / 0.5))


def test_one_state_sampler_matches_generic_episodes():
    P, R, term = g.one_state()
    env = m.make_env(P, R, term)
    b = g.one_state_policy(0.5)
    rng = np.random.default_rng(0)
    eps = [m.generate_episode(env, b, 0, rng) for _ in range(20000)]
    k_gen = np.array([sum(e.rewards) for e in eps])
    k_vec = m.sample_more_counts(20000, np.random.default_rng(1))
    # 둘 다 평균 1, 표준편차 sqrt(2): 4 x sqrt(2/20000) = 0.04 안
    assert abs(k_gen.mean() - 1.0) < 0.04 and abs(k_vec.mean() - 1.0) < 0.04
    # 같은 판 묶음이면 일반 함수와 벡터 함수의 추정값이 같다
    pi = g.one_state_policy(0.9)
    eps_k = [m.Episode([0] * (k + 1), [0] * k + [1], [1.0] * k + [0.0], True) for k in k_vec[:1000].tolist()]
    o, w = m.is_estimates_prefix(k_vec, 0.9, [1000])
    assert m.off_policy_is(eps_k, pi, b, 1.0, False) == pytest.approx(o[0], rel=1e-12)
    assert m.off_policy_is(eps_k, pi, b, 1.0, True) == pytest.approx(w[0], rel=1e-12)


def test_second_moment_threshold():
    # x = p^2 / b < 1 이어야 유한. b = 0.5면 p < 1/sqrt(2) = 0.7071...
    assert np.isfinite(m.is_second_moment(0.7)) and np.isinf(m.is_second_moment(0.71))
    assert np.isinf(m.is_second_moment(0.9))
    # 닫힌 꼴과 급수 직접 합이 같다
    for p in (0.6, 0.7):
        k = np.arange(0, 6000).astype(float)
        # 확률 0.5^(k+1) x 비율^2 = 0.5 x (2(1-p))^2 x (2p^2)^k 로 묶어 더한다(넘침 방지)
        series = np.sum(0.5 * (2 * (1 - p)) ** 2 * (2 * p * p) ** k * k**2)
        assert m.is_second_moment(p) == pytest.approx(series, rel=1e-9)


def test_ordinary_is_unbiased_when_variance_finite():
    # p = 0.6: 참값 1.5, rho G의 분산 15.80. 200개 시드 x 1,000판 평균의 표준오차 sqrt(15.8 / 200000) = 0.0089
    ests = [m.is_estimates_prefix(m.sample_more_counts(1000, np.random.default_rng(s)), 0.6, [1000])[0][0]
            for s in range(200)]
    assert abs(np.mean(ests) - 1.5) < 4 * 0.0089


def test_vectorized_walk_matches_generic():
    P, R, term = g.corner_grid(5)
    env = m.make_env(P, R, term)
    pi = g.uniform_policy(25)
    exact = g.exact_v(P, np.where(P > 0, 1.0, 0.0), pi, 1.0, term)[0]
    assert exact == pytest.approx(106.818182, abs=1e-6)
    L, alive = m.random_walk_lengths(4000, np.random.default_rng(0), P.argmax(axis=2), term, 0, 100000)
    rng = np.random.default_rng(1)
    Lg = np.array([len(m.generate_episode(env, pi, 0, rng, 100000).states) for _ in range(4000)])
    assert not alive.any()
    for x in (L, Lg):
        assert abs(x.mean() - exact) < 4 * x.std() / np.sqrt(x.size)


def test_torch_matches_numpy():
    P, R, term = g.corner_grid(4)
    env = m.make_env(P, R, term)
    pi = g.uniform_policy(16)
    rng = np.random.default_rng(0)
    eps = [m.generate_episode(env, pi, 0, rng) for _ in range(300)]
    Vn, Nn, _ = m.mc_first_visit(eps, 0.9, 16)
    Vt, Nt = mt.mc_first_visit(eps, 0.9, 16)
    np.testing.assert_allclose(Vt.numpy(), Vn, atol=1e-12)
    np.testing.assert_array_equal(Nt.numpy(), Nn)
    S, A, Rr, M = mt.pad(eps)
    G = mt.returns_batch(Rr, 0.9)
    for i, e in enumerate(eps[:50]):
        np.testing.assert_allclose(G[i, : len(e.states)].numpy(), m.returns(e.rewards, 0.9), atol=1e-12)
    Pc, Rc, termc, b = g.corridor()
    envc = m.make_env(Pc, Rc, termc)
    pic = np.tile([0.0, 0.8, 0.0, 0.2], (3, 1))
    rng = np.random.default_rng(1)
    epc = [m.generate_episode(envc, b, 0, rng) for _ in range(500)]
    ot, wt = mt.off_policy_is(epc, torch.tensor(pic), torch.tensor(b), 0.9)
    assert ot.item() == pytest.approx(m.off_policy_is(epc, pic, b, 0.9, False), rel=1e-12)
    assert wt.item() == pytest.approx(m.off_policy_is(epc, pic, b, 0.9, True), rel=1e-12)
    k = m.sample_more_counts(10000, np.random.default_rng(2))
    on, wn = m.is_estimates_prefix(k, 0.9, [10, 100, 10000])
    ot, wt = mt.is_estimates_prefix(k, 0.9, [10, 100, 10000])
    np.testing.assert_allclose(ot.numpy(), on, rtol=1e-12)
    np.testing.assert_allclose(wt.numpy(), wn, rtol=1e-12)


def test_page_hand_numbers():
    # 웹 챕터 본문의 손계산 숫자(results/ch05.json에 없는 파생값)를 다시 계산한다
    G = [0.9**10, 0.9, 0.9, 0.9]
    V, means = 0.0, []
    for n, x in enumerate(G, 1):
        V += (x - V) / n
        means.append(round(V, 4))
    assert means == [0.3487, 0.6243, 0.7162, 0.7622]
    assert round(0.7535 - 0.5711, 4) == 0.1824 and round(0.6580 - 0.6467, 4) == 0.0113
    assert round(0.6895 - 0.6184, 4) == 0.0711 and round(0.6512 - 0.6462, 4) == 0.0050
    assert round(0.2**2 / 0.5 * 0.8**2 / 0.5, 4) == 0.1024
    assert round((0.65 / 0.35) ** 2, 2) == 3.45
    assert round(m.is_second_moment(0.65) - (0.65 / 0.35) ** 2, 2) == 99.12
