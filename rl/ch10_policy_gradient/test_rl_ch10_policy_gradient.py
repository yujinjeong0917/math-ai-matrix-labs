"""강화학습 10장 검증. `uv run pytest -q rl/ch10_policy_gradient`"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch10_cartpole as cp  # noqa: E402
import rl_ch10_corridor as cor  # noqa: E402
import rl_ch10_pg_np as m  # noqa: E402
import rl_ch10_pg_torch as mt  # noqa: E402

torch.set_num_threads(2)


def test_corridor_hand_values_at_half():
    # 첫 화면 손계산: p = 0.5에서 A, B, C의 가치는 -12, -10, -6
    np.testing.assert_allclose(cor.exact_v(0.5), [-12.0, -10.0, -6.0], atol=1e-12)
    assert cor.exact_grad(0.0) == pytest.approx(2.0, abs=1e-12)


def test_corridor_matches_textbook_printed_values():
    # Sutton-Barto 2판 예제 13.1 본문: 가장 좋은 확률 약 0.59, 가치 약 -11.6, eps=0.1이면 -44, -82보다 나쁘다
    p = cor.best_p()
    assert p == pytest.approx(2 - np.sqrt(2), abs=1e-9)
    assert round(p, 2) == 0.59
    assert cor.exact_v(p)[0] == pytest.approx(-11.6, abs=0.06)
    assert cor.exact_v(0.95)[0] < -44 and cor.exact_v(0.05)[0] < -82


def test_deterministic_policies_never_finish():
    # 실패 재현: 칸을 못 보는 결정론적 정책은 끝나지 않는다
    rng = np.random.default_rng(0)
    assert not m.corridor_episode(40.0, rng, cap=500).done
    assert not m.corridor_episode(-40.0, rng, cap=500).done
    assert np.isneginf(cor.exact_v(1.0)[0]) and np.isneginf(cor.exact_v(0.0)[0])


def test_aliased_sarsa_can_only_reach_eps_policies():
    Q, p_right, _ = m.corridor_sarsa_aliased(300, 0)
    assert p_right in (0.05, 0.95)
    assert cor.exact_v(p_right)[0] < cor.exact_v(cor.best_p())[0] - 30


def test_exact_gradient_matches_finite_difference():
    h = 1e-6
    for t in (-1.5, -0.3, 0.0, 0.8):
        fd = (cor.exact_J(t + h) - cor.exact_J(t - h)) / (2 * h)
        assert cor.exact_grad(t) == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_shortest_episode_estimates():
    ep = m.Episode(states=[0, 1, 2], actions=[1, 0, 1], rewards=[-1.0] * 3, done=True)
    assert m.corridor_grad_estimate(ep, 0.0, "none") == pytest.approx(-1.0)
    assert m.corridor_grad_estimate(ep, 0.0, "const") == pytest.approx(5.0)
    assert m.corridor_grad_estimate(ep, 0.0, "value") == pytest.approx(3.0)


@pytest.mark.parametrize("theta", [0.0, -1.0])
def test_reinforce_unbiased_for_every_baseline(theta):
    # 몬테카를로 평균이 정확한 기울기에서 표준오차 4배 안. 베이스라인은 평균을 바꾸지 않는다
    rng = np.random.default_rng(123)
    eps = [m.corridor_episode(theta, rng, cap=2000) for _ in range(4000)]
    assert all(e.done for e in eps)
    exact = cor.exact_grad(theta)
    sds = {}
    for b in ("none", "const", "value"):
        g = np.array([m.corridor_grad_estimate(e, theta, b) for e in eps])
        se = g.std(ddof=1) / np.sqrt(len(g))
        assert abs(g.mean() - exact) < 4 * se
        sds[b] = g.std(ddof=1)
    # 실패 재현: 베이스라인이 없으면 흔들림이 더 크다
    assert sds["none"] > 1.3 * sds["const"]


def test_baseline_term_has_zero_mean_exactly():
    # E[(a - p)] = p(1-p) - (1-p)p = 0: 행동에 기대지 않는 b를 곱해도 평균은 0
    for p in (0.1, 0.5, 0.8):
        assert p * (1 - p) + (1 - p) * (0 - p) == pytest.approx(0.0, abs=1e-15)


def test_grad_log_pi_matches_numeric():
    theta = np.array([0.2, -0.4, 1.1, 0.3, -0.1])
    phi = np.array([0.5, -0.2, 0.3, 0.1, 1.0])
    h = 1e-7
    for a in (0, 1):
        num = np.zeros(5)
        for i in range(5):
            e = np.zeros(5)
            e[i] = h
            num[i] = (np.log(m.softmax_policy(theta + e, phi)[a]) - np.log(m.softmax_policy(theta - e, phi)[a])) / (2 * h)
        np.testing.assert_allclose(m.grad_log_pi(theta, phi, a), num, rtol=1e-6, atol=1e-9)


def test_torch_autograd_matches_numpy_corridor():
    rng = np.random.default_rng(5)
    th = 0.4
    for b in ("none", "const", "value"):
        for _ in range(20):
            e = m.corridor_episode(th, rng)
            G = m.returns(e.rewards, 1.0)
            bb = {"none": np.zeros_like(G), "const": np.full_like(G, cor.exact_J(th)),
                  "value": cor.exact_v(cor.sigmoid(th))[np.asarray(e.states)]}[b]
            g_t = mt.pg_grad([th], np.ones((len(G), 1)), e.actions, G - bb)[0]
            np.testing.assert_allclose(g_t, m.corridor_grad_estimate(e, th, b), rtol=1e-9, atol=1e-12)


def test_torch_autograd_matches_numpy_cartpole():
    env = cp.CartPole(1, 500)
    theta = np.array([0.1, 0.2, 2.0, 0.5, 0.0])
    S, A, R, _ = m.run_cartpole_episode(env, theta, np.random.default_rng(1))
    G = m.returns(R, 0.99)
    np.testing.assert_allclose(mt.returns(R, 0.99).numpy(), G, rtol=1e-12)
    Phi = np.array([m.phi_policy(s) for s in S])
    g_np = (G * (A - m.sigmoid(Phi @ theta))) @ Phi
    np.testing.assert_allclose(mt.pg_grad(theta, Phi, A, G), g_np, rtol=1e-9)


def test_cartpole_physics_falls_without_control_and_is_seeded():
    env = cp.CartPole(0)
    s = env.reset()
    env.state = np.array([0.0, 0.0, 0.05, 0.0])  # 살짝 기운 막대
    for t in range(200):
        _, _, fell, _ = env.step(1 if t % 2 else 0)  # 번갈아 밀기: 평균 힘 0
        if fell:
            break
    assert fell and t < 100
    a, b = cp.CartPole(7).reset(), cp.CartPole(7).reset()
    np.testing.assert_array_equal(a, b)
    assert np.all(np.abs(s) <= 0.05)


def test_cartpole_episode_truncates_at_cap():
    env = cp.CartPole(0, cap=30)
    theta = np.array([0.0, 0.0, 3.0, 1.0, 0.0])  # 막대가 기운 쪽으로 미는 손 규칙
    S, A, R, fell = m.run_cartpole_episode(env, theta, np.random.default_rng(0))
    assert len(A) <= 30 and (fell or len(A) == 30)


def test_reinforce_learns_corridor_from_bad_start():
    _, J, _ = m.corridor_reinforce(-2.0, 2.0 ** -8, 300, 0)
    assert J[-1] > cor.exact_J(-2.0) + 5
