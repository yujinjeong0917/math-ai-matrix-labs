"""강화학습 9장 검증. `uv run pytest -q rl/ch09_dqn`"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch09_cartpole as cp  # noqa: E402
import rl_ch09_dqn_np as m  # noqa: E402
import rl_ch09_dqn_torch as mt  # noqa: E402

torch.set_num_threads(2)


def filled_buffer(n=300, seed=5):
    rng = np.random.default_rng(seed)
    buf = m.ReplayBuffer(1000, seed + 2)
    env = cp.CartPole(seed, 500)
    s = env.reset()
    for _ in range(n):
        a = int(rng.integers(2))
        s2, r, fell, trunc = env.step(a)
        buf.add(s, a, r, s2, fell)
        s = env.reset() if (fell or trunc) else s2
    return buf


# ---------------------------------------------------------------- 첫 화면 손계산

def test_coin_max_hand_values():
    # 동전 K개(+1/-1 반반)의 최댓값 평균 = 1 - 2 * (1/2)^K : 0.5, 0.75, 0.875
    for K, v in ((2, 0.5), (3, 0.75), (4, 0.875)):
        assert m.coin_max_exact(K) == pytest.approx(v, abs=1e-12)
        assert 1 - 2 * 0.5 ** K == pytest.approx(v, abs=1e-12)


def test_target_hand_example():
    # 보상 1, 사본의 다음 상태 점수 [3.0, 3.4], 지금 망 [5.1, 4.8]
    assert 1 + 0.99 * 3.4 == pytest.approx(4.366, abs=1e-12)
    assert 1 + 0.99 * 3.0 == pytest.approx(3.97, abs=1e-12)


# ---------------------------------------------------------------- 실패 1: 최댓값의 치우침

def test_expected_max_normal_integral_known_values():
    # K=2의 정확한 값은 1/sqrt(pi)
    assert m.expected_max_normal(2) == pytest.approx(1 / np.sqrt(np.pi), abs=1e-6)
    assert m.expected_max_normal(1) == pytest.approx(0.0, abs=1e-9)


def test_single_estimator_overestimates_and_grows_with_K():
    vals = []
    for K in (2, 4, 8, 16, 32):
        r = m.max_bias_experiment(K, 10_000, seed=K)
        # 몬테카를로 평균이 수치적분과 표준오차 4배 안
        assert abs(r["single"] - m.expected_max_normal(K)) < 4 * r["single_se"]
        # 두 벌로 나누면 참값 0에서 표준오차 4배 안
        assert abs(r["double"]) < 4 * r["double_se"]
        # 설계도 검사: Double 값이 단일 값 이하인 경우가 과반
        assert r["frac_double_le_single"] > 0.5
        vals.append(r["single"])
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_double_can_underestimate_when_true_values_differ():
    # van Hasselt(2010): double estimator는 낮춰 잡을 수 있다. 한 행동만 0.5, 나머지 0
    r = m.max_bias_experiment(8, 10_000, seed=1008, mu=[0.5] + [0.0] * 7)
    assert r["double"] < 0.5 - 4 * r["double_se"]
    assert r["single"] > 0.5 + 4 * r["single_se"]


# ---------------------------------------------------------------- 신경망과 구현 대조

def test_manual_backprop_matches_finite_difference():
    buf = filled_buffer()
    net = m.MLP([4, 16, 16, 2], 0)
    S, A, R, S2, F = buf.sample_uniform(32)
    y = R + 0.99 * (1 - F) * 3.0
    _, g, _ = net.grad(m.feat(S), A, y)
    h = 1e-6
    for li in range(3):
        for (i, j) in [(0, 0), (1, 1)]:
            W = net.W[li]
            old = W[i, j]
            W[i, j] = old + h
            lp = net.grad(m.feat(S), A, y)[0]
            W[i, j] = old - h
            lm = net.grad(m.feat(S), A, y)[0]
            W[i, j] = old
            assert (lp - lm) / (2 * h) == pytest.approx(g[li][i, j], rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("double", [False, True])
def test_torch_autograd_matches_numpy(double):
    buf = filled_buffer()
    net = m.MLP([4, 64, 64, 2], 5)
    tgt = m.MLP([4, 64, 64, 2], 6)
    batch = buf.sample_uniform(32)
    S, A, R, S2, F = batch
    y = m.td_target(net, tgt, R, S2, F, 0.99, double)
    loss_np, g_np, _ = net.grad(m.feat(S), A, y)
    loss_t, g_t, y_t = mt.loss_and_grads(net, tgt, batch, 0.99, double)
    np.testing.assert_allclose(y, y_t, rtol=1e-6)
    assert loss_np == pytest.approx(loss_t, rel=1e-6)
    for a, b in zip(g_np, g_t):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)


def test_double_target_never_above_single_target():
    # 같은 타깃망이 매기면 Double 목표(지금 망이 고른 행동)는 단일 목표(최댓값) 이하
    buf = filled_buffer()
    net, tgt = m.MLP([4, 32, 32, 2], 1), m.MLP([4, 32, 32, 2], 2)
    S, A, R, S2, F = buf.sample_uniform(200)
    ys = m.td_target(net, tgt, R, S2, F, 0.99, False)
    yd = m.td_target(net, tgt, R, S2, F, 0.99, True)
    assert np.all(yd <= ys + 1e-12)


def test_fell_transition_has_no_bootstrap_but_truncated_does():
    net = m.MLP([4, 8, 8, 2], 0)
    S2 = np.zeros((2, 4))
    y = m.td_target(net, net, np.array([1.0, 1.0]), S2, np.array([1.0, 0.0]), 0.99, False)
    assert y[0] == 1.0
    assert y[1] == pytest.approx(1.0 + 0.99 * net.forward(S2[1:2]).max())


# ---------------------------------------------------------------- 재생 버퍼

def test_buffer_overwrites_oldest_first():
    buf = m.ReplayBuffer(5, 0)
    for k in range(8):
        buf.add(np.full(4, k), 0, float(k), np.full(4, k), False)
    assert buf.n == 5
    assert sorted(buf.R.tolist()) == [3.0, 4.0, 5.0, 6.0, 7.0]   # 0, 1, 2가 먼저 지워짐
    S, A, R, S2, F = buf.latest(3)
    assert R.tolist() == [7.0, 6.0, 5.0]


def test_uniform_sample_covers_buffer():
    buf = m.ReplayBuffer(100, 0)
    for k in range(100):
        buf.add(np.zeros(4), 0, float(k), np.zeros(4), False)
    seen = set()
    for _ in range(50):
        seen.update(buf.sample_uniform(32)[2].tolist())
    assert len(seen) > 95


# ---------------------------------------------------------------- 타깃망

def test_target_network_constant_between_copies():
    cfg = m.Config(steps=1_400, learn_start=1_000, C=100, eval_every=10 ** 9)
    # 고친 횟수 400번이면 복사는 4번
    run = m.dqn(3, use_replay=True, use_target=True, cfg=cfg)
    assert run.updates == 400 and run.target_copies == 4


def test_target_frozen_during_C_updates():
    buf = filled_buffer()
    net = m.MLP([4, 16, 16, 2], 0)
    tgt = net.clone()
    opt = m.Adam(net.params(), 1e-3)
    W0 = [w.copy() for w in tgt.W]
    for _ in range(20):
        S, A, R, S2, F = buf.sample_uniform(32)
        y = m.td_target(net, tgt, R, S2, F, 0.99, False)
        _, g, _ = net.grad(m.feat(S), A, y)
        opt.step(net.params(), g)
    assert all(np.array_equal(a, b) for a, b in zip(W0, tgt.W))      # 사본은 그대로
    assert not all(np.array_equal(a, b) for a, b in zip(W0, net.W))  # 지금 망은 바뀜


def test_dqn_is_deterministic_given_seed():
    cfg = m.Config(steps=1_500, eval_every=500)
    a = m.dqn(7, cfg=cfg)
    b = m.dqn(7, cfg=cfg)
    assert a.ep_len == b.ep_len and a.eval_q0 == b.eval_q0


# ---------------------------------------------------------------- 웹 챕터에 쓴 파생 숫자

def test_numbers_derived_on_page():
    g = 0.99
    assert 1 / (1 - g) == pytest.approx(100.0)                                   # 점수의 상한
    assert round((1 - g ** 500) / (1 - g), 2) == 99.34                            # 500걸음 버틴 판의 할인 리턴
    assert (7 * 1 + 1 * (-1)) / 8 == 0.75                                         # 동전 3개 손계산
    assert m.coin_max_exact(5) == pytest.approx(0.9375)                          # 연습 1
    assert round(2.095571 - 0.5, 3) == 1.596 and round(0.5 - 0.047943, 3) == 0.452  # K=32, 참값 0.5
    assert 4 * 64 + 64 * 64 + 64 * 2 == 4480                                     # 상태 하나 순전파 곱셈-덧셈
    assert sum(a * b + b for a, b in ((4, 64), (64, 64), (64, 2))) == 4610       # 매개변수 수
    assert 2 * 4480 * (1 + 3 * 32 + 32) == 1_155_840                             # 걸음당 FLOP(단일)
    assert 2 * 4480 * (1 + 4 * 32 + 32) == 1_442_560                             # 걸음당 FLOP(Double)
    assert m.ReplayBuffer(50_000, 0).nbytes() == 4_400_000                       # 버퍼 메모리
