"""강화학습 1장 검증. `uv run pytest -q rl/ch01_bandit`"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch01_bandit_np as bd  # noqa: E402
import rl_ch01_bandit_torch as bdt  # noqa: E402

P10 = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
P2 = [0.45, 0.55]


def test_same_seed_same_actions():
    env = bd.BernoulliBandit(P10, range(5), 300)
    for make in (bd.greedy, lambda: bd.eps_greedy(0.1), bd.ucb1, bd.thompson_beta):
        a1, r1 = bd.run(make(), env, 300, np.random.default_rng(0))
        a2, r2 = bd.run(make(), env, 300, np.random.default_rng(0))
        assert np.array_equal(a1, a2) and np.array_equal(r1, r2)


def test_reward_table_is_policy_independent():
    # 손잡이 a의 k번째 결과는 규칙과 상관없이 같다
    env = bd.BernoulliBandit(P2, range(3), 50)
    a, r = bd.run(bd.greedy(), env, 50, np.random.default_rng(1))
    seen = {0: 0, 1: 0}
    for t in range(50):
        arm = int(a[0, t])
        assert r[0, t] == float(env.table[0, seen[arm], arm])
        seen[arm] += 1


def test_ucb1_bonus_handles_zero_pulls():
    N = np.array([[0.0, 1.0, 9.0]])
    X = np.array([[0.0, 0.0, 4.0]])
    idx = bd.ucb1_index(N, X, 10)
    assert np.isinf(idx[0, 0]) and idx[0, 0] > 0
    assert abs(idx[0, 1] - math.sqrt(2 * math.log(10))) < 1e-12
    assert abs(idx[0, 2] - (4 / 9 + math.sqrt(2 * math.log(10) / 9))) < 1e-12
    # 손계산: 버려진 손잡이 2.146 > 밀어 주던 손잡이 1.160
    assert round(idx[0, 1], 3) == 2.146 and round(idx[0, 2], 3) == 1.160
    # n_j = 0인 손잡이가 있으면 UCB1은 그 손잡이를 먼저 고른다
    assert int(bd.argmax_random_tie(idx, np.array([0.99]))[0]) == 0


def test_beta_posterior_matches_closed_form():
    # Beta(1,1)에서 성공 s, 실패 f를 보면 Beta(1+s, 1+f), 평균 (1+s)/(2+s+f)
    rng = np.random.default_rng(0)
    r = (rng.random(200) < 0.3).astype(float)
    a, b = 1.0, 1.0
    for x in r:  # 하나씩 갱신
        a, b = a + x, b + (1 - x)
    A, B = bd.beta_posterior(np.array([200.0]), np.array([r.sum()]))
    assert A[0] == a == 1 + r.sum() and B[0] == b == 1 + 200 - r.sum()
    draws = np.random.default_rng(1).beta(A[0], B[0], size=200_000)
    assert abs(draws.mean() - A[0] / (A[0] + B[0])) < 3e-3


def test_tie_break_is_uniform():
    v = np.tile(np.array([[1.0, 0.0, 1.0, 1.0]]), (30_000, 1))
    a = bd.argmax_random_tie(v, np.random.default_rng(0).random(30_000))
    freq = np.bincount(a, minlength=4) / 30_000
    assert freq[1] == 0 and np.all(np.abs(freq[[0, 2, 3]] - 1 / 3) < 0.015)


def test_hand_lock_in_case():
    # 나쁜 쪽(0.45) 첫 결과 1, 좋은 쪽(0.55) 첫 결과 0이면 욕심쟁이는 좋은 쪽으로 돌아오지 않는다
    assert abs((1 - 0.55) * 0.45 - 0.2025) < 1e-12
    env = bd.BernoulliBandit(P2, range(400), 1000)
    a, r = bd.run(bd.greedy(), env, 1000, np.random.default_rng(12345))
    m = (r[:, 0] == 1) & (r[:, 1] == 0)
    assert m.sum() > 0
    assert np.all((a[m, 2:] == 0))


def test_greedy_gets_stuck_and_ucb1_does_not():
    # 실패 재현: 욕심쟁이는 좋은 손잡이를 10% 미만으로 고르는 반복이 적지 않다
    env = bd.BernoulliBandit(P2, range(300), 1000)
    a_g, _ = bd.run(bd.greedy(), env, 1000, np.random.default_rng(0))
    a_u, _ = bd.run(bd.ucb1(), env, 1000, np.random.default_rng(0))
    stuck_g = ((a_g == 1).mean(axis=1) < 0.1).mean()
    stuck_u = ((a_u == 1).mean(axis=1) < 0.1).mean()
    assert stuck_g > 0.2 and stuck_u == 0.0


def test_regret_matches_gap_times_counts():
    env = bd.BernoulliBandit(P10, range(4), 200)
    a, _ = bd.run(bd.ucb1(), env, 200, np.random.default_rng(0))
    T = bd.counts_over_time(a, 10)[:, -1, :]
    np.testing.assert_allclose(bd.pseudo_regret(a, env.gaps)[:, -1], T @ env.gaps, atol=1e-9)


def test_numpy_matches_torch_selection():
    env = bd.BernoulliBandit(P10, range(6), 400)
    for name, pol in (("ucb1", bd.ucb1()), ("thompson", bd.thompson_beta())):
        a_np, _ = bd.run(pol, env, 400, np.random.default_rng(5))
        a_t = bdt.run(name, env, 400, np.random.default_rng(5)).numpy()
        assert np.array_equal(a_np, a_t), name
    N = torch.tensor([[0.0, 3.0, 7.0]], dtype=torch.float64)
    X = torch.tensor([[0.0, 1.0, 5.0]], dtype=torch.float64)
    i_t = bdt.ucb1_index(N, X, 10).numpy()
    i_np = bd.ucb1_index(N.numpy(), X.numpy(), 10)
    assert np.isinf(i_t[0, 0]) and np.allclose(i_t[0, 1:], i_np[0, 1:], atol=1e-12, rtol=0)


def test_kl_and_lai_robbins_constant():
    # D(p||q) >= 2 (p-q)^2 (Auer 등 2002, 2절의 부등식)
    p = np.array(P10[:-1])
    assert np.all(bd.bernoulli_kl(p, 0.55) >= 2 * (0.55 - p) ** 2)
    _, c = bd.lai_robbins_curve(P10, 1000)
    assert abs(c - 27.805143227899748) < 1e-9
