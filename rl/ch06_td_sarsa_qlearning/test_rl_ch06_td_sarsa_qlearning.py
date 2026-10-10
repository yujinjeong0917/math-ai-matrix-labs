"""강화학습 6장 검증. `uv run pytest -q rl/ch06_td_sarsa_qlearning`"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch06_envs as E  # noqa: E402
import rl_ch06_td_np as T  # noqa: E402
import rl_ch06_td_torch as TT  # noqa: E402

torch.set_num_threads(2)


def test_hand_corridor_first_four():
    # 첫 화면 손계산: 5장의 네 판에 TD(0), alpha 0.5, gamma 0.9
    V = [0.0, 0.0, 0.0]
    after = []
    for path in E.CH05_FIRST_FOUR:
        for s, r, s2, d in E.corridor_transitions(path):
            T.td0_update(V, s, r, s2, d, 0.5, 0.9)
        after.append((V[0], V[1]))
    np.testing.assert_allclose(after, [(0.0, 0.5), (0.225, 0.75), (0.45, 0.875), (0.61875, 0.9375)], atol=1e-15)


def test_td0_update_sets_exact_target_when_alpha_one():
    V = [0.0, 0.3, 0.8]
    T.td0_update(V, 1, -1.0, 2, False, 1.0, 0.9)
    assert abs(V[1] - (-1.0 + 0.9 * 0.8)) < 1e-15  # V + 1 x (목표 - V)는 부동소수 한 칸 안에서 목표
    T.td0_update(V, 0, 2.0, 1, True, 1.0, 0.9)  # 종료 전이: 다음 추정값은 쓰지 않는다
    assert V[0] == 2.0


def test_chain_td_needs_n_episodes_mc_needs_one():
    # 앞에서부터 고치면 TD(0)는 한 판에 한 칸씩만 정답이 퍼진다(alpha 1, 결정적 사슬)
    n, gamma = 10, 0.9
    tr = [(s, *E.chain_step(s, 1, n)) for s in range(n - 1)]
    tr = [(s, r, s2, d) for s, s2, r, d in tr]
    truth = np.array([gamma ** (n - 2 - s) for s in range(n - 1)] + [0.0])
    for k in range(1, n - 1):
        V = T.td0([tr] * k, n, 1.0, gamma)
        assert np.sum(np.abs(V - truth) < 1e-12) == k + 1  # 맞은 칸 k개 + 종료 칸
    np.testing.assert_allclose(T.td0([tr] * (n - 1), n, 1.0, gamma), truth, atol=1e-15)
    np.testing.assert_allclose(T.mc_constant_alpha([tr], n, 1.0, gamma), truth, atol=1e-15)


def test_q_learning_fixed_point_is_q_star():
    Q = T.q_learning_chain(6, 0.9, 0.5, 2000, 0.3, 0)
    np.testing.assert_allclose(Q[:5], E.chain_q_star(6, 0.9)[:5], atol=1e-9)


def test_walk_true_values_and_td_close_after_many_episodes():
    truth = E.walk_true_values(5)
    np.testing.assert_allclose(truth[1:6], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
    uni = E.Uniforms(np.random.default_rng(0))
    eps = [E.walk_episode(uni) for _ in range(20_000)]
    V = T.td0(eps, 7, 0.01, 1.0, v0=0.5)
    assert np.max(np.abs(V[1:6] - truth[1:6])) < 0.05
    V = T.mc_sample_average(eps, 7, 1.0)
    assert np.max(np.abs(V[1:6] - truth[1:6])) < 0.01


def test_td_changes_one_state_after_first_walk_mc_changes_all_visited():
    uni = E.Uniforms(np.random.default_rng(5))
    ep = E.walk_episode(uni)
    Vt = T.td0([ep], 7, 0.1, 1.0, v0=0.5)
    Vm = T.mc_constant_alpha([ep], 7, 0.1, 1.0, v0=0.5)
    visited = {s for s, _, _, _ in ep}
    assert np.sum(Vt[1:6] != 0.5) == 1           # 종료 칸 바로 옆 한 칸만 (나머지는 목표 = 0.5 그대로)
    assert np.sum(Vm[1:6] != 0.5) == len(visited)  # MC는 지나간 칸 모두


def test_cliff_env():
    env = E.Cliff()
    assert (env.S, env.start, env.goal) == (40, 30, 39)
    assert env.step(30, 1) == (30, -50.0, False, True)  # 출발 칸 오른쪽은 절벽
    assert env.step(29, 2) == (39, -1.0, True, False)   # 도착 칸 바로 위에서 아래로
    assert env.step(0, 0) == (0, -1.0, False, False)    # 벽


def test_stuck_greedy_mc_learns_nothing_td_escapes_only_with_step_cost():
    env = E.Cliff()
    mc = T.control(env, "mc", 1, 0.5, 1.0, 0.0, 0, max_steps=2000, tie="first")
    assert mc["truncated"] == 1 and mc["updates"] == 0 and np.all(mc["Q"] == 0)
    q = T.control(env, "q", 1, 0.5, 1.0, 0.0, 0, max_steps=2000, tie="first")
    assert q["truncated"] == 0
    env0 = E.Cliff(step=0.0)
    q0 = T.control(env0, "q", 1, 0.5, 1.0, 0.0, 0, max_steps=2000, tie="first")
    assert q0["truncated"] == 1 and q0["updates"] == 2000 and np.all(q0["Q"] == 0)  # 고쳐도 0 -> 0


def test_replay_reproduces_run_and_torch_matches():
    env = E.Cliff()
    for m in ("sarsa", "q", "expected"):
        log = []
        o = T.control(env, m, 30, 0.5, 1.0, 0.1, 3, log=log)
        Qn = T.replay(log, env.S, env.A, m, 0.5, 1.0, 0.1)
        np.testing.assert_array_equal(Qn, o["Q"])
        Qt = TT.replay_t(log, env.S, env.A, m, 0.5, 1.0, 0.1).numpy()
        np.testing.assert_allclose(Qt, Qn, atol=1e-12, rtol=0)


def test_batched_walk_matches_numpy():
    runs = []
    for sd in range(8):
        uni = E.Uniforms(np.random.default_rng(sd))
        runs.append([E.walk_episode(uni) for _ in range(20)])
    for fn, fb in ((T.td0, TT.td0_batch), (T.mc_constant_alpha, TT.mc_batch)):
        Ht = fb(runs, 7, 0.1, 1.0, v0=0.5).numpy()
        for b in range(8):
            rec = []
            fn(runs[b], 7, 0.1, 1.0, v0=0.5, record=rec)
            np.testing.assert_allclose(Ht[:, b, :], np.array(rec), atol=1e-12, rtol=0)


def test_eps_greedy_probs_sum_and_ties():
    p = T.eps_greedy_probs([1.0, 3.0, 3.0, 0.0], 0.1)
    np.testing.assert_allclose(p, [0.025, 0.475, 0.475, 0.025])
    pt = TT.eps_greedy_probs_t(torch.tensor([1.0, 3.0, 3.0, 0.0], dtype=torch.float64), 0.1).numpy()
    np.testing.assert_allclose(pt, p, atol=1e-15)


def test_q_learning_learns_edge_path_sarsa_does_not():
    env = E.Cliff()
    q = T.control(env, "q", 500, 0.5, 1.0, 0.1, 1)
    s = T.control(env, "sarsa", 500, 0.5, 1.0, 0.1, 1)
    gq = T.greedy_rollout(env, q["Q"])
    assert gq[2] and gq[1] == env.w + 1  # 절벽 바로 위 줄: 위 1, 오른쪽 w-1, 아래 1
    assert sum(q["falls"]) > sum(s["falls"])
