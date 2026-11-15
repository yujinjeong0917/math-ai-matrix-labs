"""강화학습 11장 검증. `uv run pytest -q rl/ch11_trpo_ppo`"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch11_cartpole as cp  # noqa: E402
import rl_ch11_ppo_np as m  # noqa: E402
import rl_ch11_ppo_torch as mt  # noqa: E402

torch.set_num_threads(2)


# 무작위 정책 w(정규분포 sd 0.5)로 16판을 모아 이득까지 만든다. 모든 테스트가 같은 데이터를 쓰도록 시드 고정
def sample_batch(seed=4, w_seed=3):
    w = np.random.default_rng(w_seed).normal(0, 0.5, 5)
    batch = m.collect(w, 16, np.random.default_rng(seed))
    return w, batch, m.advantages(batch)


# ---------- 첫 화면 손계산 ----------
def test_hand_example_numbers():
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    # 손계산 표: 손잡이 z가 1, 4, 5일 때의 오른쪽 확률
    assert round(sig(1), 4) == 0.7311 and round(sig(4), 4) == 0.982 and round(sig(5), 4) == 0.9933
    kl = lambda p, q: p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))  # noqa: E731
    # 같은 손잡이 이동(z+1)이라도 p=0.5에서는 KL 0.12, p가 이미 0.98인 곳(z=4)에서는 0.0066: 매개변수 거리와 정책 거리가 다르다는 TRPO의 동기
    assert round(kl(0.5, sig(1)), 4) == 0.1201
    assert round(kl(sig(4), sig(5)), 4) == 0.0066
    # 이득 +1, 오른쪽 확률 0.5 -> 0.7311: 비율 1.4621, 잘라 낸 목적은 1.2에서 멈춘다
    lo = np.log(np.array([0.5]))
    ln = np.log(np.array([sig(1)]))
    assert round(float(np.exp(ln - lo)[0]), 4) == 1.4621
    # A=+1, 비율 1.4621 > 1+eps이므로 min(1.4621, 1.2) = 1.2에서 멈춘다(더 올려도 이득 없음)
    assert m.ppo_clip_objective(ln, lo, np.array([1.0]), 0.2) == pytest.approx(1.2)
    # 이득 -1이면 비율을 0.6까지 내려도 -0.8에서 멈춘다
    # A=-1, 비율 0.6 < 1-eps: min(-0.6, -0.8) = -0.8. 나쁜 행동 확률을 더 내려도 이득이 늘지 않는 쪽으로 막힌다
    assert m.ppo_clip_objective(np.log([0.3]), np.log([0.5]), np.array([-1.0]), 0.2) == pytest.approx(-0.8)


# ---------- 환경 ----------
def test_cartpole_one_step_matches_hand_formula():
    s = np.zeros((1, 4))
    nxt, done = cp.step(s, np.array([1]))
    # 정지 상태(0,0,0,0)에서 오른쪽 힘 10N: temp = F / 전체질량 1.1. 이 아래는 운동 방정식을 직접 손으로 푼 값
    temp = 10.0 / 1.1
    th_acc = (0.0 - temp) / (0.5 * (4 / 3 - 0.1 / 1.1))
    x_acc = temp - 0.05 * th_acc / 1.1
    # 오일러 한 걸음: 위치와 각도는 옛 속도(0)로 변하지 않고 속도만 TAU * 가속도
    np.testing.assert_allclose(nxt[0], [0.0, 0.02 * x_acc, 0.0, 0.02 * th_acc], atol=1e-15)
    assert not done[0]


def test_episode_lengths_and_returns():
    w, batch, _ = sample_batch()
    assert batch["lengths"].sum() == batch["a"].size
    assert batch["lengths"].max() <= cp.MAX_STEPS
    # 마지막 걸음의 할인 리턴은 1, 첫 걸음은 (1 - 0.99^L) / 0.01
    first = batch["t"] == 0
    # t=0의 G = (1 - gamma^L)/(1 - gamma), 분모 1-0.99 = 0.01
    np.testing.assert_allclose(batch["G"][first], (1 - 0.99 ** batch["lengths"]) / 0.01)


# ---------- PPO ----------
# 손으로 쓴 PPO 기울기(live 마스크 포함)를 autograd(rtol 1e-9)와 중앙차분(rtol 1e-5, 클리핑 경계 근처의 차분 오차 여유)과 비교. eps=inf는 클리핑 없음
def test_ppo_grad_numpy_matches_torch_and_finite_difference():
    w, batch, adv = sample_batch()
    w2 = w + np.random.default_rng(9).normal(0, 0.3, 5)
    for eps in (0.1, 0.2, float("inf")):
        ln, gn = m.ppo_loss_grad_w(w2, batch["phi"], batch["a"], batch["logp_old"], adv, eps)
        lt, gt = mt.loss_and_grad_w(w2, batch["phi"], batch["a"], batch["logp_old"], adv, eps)
        np.testing.assert_allclose(ln, lt, rtol=1e-9)
        np.testing.assert_allclose(gn, gt, rtol=1e-9, atol=1e-14)
        h = 1e-6
        fd = [(m.ppo_loss_grad_w(w2 + h * e, batch["phi"], batch["a"], batch["logp_old"], adv, eps)[0]
               - m.ppo_loss_grad_w(w2 - h * e, batch["phi"], batch["a"], batch["logp_old"], adv, eps)[0]) / (2 * h)
              for e in np.eye(5)]
        np.testing.assert_allclose(gn, fd, rtol=1e-5, atol=1e-8)


# 두 항등식: (1) 클리핑이 없으면 -기울기 = E[r grad log pi A](비율 가중 정책경사), (2) w = w_old이면 r=1이라 PPO 기울기 = 바닐라 정책경사
def test_no_clip_equals_ratio_weighted_pg_and_r1_equals_vanilla():
    w, batch, adv = sample_batch()
    w2 = w + 0.2
    _, g_inf = m.ppo_loss_grad_w(w2, batch["phi"], batch["a"], batch["logp_old"], adv, float("inf"))
    r = np.exp(m.log_prob(w2, batch["phi"], batch["a"]) - batch["logp_old"])
    ratio_pg = (r[:, None] * m.grad_log_prob(w2, batch["phi"], batch["a"]) * adv[:, None]).mean(0)
    np.testing.assert_allclose(-g_inf, ratio_pg, rtol=1e-12)
    _, g_r1 = m.ppo_loss_grad_w(w, batch["phi"], batch["a"], batch["logp_old"], adv, 0.2)  # w = w_old: r = 1
    np.testing.assert_allclose(-g_r1, m.pg_gradient(w, batch, adv), rtol=1e-12)


# 같은 미니배치 순서(시드 7)에서 Adam 40걸음 뒤 매개변수가 일치. 오차 누적 허용 rtol 1e-9
def test_ppo_update_numpy_matches_torch_adam():
    w, batch, adv = sample_batch()
    wn, _ = m.ppo_update(w, batch, adv, 0.2, 10, 4, m.Adam(5, 0.05), np.random.default_rng(7))
    wt = mt.ppo_update(w, batch, adv, 0.2, 10, 4, 0.05, np.random.default_rng(7))
    np.testing.assert_allclose(wn, wt, rtol=1e-9, atol=1e-12)


# ---------- KL, TRPO ----------
def test_kl_nonnegative_and_zero_on_itself():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1], size=200)
    q = rng.dirichlet([1, 1], size=200)
    assert np.all(m.kl_categorical(p, q) >= 0)
    np.testing.assert_allclose(m.kl_categorical(p, p), 0, atol=1e-15)
    np.testing.assert_allclose(m.kl_from_logp(np.log(p), np.log(q)), m.kl_categorical(p, q), rtol=1e-12)


# 세 가지 확인: 행렬 없는 F v = 만든 행렬 F @ v, torch 이중 미분 헤세-벡터 곱과 일치, CG(5차원이라 10회면 충분)가 직접 풀이 F^-1 g와 일치
def test_fisher_vector_product_and_conjugate_gradient():
    w, batch, adv = sample_batch()
    v = np.random.default_rng(5).normal(size=5)
    F = m.fisher_matrix(w, batch["phi"])
    np.testing.assert_allclose(m.fisher_vector_product(w, batch["phi"], v), F @ v, rtol=1e-12)
    np.testing.assert_allclose(mt.fisher_vector_product(w, batch["phi"], v), F @ v, rtol=1e-9)
    g = m.pg_gradient(w, batch, adv)
    x, k = m.conjugate_gradient(lambda u: F @ u, g, iters=10, tol=1e-30)
    np.testing.assert_allclose(x, np.linalg.solve(F, g), rtol=1e-6)
    assert k <= 10


def test_trpo_step_respects_kl_and_improves_surrogate():
    w, batch, adv = sample_batch()
    w_new, info = m.trpo_step(w, batch, adv, 0.01)
    assert info["accepted"]
    assert m.mean_kl(batch["logp_old_all"], w_new, batch["phi"]) <= 0.01
    # TRPO 걸음이 평균 KL <= delta=0.01을 지키면서 대리 목적을 개선했는지
    assert m.surrogate(w_new, batch, adv) > m.surrogate(w, batch, adv)


# ---------- 실패 재현 ----------
# 실패 재현: 학습률이 너무 크면 정책이 한쪽 행동 확률 1로 굳고(고른 행동 확률 > 0.999) 기울기도 0이라 못 돌아온다. 기대 숫자(20 미만 등)는 이 시드에서 관측한 값에 맞춘 문턱
def test_failure_large_lr_freezes_policy():
    """학습률 1000의 바닐라 정책경사, 시드 3: 몇 번 만에 한쪽으로 굳고 기울기가 0이 되어 돌아오지 못한다."""
    h = m.train("pg", 3, n_iters=20, lr=1000.0)
    assert np.mean(h["ret"][-10:]) < 20
    d = m.stuck_diagnosis(h["final_w"])
    assert d["mean_p_chosen"] > 0.999 and d["grad_norm"] < 1e-3


# 같은 시드와 큰 Adam 학습률에서 클리핑 유무만 바꿔 대조: 없으면 굳고(<20), eps=0.2면 150 초과(최대 200)
def test_clip_prevents_the_same_freeze():
    """같은 시드, 같은 Adam 학습률 0.2. 잘라 내기를 빼면 굳고, eps = 0.2면 굳지 않는다."""
    no = m.train("ppo", 3, n_iters=60, eps=float("inf"), adam_lr=0.2)
    yes = m.train("ppo", 3, n_iters=60, eps=0.2, adam_lr=0.2)
    assert np.mean(no["ret"][-10:]) < 20
    assert np.mean(yes["ret"][-10:]) > 150


def test_collapse_rule():
    steps = list(range(1000, 21000, 1000))
    up = [20, 60, 100] + [100] * 17
    assert m.collapsed(up, steps) is None
    # 최고 100의 50% = 50. 인덱스 3에서 30으로 떨어지고 1만 걸음 안에 못 돌아오므로 붕괴 시점 = 3
    fall = [20, 60, 100, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30] + [30] * 7
    assert m.collapsed(fall, steps) == 3
    # 한 번 떨어졌지만 다음 갱신에서 회복: 붕괴가 아니고 drop_events로는 1회로 센다(회복 여부를 따지지 않는 지표)
    dip = [20, 60, 100, 30, 90] + [100] * 15  # 1만 걸음 안에 50% 위로 돌아오면 붕괴가 아니다
    assert m.collapsed(dip, steps) is None
    assert m.drop_events(dip) == 1


def test_same_seed_same_numbers():
    a = m.train("ppo", 2, n_iters=8, eps=0.2, adam_lr=0.05)
    b = m.train("ppo", 2, n_iters=8, eps=0.2, adam_lr=0.05)
    assert a["ret"] == b["ret"] and a["final_w"] == b["final_w"]


# ---------- 통계 ----------
def test_t_distribution_p_values():
    # 직접 구현한 t분포 p값을 알려진 표의 값(임계값)과 비교. 코시(df=1) 값 0.5는 정확한 해석해
    assert m.t_two_sided_p(2.262, 9) == pytest.approx(0.05, abs=1e-3)   # t 표: 자유도 9, 양측 5%
    assert m.t_two_sided_p(1.96, 1e6) == pytest.approx(0.05, abs=1e-3)  # 자유도가 크면 정규분포
    assert m.t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert m.t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-12)     # 코시 분포: P(|T|>=1) = 1/2


def test_welch_matches_hand_formula():
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    t, df, p = m.welch_t(x, y)
    vx, vy = 2.5 / 5, 10.0 / 5
    # Welch t: 평균 차 / sqrt(s_x^2/n + s_y^2/n). 분산 x: 2.5, y: 10 (ddof=1), 각각 n=5로 나눔
    assert t == pytest.approx((3 - 6) / math.sqrt(vx + vy))
    assert df == pytest.approx((vx + vy) ** 2 / (vx**2 / 4 + vy**2 / 4))
    assert 0 < p < 1


def test_bootstrap_interval_contains_mean():
    x = np.random.default_rng(1).normal(10, 2, 10)
    mean, lo, hi = m.bootstrap_ci(x)
    assert lo < mean < hi
    assert mean == pytest.approx(x.mean())
