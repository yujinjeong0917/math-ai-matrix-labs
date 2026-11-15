"""강화학습 연결 장 검증. `uv run pytest -q rl/chbridge_rlhf_dpo`"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_chbridge_rlhf_dpo_np as m  # noqa: E402
import rl_chbridge_rlhf_dpo_torch as mt  # noqa: E402

torch.set_num_threads(2)


# 학습용 선호 쌍 n=500개. 난수열 [seed, 1]은 experiments.py의 pairs_for와 같아서 그 쪽 숫자와 대응한다
def pairs(seed=0, n=500):
    return m.make_pairs(n, np.random.default_rng([seed, 1]))


# ---------- 첫 화면 손계산 ----------
# 본문 첫 화면 손계산의 숫자를 코드로 검증. 아래 기대값은 시드 0의 학습 점수를 소수 둘째 자리로 반올림한 u에서 나온다
def test_hand_example_numbers():
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    assert round(sig(1.0), 4) == 0.7311
    # 바 8개: 두 번까지 인정 2 * 1.0, 나머지 6번은 벌점 6 * 1.0 -> -4
    assert m.true_reward(np.array([[5] * 8]))[0] == pytest.approx(2 * 1.0 - 6 * 1.0)  # 바x8 = -4
    # 바,마,라,다 각 두 번: 2(1.0+0.8+0.6+0.4) = 5.6 (공간 전체의 최댓값)
    assert m.true_reward(np.array([[5, 5, 4, 4, 3, 3, 2, 2]]))[0] == pytest.approx(5.6)
    tab = m._enum_tables()
    assert tab["r_true"].max() == pytest.approx(5.6)
    # 균등 기준 정책의 평균 참 보상(전수 평균)
    assert round(tab["r_true"].mean(), 4) == 2.4648
    # 시드 0의 단어 점수(소수 둘째 자리)로 계산한 학습 점수
    u = np.round(m.bt_reward_model_fit(*pairs(0)), 2)
    assert u.tolist() == [-0.19, -0.18, -0.03, 0.09, 0.15, 0.16]
    assert 8 * u[5] == pytest.approx(1.28)
    assert m.counts(np.array([[5, 5, 4, 4, 3, 3, 2, 2]]))[0] @ u == pytest.approx(0.74)
    # 단어별 기울인 확률: 확률 ∝ e^(점수/beta)
    # 단어별 기울인 확률 ∝ exp(점수/beta). beta=0.1이면 점수 차이가 10배로 증폭된다
    w = np.exp(u / 0.1)
    assert round(float(w[5] / w.sum()), 4) == 0.3825
    assert round(math.exp(1.6), 2) == 4.95 and round(math.exp(-1.9), 2) == 0.15
    assert round(float(w.sum()), 2) == 12.95
    assert round(float(w[0] / w.sum()) * 100, 2) == 1.15
    w1 = np.exp(u / 1.0)
    assert round(float(w1[5] / w1.sum()), 4) == 0.1936 and round(float(w1[0] / w1.sum()), 4) == 0.1364
    # 연습 1, 2의 정답 기준
    w3 = np.exp(u / 0.3)
    assert round(float(w3[5] / w3.sum()), 3) == 0.255
    assert m.true_reward(np.array([[4] * 8]))[0] == pytest.approx(-4.4)
    assert 8 * u[4] == pytest.approx(1.2)
    assert round(float(m.rlvr_reward(m.all_sequences()).mean()), 4) == round(1 / 6, 4)


# ---------- 열거와 정책 ----------
# 전수 열거의 확률 질량 합이 1이고 균등 정책의 KL이 0인지
def test_enumeration_and_uniform_reference():
    st = m.exact_stats(m.zeros_theta())
    assert st["mass"] == pytest.approx(1.0, abs=1e-12)
    assert st["kl"] == pytest.approx(0.0, abs=1e-12)
    # 한 문장에 확률이 전부 몰리면 KL = 8 log 6
    th = np.full((m.L, m.V + 1, m.V), -60.0)
    th[:, :, 5] = 60.0
    # 한 문장에 전부 몰리면 KL = log(1 / (1/6^8)) = 8 log 6
    assert m.exact_stats(th)["kl"] == pytest.approx(8 * math.log(6), rel=1e-9)


def test_kl_nonnegative_and_zero_on_itself():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, (2, m.L, m.V + 1, m.V))
    assert m.kl_between(a, b) > 0
    assert m.kl_between(a, a) == pytest.approx(0.0, abs=1e-12)


# 손으로 쓴 기울기(식 onehot - p)가 PyTorch 자동미분과 일치하는지(허용 오차 rtol 1e-9)
def test_grad_logprob_matches_autograd():
    rng = np.random.default_rng(2)
    th = rng.normal(0, 0.7, (m.L, m.V + 1, m.V))
    Y = m.sample(th, 50, rng)
    w = rng.normal(0, 1, 50)
    tt = torch.tensor(th, requires_grad=True)
    (mt.logprob(tt, Y) @ torch.tensor(w)).backward()
    np.testing.assert_allclose(m.grad_logprob(th, Y, w), tt.grad.numpy(), rtol=1e-9, atol=1e-12)


# ---------- Bradley-Terry ----------
# 해석적 기울기를 중앙차분(h=1e-6)과 비교. 중앙차분 오차 O(h^2)와 반올림 오차 때문에 rtol 1e-6
def test_bt_gradient_finite_difference():
    Yw, Yl = pairs(0, 200)
    u = np.random.default_rng(3).normal(0, 0.5, m.V)
    _, g = m.bt_loss(u, Yw, Yl)
    h = 1e-6
    fd = np.array([(m.bt_loss(u + h * e, Yw, Yl)[0] - m.bt_loss(u - h * e, Yw, Yl)[0]) / (2 * h) for e in np.eye(m.V)])
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)


def test_bt_numpy_matches_torch():
    Yw, Yl = pairs(1, 300)
    u = np.random.default_rng(4).normal(0, 0.5, m.V)
    loss, g = m.bt_loss(u, Yw, Yl)
    ut = torch.tensor(u, requires_grad=True)
    lt = mt.bt_loss(ut, Yw, Yl)
    lt.backward()
    assert loss == pytest.approx(lt.item(), rel=1e-9)
    np.testing.assert_allclose(g, ut.grad.numpy(), rtol=1e-9, atol=1e-12)


# 뉴턴법이 끝난 점에서 기울기가 거의 0(< 1e-10)인지 = 최소점에 도달
def test_reward_model_fit_is_stationary():
    Yw, Yl = pairs(0)
    u = m.bt_reward_model_fit(Yw, Yl)
    assert np.abs(m.bt_loss(u, Yw, Yl)[1]).max() < 1e-10


# ---------- KL 정규 최적해 ----------
def test_tilt_theta_equals_enumerated_tilt():
    u = m.bt_reward_model_fit(*pairs(0))
    for beta in (1.0, 0.1):
        a = m.exact_stats(m.tilt_theta_linear(u, beta), u)
        b = m.tilt_stats(m._enum_tables()["counts"] @ u, beta)
        assert a["kl"] == pytest.approx(b["kl"], rel=1e-9)
        assert a["r_true"] == pytest.approx(b["r_true"], rel=1e-9)


# 식: 목적 E[r_hat] - beta KL의 최대점이 pi ∝ pi_ref exp(r_hat/beta). 기울기 0과 주변 섭동보다 높음을 함께 확인(RLHF 최적해 공식의 검증)
def test_tilt_is_stationary_point_of_kl_regularized_objective():
    """pi ∝ pi_ref exp(r_hat / beta)에서 E[r_hat] - beta KL의 기울기가 0이고, 주변보다 높다."""
    u = m.bt_reward_model_fit(*pairs(0))
    beta = 0.3
    th = torch.tensor(m.tilt_theta_linear(u, beta), requires_grad=True)
    obj = mt.kl_regularized_objective(th, u, beta)
    obj.backward()
    assert th.grad.abs().max().item() < 1e-10
    rng = np.random.default_rng(5)
    for _ in range(3):
        pert = torch.tensor(m.tilt_theta_linear(u, beta) + rng.normal(0, 0.1, th.shape))
        assert mt.kl_regularized_objective(pert, u, beta).item() < obj.item()


# ---------- DPO ----------
def test_dpo_numpy_matches_torch():
    Yw, Yl = pairs(2, 300)
    th = np.random.default_rng(6).normal(0, 0.7, (m.L, m.V + 1, m.V))
    for beta in (0.01, 0.1, 1.0):
        loss, g = m.dpo_grad_theta(th, Yw, Yl, beta)
        tt = torch.tensor(th, requires_grad=True)
        lt = mt.dpo_loss_theta(tt, Yw, Yl, beta)
        lt.backward()
        assert loss == pytest.approx(lt.item(), rel=1e-9)
        np.testing.assert_allclose(g, tt.grad.numpy(), rtol=1e-9, atol=1e-14)


# DPO의 핵심 항등식: 암묵 보상 beta log(pi/pi_ref) = r_hat - beta log Z이고 Z는 쌍 차이에서 상쇄되므로 손실이 BT 손실과 같다. l2=0이라야 같다
def test_dpo_loss_at_tilted_policy_equals_bt_loss():
    """분배함수가 빠지는 자리: pi = pi_ref exp(r_hat/beta)/Z를 DPO 손실에 넣으면 r_hat의 BT 손실과 같다."""
    Yw, Yl = pairs(0)
    u = m.bt_reward_model_fit(Yw, Yl)
    for beta in (0.05, 0.5, 2.0):
        th = m.tilt_theta_linear(u, beta)
        loss_dpo, _ = m.dpo_grad_theta(th, Yw, Yl, beta)
        loss_bt, _ = m.bt_loss(u, Yw, Yl, l2=0.0)
        assert loss_dpo == pytest.approx(loss_bt, rel=1e-12)


def test_dpo_shared_converges_to_rlhf_optimum():
    """표현력을 단어 개수 모델과 같게 묶으면 DPO의 해와 RLHF의 정확한 최적해가 같은 정책이다."""
    Yw, Yl = pairs(0)
    u = m.bt_reward_model_fit(Yw, Yl, l2=1e-10)
    for beta in (0.1, 1.0):
        th, _ = m.train_dpo_shared(Yw, Yl, beta, steps=3000, lr=0.05)
        # 표현력을 맞추면 DPO의 해 = RLHF의 정확한 최적해. 3000걸음 수치 최적화라 KL 허용 1e-5(정확한 0은 아님)
        assert m.kl_between(th, m.tilt_theta_linear(u, beta)) < 1e-5


# 같은 Adam 설정이면 30걸음 뒤 매개변수가 일치(오차 누적을 허용하는 atol 1e-10)
def test_dpo_training_numpy_matches_torch():
    Yw, Yl = pairs(3)
    th_np, _ = m.train_dpo(Yw, Yl, 0.1, steps=30, lr=0.05, eval_every=10**9)
    th_t = mt.train_dpo(Yw, Yl, 0.1, steps=30, lr=0.05)
    np.testing.assert_allclose(th_np, th_t, rtol=1e-9, atol=1e-10)


def test_dpo_kl_keeps_growing_with_steps():
    """beta = 1에서도 500쌍 DPO의 KL은 걸음 수와 함께 계속 커진다(같은 beta의 RLHF 최적해는 KL 0.1 안팎)."""
    Yw, Yl = pairs(0)
    _, c = m.train_dpo(Yw, Yl, 1.0, steps=400, lr=0.01, eval_every=200)
    assert c[0]["kl"] < 1e-12 < c[1]["kl"] < c[2]["kl"]
    u = m.bt_reward_model_fit(Yw, Yl)
    assert c[2]["kl"] > 10 * m.tilt_stats(m._enum_tables()["counts"] @ u, 1.0)["kl"]


# ---------- 실패 재현: 보상 모델 과최적화 ----------
# KL 벌점 0(beta=0)이면 보상 모델 점수만 쫓다가 참 보상이 무너지는 현상(보상 과최적화)을 재현
def test_overoptimization_without_kl():
    u = m.bt_reward_model_fit(*pairs(0))
    _, c = m.kl_regularized_pg(u, 0.0, 0, steps=600, lr=0.01, eval_every=20)
    peak = max(c, key=lambda s: s["r_true"])
    assert peak["r_true"] > c[0]["r_true"]  # 처음에는 참 보상도 오른다
    assert c[-1]["r_hat"] > peak["r_hat"]  # 학습한 점수는 계속 오르는데
    assert c[-1]["r_true"] < c[0]["r_true"] - 2  # 참 보상은 출발점보다 아래로 떨어진다
    assert c[-1]["uniq"] < 2.0  # 같은 단어 반복으로 무너진다


def test_kl_penalty_holds_at_beta1_and_reaches_exact_optimum():
    u = m.bt_reward_model_fit(*pairs(0))
    th, c = m.kl_regularized_pg(u, 1.0, 0, steps=600, lr=0.01, eval_every=100)
    assert max(s["r_true"] for s in c) - c[-1]["r_true"] < 0.01
    assert m.kl_between(th, m.tilt_theta_linear(u, 1.0)) < 1e-4


# ---------- RLVR ----------
# 검증기 보상 확인: 문제 a에 대해 답 (a+2) mod 6. 기본 성공률 1/6(균등 정책)에서 100걸음 후 0.9 초과
def test_rlvr_verifier_and_learning():
    Y = np.array([[1, 3, 0, 0, 0, 0, 0, 0], [5, 1, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0, 0]])
    assert m.rlvr_reward(Y).tolist() == [1.0, 1.0, 0.0]  # 1+2=3, (5+2) mod 6 = 1
    th, c = m.rlvr_pg(0, steps=100, lr=0.05, eval_every=100)
    assert c[0]["success"] == pytest.approx(1 / 6)
    assert c[-1]["success"] > 0.9
    np.testing.assert_array_equal(th[0], 0.0)  # 문제 분포는 그대로


# 같은 시드 -> 같은 난수열 -> 정확히 같은 숫자(허용 오차 없는 == 비교)
def test_same_seed_same_numbers():
    u = m.bt_reward_model_fit(*pairs(0))
    _, c1 = m.kl_regularized_pg(u, 0.0, 0, steps=50, lr=0.01, eval_every=50)
    _, c2 = m.kl_regularized_pg(u, 0.0, 0, steps=50, lr=0.01, eval_every=50)
    assert [x["r_true"] for x in c1] == [x["r_true"] for x in c2]
