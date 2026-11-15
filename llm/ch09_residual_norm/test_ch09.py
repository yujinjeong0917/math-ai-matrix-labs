"""9장 검증. `uv run pytest -q llm/ch09_residual_norm` 로 실행한다."""

import numpy as np
import pytest
import torch

import resnorm_np as rn
import resnorm_torch as rt


def _params(d=8, ff=4, seed=0):
    # 시드 고정. W*는 1/sqrt(d), W1은 He(2/d), 정규화 gamma는 1 근처, beta는 0 근처로 무작위
    rng = np.random.default_rng(seed)
    p = {k: rng.standard_normal((d, d)) / np.sqrt(d) for k in ("Wq", "Wk", "Wv", "Wo")}
    p["W1"], p["b1"] = rng.standard_normal((d, ff * d)) * np.sqrt(2 / d), rng.standard_normal(ff * d) * 0.1
    p["W2"], p["b2"] = rng.standard_normal((ff * d, d)) / np.sqrt(ff * d), rng.standard_normal(d) * 0.1
    for g, b in (("g1", "be1"), ("g2", "be2")):
        p[g], p[b] = 1 + 0.1 * rng.standard_normal(d), 0.1 * rng.standard_normal(d)
    return p


def test_layer_norm_feature_axis_mean0_var1():
    # 평균 3, 표준편차 7로 흩어진 입력. LN은 위치(행)마다 따로 정규화한다
    x = np.random.default_rng(0).standard_normal((5, 16)) * 7 + 3
    y = rn.layer_norm(x, np.ones(16), np.zeros(16))
    # 각 행 평균 0
    np.testing.assert_allclose(y.mean(-1), 0.0, atol=1e-12)
    # 분산은 eps 때문에 정확히 1보다 약간 작다. 허용 오차 1e-5
    np.testing.assert_allclose(y.var(-1), 1.0, atol=1e-5)  # eps=1e-5 만큼만 1에서 벗어난다


def test_layer_norm_hand_example():
    """첫 화면의 손 계산: [1, 3, 5, 7] -> 평균 4, 분산 5, 결과 (-3, -1, 1, 3)/sqrt(5)."""
    # eps=0으로 손계산과 정확히 맞춘다: 평균 4, 분산 (9+1+1+9)/4 = 5
    y = rn.layer_norm(np.array([1.0, 3.0, 5.0, 7.0]), 1.0, 0.0, eps=0.0)
    np.testing.assert_allclose(y, np.array([-3, -1, 1, 3]) / np.sqrt(5), atol=1e-12)
    # -3/sqrt(5) = -1.3416, -1/sqrt(5) = -0.4472 반올림
    np.testing.assert_allclose(np.round(y, 2), [-1.34, -0.45, 0.45, 1.34])
    # 층 10개에서 배율 0.5가 누적되면 약 0.001, 1.5면 약 58. 첫 화면 예(장 본문 대응은 확인 필요)
    assert round(0.5**10, 4) == 0.001 and round(1.5**10) == 58


def test_layer_norm_backward_matches_finite_difference():
    rng = np.random.default_rng(1)
    x, g, b = rng.standard_normal((3, 6)), rng.standard_normal(6), rng.standard_normal(6)
    dy = rng.standard_normal((3, 6))
    dx, dg, db = rn.layer_norm_backward(dy, x, g)
    # 손실 대리: sum(LN(z) * dy). dy가 상류 기울기이므로 이 함수의 기울기가 dx
    f = lambda z: float((rn.layer_norm(z, g, b) * dy).sum())  # noqa: E731
    num = np.zeros_like(x)
    eps = 1e-6
    for idx in np.ndindex(*x.shape):
        e = np.zeros_like(x)
        e[idx] = eps
        # 중앙차분: (f(x+e) - f(x-e)) / 2 eps, 원소마다 하나씩
        num[idx] = (f(x + e) - f(x - e)) / (2 * eps)
    np.testing.assert_allclose(dx, num, atol=1e-7)
    # torch autograd 와도 대조
    xt = torch.tensor(x, requires_grad=True)
    gt, bt = torch.tensor(g, requires_grad=True), torch.tensor(b, requires_grad=True)
    # autograd로 같은 손실의 기울기를 구해 해석식과 대조(eps는 NumPy와 같은 1e-5)
    (torch.nn.functional.layer_norm(xt, (6,), gt, bt, 1e-5) * torch.tensor(dy)).sum().backward()
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(dg, gt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(db, bt.grad.numpy(), atol=1e-10)


def test_one_head_mha_equals_chapter7_self_attention():
    p = _params()
    x = np.random.default_rng(2).standard_normal((7, 8))
    # h=1, W_O=단위행렬이면 멀티헤드는 7장의 단일 헤드 어텐션과 같아야 한다
    y1 = rn.multi_head_attention(x, p["Wq"], p["Wk"], p["Wv"], np.eye(8), h=1)
    y7 = rn.self_attention(x, p["Wq"], p["Wk"], p["Wv"], causal=True)
    np.testing.assert_allclose(y1, y7, atol=1e-12)


def test_numpy_mha_matches_torch_builtin():
    p = _params()
    x = np.random.default_rng(3).standard_normal((7, 8))
    ref = rn.multi_head_attention(x, p["Wq"], p["Wk"], p["Wv"], p["Wo"], h=2)
    mha = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True).double()
    with torch.no_grad():  # 내장 모듈은 x @ W.T 로 쓰므로 전치해서 넣는다
        # in_proj_weight: (3d, d)로 Q, K, V 가중치를 위아래로 쌓은 것. x @ W.T 형태라 전치해서 넣는다
        mha.in_proj_weight.copy_(torch.tensor(np.concatenate([p["Wq"].T, p["Wk"].T, p["Wv"].T])))
        mha.out_proj.weight.copy_(torch.tensor(p["Wo"].T))
    xt = torch.tensor(x)[None]
    # attn_mask: True인 칸을 가림(위쪽 삼각). batch_first라 입력은 (1, 7, 8)
    out, _ = mha(xt, xt, xt, attn_mask=rt.causal_bool_mask(7), need_weights=False)
    np.testing.assert_allclose(out[0].detach().numpy(), ref, atol=1e-10)


@pytest.mark.parametrize("arch", rn.ARCHS)
def test_numpy_block_matches_torch_block(arch):
    p = _params()
    x = np.random.default_rng(4).standard_normal((7, 8))
    ref = rn.block(x, p, arch, h=2)
    out = rt.block_from_numpy(p, 8, 2, arch)(torch.tensor(x)[None])[0].detach().numpy()
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_residual_keeps_identity_when_sublayer_is_zero():
    """F = 0 이면 잔차 블록(Pre-LN 포함)은 입력을 그대로 내보낸다. 잔차 없는 블록은 그렇지 않다."""
    p = _params()
    for k in ("Wo", "W2", "b2"):
        p[k] = np.zeros_like(p[k])
    x = np.random.default_rng(5).standard_normal((7, 8))
    # Wo = W2 = b2 = 0이면 두 부분층이 0을 내놓아 Pre-LN과 residual은 x + 0 = x
    np.testing.assert_allclose(rn.block(x, p, "pre", h=2), x, atol=1e-12)
    np.testing.assert_allclose(rn.block(x, p, "residual", h=2), x, atol=1e-12)
    # 잔차 없는 plain은 0을 내놓으므로 x와 달라야 한다
    assert not np.allclose(rn.block(x, p, "plain", h=2), x)


def test_batch_norm_depends_on_batch_layer_norm_does_not():
    rng = np.random.default_rng(6)
    g, b = np.ones(4), np.full(4, 0.5)
    one = rng.standard_normal((1, 4))
    # 배치 크기 1: 평균이 자기 자신이라 x - mu = 0, 출력은 beta뿐
    np.testing.assert_allclose(rn.batch_norm(one, g, b), b[None])  # 배치 1: x - 평균 = 0, 출력은 beta뿐
    # nn.BatchNorm1d는 학습 모드에서 배치 1(표본당 값 1개)이면 ValueError. 버전에 따라 다를 수 있음(확인 필요)
    with pytest.raises(ValueError):
        torch.nn.BatchNorm1d(4).train()(torch.tensor(one, dtype=torch.float32))
    a = np.vstack([one, rng.standard_normal((5, 4))])
    # 같은 첫 표본에 다른 친구(평균 +3)를 붙인다
    c = np.vstack([one, rng.standard_normal((5, 4)) + 3])
    # BN: 친구가 바뀌면 같은 표본의 출력이 바뀐다
    assert not np.allclose(rn.batch_norm(a, g, b)[0], rn.batch_norm(c, g, b)[0])
    # LN: 위치(행) 단위라 친구와 무관, 정확히 같아야 한다
    np.testing.assert_allclose(rn.layer_norm(a, g, b)[0], rn.layer_norm(c, g, b)[0], atol=1e-15)
    # running 통계 없이 training=True로 직접 호출해 NumPy 구현과 대조
    t = torch.nn.functional.batch_norm(torch.tensor(a), None, None, torch.tensor(g), torch.tensor(b), training=True)
    np.testing.assert_allclose(t.numpy(), rn.batch_norm(a, g, b), atol=1e-10)


def test_flop_counts():
    # 손계산: 어텐션 투영 8 + 점수·혼합 4 + MLP 16 = 28
    assert rn.block_flops(1, 1) == 8 + 4 + 16
    assert rn.residual_flops(16, 32) == 2 * 16 * 32
    # d=768, n=1024: 잔차+LN이 블록 연산의 1% 미만
    assert (rn.residual_flops(1024, 768) + rn.layer_norm_flops(1024, 768)) / rn.block_flops(1024, 768) < 0.01
