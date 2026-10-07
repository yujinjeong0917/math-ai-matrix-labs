"""10장 검증. `uv run pytest -q llm/ch10_train_gpt` 로 실행한다."""

import math

import numpy as np
import torch
import torch.nn.functional as F

import ch10_data as data
import ch10_np as cnp
from ch10_torch import MLPLM, TinyGPT, load_checkpoint, make_optimizer, save_checkpoint


def test_bias_correction_by_hand():
    """첫 스텝: m1 = 0.1·g, v1 = 0.001·g². 편향 보정 뒤 m̂1 = g, v̂1 = g² 이라 움직임은 lr·g/(|g|+ε)."""
    for g0 in (0.5, 0.001):
        p = [np.array([1.0])]
        st = cnp.adam_init(p)
        cnp.adam_step(st, p, [np.array([g0])], lr=0.01, eps=1e-8)
        assert math.isclose(st["m"][0][0], 0.1 * g0, rel_tol=1e-12)
        assert math.isclose(st["v"][0][0], 0.001 * g0 * g0, rel_tol=1e-12)
        assert math.isclose(1.0 - p[0][0], 0.01 * g0 / (g0 + 1e-8), rel_tol=1e-9)  # 거의 0.01, 기울기 크기와 무관
    # 둘째 스텝(같은 g): m2 = 0.19·g, 보정 분모 1 − 0.9² = 0.19 → m̂2 = g
    p = [np.array([1.0])]
    st = cnp.adam_init(p)
    for _ in range(2):
        cnp.adam_step(st, p, [np.array([0.5])], lr=0.01)
    assert math.isclose(st["m"][0][0], 0.19 * 0.5, rel_tol=1e-12)
    assert math.isclose(st["m"][0][0] / (1 - 0.9**2), 0.5, rel_tol=1e-12)


def test_clip_global_norm():
    rng = np.random.default_rng(0)
    grads = [rng.standard_normal((3, 4)) * 10, rng.standard_normal(5) * 10]
    clipped, before = cnp.clip_global_norm(grads, 1.0)
    assert before > 1.0
    assert cnp.global_norm(clipped) <= 1.0
    cos = sum((a * b).sum() for a, b in zip(grads, clipped)) / (before * cnp.global_norm(clipped))
    assert abs(cos - 1.0) < 1e-12  # 방향은 그대로
    small = [g * 1e-3 for g in grads]
    same, _ = cnp.clip_global_norm(small, 1.0)
    assert all(np.array_equal(a, b) for a, b in zip(small, same))  # 상한 안이면 그대로


def test_sgd_l2_equals_decoupled_but_adam_does_not():
    """Loshchilov·Hutter 명제 1·2: SGD에서는 L2 계수 λ/α가 분리형 감쇠 λ와 같지만, Adam에서는 아니다."""
    rng = np.random.default_rng(1)
    theta0 = rng.standard_normal(6)
    grads = [rng.standard_normal(6) for _ in range(5)]
    lr, lam = 0.1, 0.01
    a, b = [theta0.copy()], [theta0.copy()]
    for g in grads:
        cnp.sgd_step(a, [g], lr, l2=lam / lr)
        cnp.sgd_step(b, [g], lr, wd=lam)
    np.testing.assert_allclose(a[0], b[0], atol=1e-15)

    # Adam: L2 계수를 λ로 두든 λ/α로 두든 AdamW(λ)와 같은 궤적이 나오지 않는다
    for l2 in (lam, lam / lr):
        a, b = [theta0.copy()], [theta0.copy()]
        sa, sb = cnp.adam_init(a), cnp.adam_init(b)
        for g in grads:
            cnp.adam_step(sa, a, [g], lr, l2=l2)
            cnp.adamw_step(sb, b, [g], lr, wd=lam / lr)  # PyTorch식: 한 스텝 감쇠 lr·wd = λ
        assert np.abs(a[0] - b[0]).max() > 1e-4


def _mlp_setup():
    X, _ = data.make_windows(32, 16, seed=3)
    ctx, y = cnp.ngram_pairs(X, 3)
    p0 = cnp.init_mlp(data.V, 3, 8, 16, seed=0)
    return ctx, y, p0


def test_mlp_gradients_match_autograd():
    ctx, y, p0 = _mlp_setup()
    loss, grads = cnp.mlp_loss_and_grads(p0, ctx[:50], y[:50])
    m = MLPLM(p0)
    tl = F.cross_entropy(m(torch.tensor(ctx[:50])), torch.tensor(y[:50]))
    tl.backward()
    assert abs(loss - tl.item()) < 1e-12
    for g, p in zip(grads, (m.E, m.W1, m.b1, m.W2, m.b2)):
        np.testing.assert_allclose(g, p.grad.numpy(), atol=1e-12)


def _trajectory(kind, clip):
    """같은 기울기 열에서 NumPy 옵티마이저와 torch.optim의 10스텝 궤적 차이."""
    ctx, y, p0 = _mlp_setup()
    params = [p.copy() for p in p0]
    st = cnp.adam_init(params)
    m = MLPLM(p0)
    decay, nodecay = [m.E, m.W1, m.W2], [m.b1, m.b2]
    groups = [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}]
    opt = (torch.optim.AdamW if kind == "adamw" else torch.optim.Adam)(groups, lr=0.01, foreach=False)
    rng = np.random.default_rng(0)
    for _ in range(10):
        idx = rng.integers(0, len(y), 32)
        _, grads = cnp.mlp_loss_and_grads(params, ctx[idx], y[idx])
        if clip:
            grads, _ = cnp.clip_global_norm(grads, clip)
        if kind == "adamw":
            cnp.adamw_step(st, params, grads, 0.01, wd=0.1, decay_mask=cnp.DECAY_MASK)
        else:
            cnp.adam_step(st, params, grads, 0.01, l2=0.1, decay_mask=cnp.DECAY_MASK)
        loss = F.cross_entropy(m(torch.tensor(ctx[idx])), torch.tensor(y[idx]))
        opt.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
    return max(np.abs(a - p.detach().numpy()).max() for a, p in zip(params, (m.E, m.W1, m.b1, m.W2, m.b2)))


def test_numpy_optimizers_match_torch_trajectory():
    for kind in ("adamw", "adam_l2"):
        for clip in (None, 0.5):
            assert _trajectory(kind, clip) < 1e-10


def test_torch_adam_weight_decay_is_l2():
    """PyTorch 2.8: Adam(weight_decay=λ)는 손실에 (λ/2)||θ||²를 더한 Adam과 같은 궤적이다."""
    torch.manual_seed(0)
    w0 = torch.randn(5, dtype=torch.float64)
    xs = [torch.randn(5, dtype=torch.float64) for _ in range(5)]
    a = w0.clone().requires_grad_()
    b = w0.clone().requires_grad_()
    oa = torch.optim.Adam([a], lr=0.01, weight_decay=0.3, foreach=False)
    ob = torch.optim.Adam([b], lr=0.01, foreach=False)
    for x in xs:
        oa.zero_grad()
        ((a * x).sum() ** 2).backward()
        oa.step()
        ob.zero_grad()
        ((b * x).sum() ** 2 + 0.15 * (b * b).sum()).backward()
        ob.step()
    assert torch.allclose(a, b, atol=1e-12)


def _tiny_run(seed):
    torch.manual_seed(seed)
    model = TinyGPT(data.V, 16, 32, 2, 4)
    opt = make_optimizer("adamw", model, 3e-3, 0.1)
    X = torch.tensor(data.make_windows(16, 16, seed=5)[0])
    losses = []
    for _ in range(5):
        logits = model(X[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, data.V), X[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return model, losses


def test_fixed_seed_rerun_is_identical():
    torch.set_num_threads(1)
    _, l1 = _tiny_run(0)
    _, l2 = _tiny_run(0)
    assert l1 == l2


def test_weight_decay_only_on_matrices():
    model = TinyGPT(data.V, 16, 32, 2, 4)
    opt = make_optimizer("adamw", model, 1e-3, 0.1)
    decay, nodecay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() < 2 for p in nodecay["params"]) and nodecay["weight_decay"] == 0.0


def test_checkpoint_roundtrip(tmp_path):
    model, _ = _tiny_run(0)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, data.VOCAB, {"note": "test"})
    m2, ck = load_checkpoint(path)
    x = torch.tensor([[0, 3, 11, 25, 34]])
    assert torch.equal(model(x), m2(x))
    assert ck["vocab"] == data.VOCAB


def test_grammar_and_entropy_floor():
    rng = np.random.default_rng(0)
    for _ in range(200):
        toks, ents = data.sample_sentence(rng)
        assert toks[-1] == "." and len(toks) == len(ents)
        assert data.is_valid_sentence(toks[:-1])
    assert not data.is_valid_sentence(["고양이가", "책을", "읽었다"])  # 동물은 책을 읽지 않는다
    assert not data.is_valid_sentence(["민수가", "공원에", "먹었다"])
    # 첫 토큰 엔트로피 손계산: 부사 3개(각 1/6) + 주어 8개(각 1/16)
    h = -(3 * (1 / 6) * math.log(1 / 6) + 8 * (1 / 16) * math.log(1 / 16))
    assert abs(data.H_FIRST - h) < 1e-12
