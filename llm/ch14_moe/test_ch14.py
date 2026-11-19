"""14장 검증. 이 폴더에서 `python -m pytest -q` 로 실행한다(저장소에 옮기면 `uv run pytest -q llm/ch14_moe`)."""

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import moe_np as mn
import moe_torch as mt

MIXTRAL = {"d": 4096, "L": 32, "dh": 128, "ff": 14336, "h": 32, "kv": 8, "V": 32000}


def test_mixtral_total_and_active_round_to_paper_values():
    # Jiang 등(2024) 초록: 토큰은 47B 파라미터에 접근하고 추론에서 13B만 쓴다
    tot, act = mn.model_params(MIXTRAL, E=8, k=2)
    assert round(tot / 1e9) == 47 and round(act / 1e9) == 13
    # 손으로 센 값과 정확히 같다(식 그대로): 어텐션 41,943,040, 전문가 하나 176,160,768
    assert mn.params_attn(4096, 32, 8, 128) == 41_943_040
    assert mn.params_ffn(4096, 14336) == 176_160_768
    assert tot == 46_702_526_464 and act == 12_879_659_008


def test_dense_equivalent_is_about_7b():
    tot, act = mn.model_params(MIXTRAL, E=1, k=1)
    assert tot == act and round(tot / 1e9, 1) == 7.2


def test_topk_gate_rows_sum_to_one_and_have_k_nonzero():
    rng = np.random.default_rng(0)
    G, idx = mn.topk_gate(rng.normal(size=(50, 8)), 2)
    assert np.allclose(G.sum(axis=1), 1.0)
    assert ((G > 0).sum(axis=1) == 2).all()
    assert idx.shape == (50, 2)


def test_numpy_gate_matches_torch_gate():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(40, 6))
    G, _ = mn.topk_gate(logits, 2)
    tv, ti = torch.tensor(logits).topk(2, dim=-1)
    gt = torch.zeros(40, 6, dtype=torch.float64).scatter(1, ti, torch.softmax(tv, dim=-1))
    assert np.allclose(G, gt.numpy())


import pytest


@pytest.mark.parametrize("GATE", ["renorm", "switch"])
def test_numpy_moe_forward_matches_torch_module(GATE):
    torch.manual_seed(0)
    m = mt.MoE(E=4, k=2, h=5, gate=GATE)
    x = torch.randn(30, mt.D_IN)
    with torch.no_grad():
        y_t, _, _ = m(x)
    Wg = m.router.weight.detach().numpy().T.astype(np.float64)

    def make(ex):
        W1, b1 = ex.net[0].weight.detach().numpy().astype(np.float64), ex.net[0].bias.detach().numpy().astype(np.float64)
        W2, b2 = ex.net[2].weight.detach().numpy().astype(np.float64), ex.net[2].bias.detach().numpy().astype(np.float64)
        return lambda z: np.maximum(z @ W1.T + b1, 0) @ W2.T + b2

    y_n, _, calls = mn.moe_forward(x.numpy().astype(np.float64), Wg, [make(e) for e in m.experts], 2, gate=GATE)
    assert np.allclose(y_n, y_t.numpy(), atol=1e-5)
    assert calls.sum() == 30 * 2  # 토큰 하나가 정확히 k명을 부른다


def test_balance_loss_uniform_equals_alpha_and_grows_with_skew():
    E = 4
    uniform = np.full((8, E), 1 / E)
    uniform[np.arange(8), np.arange(8) % E] += 1e-9  # argmax를 고르게 나눈다
    loss_u, f, P = mn.balance_loss(uniform, alpha=1.0)
    assert np.isclose(loss_u, 1.0) and np.allclose(f, 0.25)
    skew = np.tile([0.7, 0.1, 0.1, 0.1], (8, 1))
    loss_s, f2, _ = mn.balance_loss(skew, alpha=1.0)
    assert loss_s > loss_u and np.isclose(loss_s, 4 * 0.7)  # f=(1,0,0,0), P=(0.7,...) -> 4 * 1 * 0.7


def test_numpy_balance_loss_matches_torch():
    rng = np.random.default_rng(2)
    probs = np.exp(rng.normal(size=(64, 8)))
    probs /= probs.sum(axis=1, keepdims=True)
    ln, _, _ = mn.balance_loss(probs, alpha=0.01)
    pt = torch.tensor(probs)
    lt = mt.balance_loss(pt, pt.argmax(dim=1, keepdim=True), 8, 0.01)
    assert np.isclose(ln, lt.item())


def test_capacity_and_dropped_tokens():
    assert mn.expert_capacity(1024, 8, 1.25) == 160
    assign = np.array([0] * 6 + [1] * 2)  # 전문가 2명, 용량 4
    assert mn.dropped_tokens(assign, 2, 4) == 2


def test_flop_counter_matches_formula():
    T = 64
    x = torch.randn(T, mt.D_IN)
    with FlopCounterMode(display=False) as fc:
        mt.Dense(16)(x)
    assert fc.get_total_flops() == T * (2 * mt.D_IN * 16 + 2 * 16 * mt.D_OUT)
    with FlopCounterMode(display=False) as fc:
        mt.MoE(8, 1, 16)(x)
    # 토큰이 전문가 한 명만 거치므로 밀집 h=16과 같고, 라우터 2·d·E만 더해진다
    assert fc.get_total_flops() == T * (2 * mt.D_IN * 16 + 2 * 16 * mt.D_OUT) + T * 2 * mt.D_IN * 8


def test_kv_bytes():
    assert mn.kv_bytes_per_token(32, 8, 128, 2) == 131072
    assert mn.kv_bytes_per_token(32, 32, 128, 2) == 4 * mn.kv_bytes_per_token(32, 8, 128, 2)
    assert mn.kv_bytes_per_token(32, 1, 128, 2) * 8 == mn.kv_bytes_per_token(32, 8, 128, 2)


def test_k1_renorm_gate_is_constant_one_and_router_gets_no_task_gradient():
    torch.manual_seed(0)
    x, y = torch.randn(64, mt.D_IN), torch.randn(64, mt.D_OUT)
    m = mt.MoE(8, 1, 16, gate="renorm")
    out, probs, top = m(x)
    # 상위 1개에만 softmax하면 게이트가 늘 1이라 라우터 가중치로 가는 기울기가 없다(Shazeer 등 2017이 k > 1을 말한 까닭)
    ((out - y) ** 2).mean().backward()
    assert m.router.weight.grad is None or float(m.router.weight.grad.abs().max()) == 0.0
    m2 = mt.MoE(8, 1, 16, gate="switch")
    out2, _, _ = m2(x)
    ((out2 - y) ** 2).mean().backward()
    assert float(m2.router.weight.grad.abs().max()) > 0.0  # Switch식은 p_i(x)가 곱해져서 기울기가 흐른다


def test_switch_gate_rows_do_not_sum_to_one_but_equal_full_softmax_entries():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(20, 8))
    G, idx = mn.switch_gate(logits, 1)
    p = mn.softmax(logits)
    assert ((G > 0).sum(axis=1) == 1).all() and (G.sum(axis=1) < 1).all()
    assert np.allclose(G.sum(axis=1), p.max(axis=1))
