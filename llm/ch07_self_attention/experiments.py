"""7장 비교 실험. `uv run python llm/ch07_self_attention/experiments.py` 로 실행한다.

E1 거리와 기울기: RNN의 ||dh_t/dh_0|| vs self-attention의 ||dy_n/dx_1||
E2 계산량: 순차 RNN 루프 vs 행렬곱 한 번의 attention (이론 FLOP + 실측 중앙값)
E3 sqrt(d_k) 스케일: 점수 분산, softmax 최대 가중치, softmax 야코비안 노름
E4 학습 안정성: 첫 토큰 기억 과제에서 RNN / LSTM / self-attention, 시드 3개
E4b 공정성 점검: LSTM을 5배 더 오래, forget 편향 1 초기화로도 학습
E4c 경계 찾기: RNN과 LSTM이 버티는 길이
E5 스케일 유무: d_k가 클 때 scale=True/False 학습 비교, 시드 3개

결과는 results/ch07.json 에 저장한다. 수치는 CPU·버전에 따라 조금씩 달라질 수 있고,
챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import attention_np as anp
from attention_torch import AttentionClassifier, RNNClassifier

OUT = Path(__file__).parent / "results" / "ch07.json"
SIGNALS, NOISE = 2, 16  # 토큰 0, 1은 신호(첫 위치에만 등장), 2..17은 잡음
VOCAB = SIGNALS + NOISE


def e1_distance():
    rng = np.random.default_rng(0)
    d, n = 32, 200
    x = rng.standard_normal((n, d))
    U = rng.standard_normal((d, d)) * 0.1
    base = rng.standard_normal((d, d))
    base /= np.max(np.abs(np.linalg.eigvals(base)))  # 스펙트럼 반경 1로 정규화
    rnn = {}
    for rho in (0.8, 1.0, 1.5):
        norms = anp.rnn_state_jacobian_norms(x, base * rho, U)
        rnn[str(rho)] = {str(t): float(norms[t - 1]) for t in (1, 10, 50, 100, 200)}

    from attention_torch import self_attention

    torch.manual_seed(0)
    W = [torch.randn(d, d) / d**0.5 for _ in range(3)]  # 가중치는 고정하고 입력만 20번 새로 뽑는다
    attn = {}
    for length in (10, 50, 100, 200):
        norms = []
        for _ in range(20):
            xt = torch.randn(length, d)
            J = torch.autograd.functional.jacobian(lambda z: self_attention(z, *W, causal=True)[-1], xt)
            norms.append(float(torch.linalg.matrix_norm(J[:, 0, :], ord=2)))
        attn[str(length)] = {"mean": float(np.mean(norms)), "std": float(np.std(norms)), "times_n": float(np.mean(norms) * length)}
    return {"rnn_jacobian_norm_by_distance": rnn, "attention_jacobian_norm_first_token": attn}


def _median_time(fn, repeats=7):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def e2_compute():
    rng = np.random.default_rng(0)
    rows = []
    for d in (64, 256):
        W, U = rng.standard_normal((d, d)) * 0.1, rng.standard_normal((d, d)) * 0.1
        Wq, Wk, Wv = (rng.standard_normal((d, d)) / d**0.5 for _ in range(3))
        for n in (32, 128, 512, 2048):
            x = rng.standard_normal((n, d))
            rows.append({
                "d": d,
                "n": n,
                "rnn_sequential_steps": n,
                "attention_sequential_steps": 1,
                "rnn_flops": anp.rnn_flops(n, d),
                "attention_flops": anp.attention_flops(n, d),
                "attention_score_matrix_floats": n * n,
                "rnn_ms": 1000 * _median_time(lambda: anp.rnn_forward(x, W, U)),
                "attention_ms": 1000 * _median_time(lambda: anp.self_attention(x, Wq, Wk, Wv, causal=True)),
            })
    return {"rows": rows, "note": "NumPy BLAS 스레드는 고정하지 않았다. 시간 열은 기기마다 다르다."}


def e3_scaling():
    out = {}
    for d in (16, 64, 256, 1024):
        for scale in (False, True):
            v, p, j = anp.softmax_saturation(d, scale=scale)
            out[f"d={d},scale={scale}"] = {"score_var": v, "softmax_peak": p, "softmax_jacobian_norm": j}
    return out


def make_batch(gen, batch, n):
    """첫 위치에 신호 토큰(0 또는 1), 나머지 n-1개는 잡음. 정답은 첫 토큰."""
    y = torch.randint(0, SIGNALS, (batch,), generator=gen)
    noise = torch.randint(SIGNALS, VOCAB, (batch, n - 1), generator=gen)
    return torch.cat([y[:, None], noise], dim=1), y


def train(model, n, seed, steps=600, batch=64, lr=3e-3):
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, max_grad, first_hit = [], 0.0, None
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = make_batch(gen, batch, n)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        g = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        max_grad = max(max_grad, g)
        opt.step()
        losses.append(loss.item())
        if first_hit is None and step >= 20 and np.mean(losses[-20:]) < 0.1:
            first_hit = step
    sec = time.perf_counter() - t0
    test_gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x, y = make_batch(test_gen, 1000, n)
        acc = float((model(x).argmax(-1) == y).float().mean())
        first_w = float(model.last_position_weights(x)[:, 0].mean()) if hasattr(model, "last_position_weights") else None
    return {
        "attention_weight_on_first_token": first_w,
        "test_acc": acc,
        "final_loss": float(np.mean(losses[-20:])),
        "steps_to_loss_below_0.1": first_hit,
        "max_grad_norm": max_grad,
        "nan_loss": bool(np.isnan(losses).any()),
        "ms_per_step": 1000 * sec / steps,
    }


def build(kind, d, seed, scale=True):
    torch.manual_seed(seed)
    if kind == "attention":
        return AttentionClassifier(VOCAB, d, SIGNALS, scale=scale)
    return RNNClassifier(VOCAB, d, SIGNALS, cell=kind)


def e4_training():
    out = {}
    for n in (10, 50, 200):
        for kind in ("rnn", "lstm", "attention"):
            out[f"n={n},{kind}"] = [train(build(kind, 32, s), n, s) for s in range(3)]
    return out


def e4b_lstm_longer():
    """공정성 점검: LSTM이 600스텝 안에 못 푼 n=50을 5배 더 학습시킨다."""
    def lstm_forget_bias(seed):
        m = build("lstm", 32, seed)
        d = 32
        with torch.no_grad():  # 게이트 순서는 (input, forget, cell, output). forget 편향만 1로
            m.rnn.bias_ih_l0[d : 2 * d].fill_(1.0)
            m.rnn.bias_hh_l0[d : 2 * d].fill_(0.0)
        return m

    return {
        "n=50,lstm,steps=3000": [train(build("lstm", 32, s), 50, s, steps=3000) for s in range(3)],
        "n=50,lstm_forget_bias1,steps=3000": [train(lstm_forget_bias(s), 50, s, steps=3000) for s in range(3)],
    }


def e4c_boundary():
    """RNN과 LSTM이 어느 길이까지 버티는지(1500스텝, 시드 3개)."""
    return {
        f"n={n},{kind}": [train(build(kind, 32, s), n, s, steps=1500) for s in range(3)]
        for n in (20, 30)
        for kind in ("rnn", "lstm")
    }


def e5_scale():
    out = {}
    for d in (16, 512):
        for scale in (True, False):
            out[f"d={d},scale={scale}"] = [
                train(build("attention", d, s, scale=scale), 50, s, steps=300, lr=1e-3) for s in range(3)
            ]
    return out


def main():
    torch.set_num_threads(1)  # 실측 시간을 비교 가능하게 단일 스레드로 고정
    results = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": 1,
        },
        "E1": e1_distance(),
        "E2": e2_compute(),
        "E3": e3_scaling(),
        "E4": e4_training(),
        "E4b": e4b_lstm_longer(),
        "E4c": e4c_boundary(),
        "E5": e5_scale(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
