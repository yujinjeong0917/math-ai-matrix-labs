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

입력: 없음(시드는 파일 안에 고정). 출력: results/ch07.json. 단위: 노름은 무차원, 시간 ms, 손실 nat.
과제(E4~E5): 길이 n 토큰열의 첫 토큰(신호 0 또는 1)을 마지막 위치에서 맞힌다. 나머지는 잡음 토큰.
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
# 토큰 id: 0, 1 = 신호, 2..17 = 잡음 16종. 정답은 첫 위치의 신호
SIGNALS, NOISE = 2, 16  # 토큰 0, 1은 신호(첫 위치에만 등장), 2..17은 잡음
VOCAB = SIGNALS + NOISE


def e1_distance():
    rng = np.random.default_rng(0)
    d, n = 32, 200
    x = rng.standard_normal((n, d))
    U = rng.standard_normal((d, d)) * 0.1
    base = rng.standard_normal((d, d))
    # 고유값 절댓값의 최댓값으로 나눠 스펙트럼 반경을 1로 만든다. 아래 rho 배수로 반경을 0.8, 1.0, 1.5로 조절
    base /= np.max(np.abs(np.linalg.eigvals(base)))  # 스펙트럼 반경 1로 정규화
    rnn = {}
    for rho in (0.8, 1.0, 1.5):
        # U를 0.1배로 작게 두어 입력 항이 상태 전달을 압도하지 않게 한다(모형의 선택)
        norms = anp.rnn_state_jacobian_norms(x, base * rho, U)
        # 거리 t = 1, 10, 50, 100, 200에서의 ||dh_t/dh_0||. norms[t-1]이 거리 t 값
        rnn[str(rho)] = {str(t): float(norms[t - 1]) for t in (1, 10, 50, 100, 200)}

    from attention_torch import self_attention

    torch.manual_seed(0)
    # 가중치는 1/sqrt(d)로 고정하고 입력 xt만 20번 새로 뽑는다
    W = [torch.randn(d, d) / d**0.5 for _ in range(3)]  # 가중치는 고정하고 입력만 20번 새로 뽑는다
    attn = {}
    for length in (10, 50, 100, 200):
        norms = []
        for _ in range(20):
            xt = torch.randn(length, d)
            # y_n = self_attention(z)[-1]: (d,)을 입력 z (length, d)로 미분 -> J: (d, length, d)
            J = torch.autograd.functional.jacobian(lambda z: self_attention(z, *W, causal=True)[-1], xt)
            # J[:, 0, :]: 출력 y_n이 첫 위치 입력 x_1에 기대는 (d, d) 블록의 스펙트럼 노름
            norms.append(float(torch.linalg.matrix_norm(J[:, 0, :], ord=2)))
        # times_n = 노름 * 길이: 길이에 반비례하게 줄어드는지(약 1/n)를 보려는 지표
        attn[str(length)] = {"mean": float(np.mean(norms)), "std": float(np.std(norms)), "times_n": float(np.mean(norms) * length)}
    return {"rnn_jacobian_norm_by_distance": rnn, "attention_jacobian_norm_first_token": attn}


def _median_time(fn, repeats=7):
    times = []
    for _ in range(repeats):
        # 성능 카운터로 호출 한 번의 시간(초)을 잰다. 첫 호출 효과를 줄이려고 중앙값을 쓴다
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def e2_compute():
    rng = np.random.default_rng(0)
    rows = []
    for d in (64, 256):
        W, U = rng.standard_normal((d, d)) * 0.1, rng.standard_normal((d, d)) * 0.1
        # 가중치는 1/sqrt(d)로 줄여 점수 크기를 안정적으로 둔다
        Wq, Wk, Wv = (rng.standard_normal((d, d)) / d**0.5 for _ in range(3))
        for n in (32, 128, 512, 2048):
            x = rng.standard_normal((n, d))
            rows.append({
                "d": d,
                "n": n,
                # 이론: RNN은 시점 수 n만큼 순차 단계가 필요하고 attention은 행렬곱 한 번(병렬)이다
                "rnn_sequential_steps": n,
                "attention_sequential_steps": 1,
                "rnn_flops": anp.rnn_flops(n, d),
                "attention_flops": anp.attention_flops(n, d),
                # 점수 행렬 (n, n)의 원소 수: 메모리가 n^2로 늘어난다
                "attention_score_matrix_floats": n * n,
                "rnn_ms": 1000 * _median_time(lambda: anp.rnn_forward(x, W, U)),
                "attention_ms": 1000 * _median_time(lambda: anp.self_attention(x, Wq, Wk, Wv, causal=True)),
            })
    # BLAS(행렬 라이브러리) 스레드 수를 고정하지 않아 시간은 기기마다 다르다
    return {"rows": rows, "note": "NumPy BLAS 스레드는 고정하지 않았다. 시간 열은 기기마다 다르다."}


def e3_scaling():
    out = {}
    for d in (16, 64, 256, 1024):
        for scale in (False, True):
            # 점수 분산, softmax 최대 가중치, 야코비안 노름(스케일 유무별, d별)
            v, p, j = anp.softmax_saturation(d, scale=scale)
            out[f"d={d},scale={scale}"] = {"score_var": v, "softmax_peak": p, "softmax_jacobian_norm": j}
    return out


def make_batch(gen, batch, n):
    """첫 위치에 신호 토큰(0 또는 1), 나머지 n-1개는 잡음. 정답은 첫 토큰."""
    # 정답 신호 (batch,). 생성기를 주어 시드를 재현 가능하게
    y = torch.randint(0, SIGNALS, (batch,), generator=gen)
    # 잡음 (batch, n-1), 토큰 id 2..17
    noise = torch.randint(SIGNALS, VOCAB, (batch, n - 1), generator=gen)
    # 입력 (batch, n): 첫 칸이 신호, 나머지가 잡음. 정답 y (batch,)
    return torch.cat([y[:, None], noise], dim=1), y


def train(model, n, seed, steps=600, batch=64, lr=3e-3):
    # 데이터 시드는 모델 시드와 분리(1000 + seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, max_grad, first_hit = [], 0.0, None
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = make_batch(gen, batch, n)
        # model(x): (batch, 2) 로짓. 마지막 위치 상태만으로 신호를 분류
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        # 전체 기울기 L2 노름. 학습 중 최댓값을 기록해 폭주를 본다
        g = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        max_grad = max(max_grad, g)
        opt.step()
        losses.append(loss.item())
        # 최근 20스텝 평균 손실이 0.1 미만이 되는 첫 스텝(20스텝 이후부터 판정)
        if first_hit is None and step >= 20 and np.mean(losses[-20:]) < 0.1:
            first_hit = step
    sec = time.perf_counter() - t0
    # 평가 데이터는 시드 999로 고정(학습과 분리)
    test_gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x, y = make_batch(test_gen, 1000, n)
        acc = float((model(x).argmax(-1) == y).float().mean())
        # 어텐션 모델만: 마지막 위치가 첫 토큰에 준 가중치 평균. 1에 가까우면 첫 토큰을 정확히 찾은 것
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
    # 가중치 초기화 시드
    torch.manual_seed(seed)
    if kind == "attention":
        return AttentionClassifier(VOCAB, d, SIGNALS, scale=scale)
    return RNNClassifier(VOCAB, d, SIGNALS, cell=kind)


def e4_training():
    out = {}
    for n in (10, 50, 200):
        for kind in ("rnn", "lstm", "attention"):
            # 길이 n * 모델 3종 * 시드 3개. 모델 차원 d = 32
            out[f"n={n},{kind}"] = [train(build(kind, 32, s), n, s) for s in range(3)]
    return out


def e4b_lstm_longer():
    """공정성 점검: LSTM이 600스텝 안에 못 푼 n=50을 5배 더 학습시킨다."""
    def lstm_forget_bias(seed):
        m = build("lstm", 32, seed)
        d = 32
        with torch.no_grad():  # 게이트 순서는 (input, forget, cell, output). forget 편향만 1로
            # PyTorch LSTM의 편향 순서는 (input, forget, cell, output)로 각 d칸. 두 번째 d칸이 forget 게이트(PyTorch LSTM 규칙, 위 원래 주석과 같음)
            m.rnn.bias_ih_l0[d : 2 * d].fill_(1.0)
            m.rnn.bias_hh_l0[d : 2 * d].fill_(0.0)
        return m

    return {
        "n=50,lstm,steps=3000": [train(build("lstm", 32, s), 50, s, steps=3000) for s in range(3)],
        # forget 편향 1 초기화: 초기에 기억을 더 잘 유지하게 한다는 관행적 기법
        "n=50,lstm_forget_bias1,steps=3000": [train(lstm_forget_bias(s), 50, s, steps=3000) for s in range(3)],
    }


def e4c_boundary():
    """RNN과 LSTM이 어느 길이까지 버티는지(1500스텝, 시드 3개)."""
    return {
        # 1500스텝 * 시드 3개. 길이 20과 30에서 RNN과 LSTM이 버티는지
        f"n={n},{kind}": [train(build(kind, 32, s), n, s, steps=1500) for s in range(3)]
        for n in (20, 30)
        for kind in ("rnn", "lstm")
    }


def e5_scale():
    out = {}
    for d in (16, 512):
        for scale in (True, False):
            out[f"d={d},scale={scale}"] = [
                # d가 클수록 스케일 없는 점수 분산이 커져 softmax가 포화한다는 가설 점검. 짧게(300스텝) 낮은 학습률로
                train(build("attention", d, s, scale=scale), 50, s, steps=300, lr=1e-3) for s in range(3)
            ]
    return out


def main():
    # 실측 시간을 비교 가능하게 단일 스레드로 고정
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
