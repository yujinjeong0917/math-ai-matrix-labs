"""5장 비교 실험. `uv run python llm/ch05_rnn_lstm/experiments.py` 로 실행한다 (단일 스레드 CPU에서 약 5분).

E1 거리와 기울기: RNN의 ||dh_t/dh_0||, LSTM의 셀 경로 곱 prod(f_t)와 전체 상태 야코비안 (forget 편향별)
E2 첫 토큰 맞히기, 7장 기준 설정(600스텝, n=10/50/200): 7장 2절 표를 이 장에서 다시 만든다
E2b 첫 토큰 맞히기, 경계 찾기(1500스텝, n=5/10/20/50): RNN / LSTM / forget 편향 3인 LSTM
E2c forget 편향 3인 LSTM이 버티는 길이(n=100 1500스텝, n=200 600스텝)
E3 학습 전 위치별 기울기: 손실이 첫 토큰과 마지막 토큰의 입력에 주는 기울기 크기
E4 클리핑: 큰 학습률 SGD에서 없음 / 원소별 / 노름 클리핑, 문턱값별, 시드 10개
E5 계산 순서: 같은 토큰 1,024개를 (시퀀스 수 B, 길이 n)으로 나눠 처리한 실측 시간, 스텝당 FLOP
손 계산 예: f^n (게이트를 f만큼 열어 두고 n걸음 갔을 때 남는 비율)

결과는 results/ch05.json 에 저장한다. 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import rnn_np as R
from rnn_torch import RNNClassifier

OUT = Path(__file__).parent / "results" / "ch05.json"
# 과제: 길이 n 시퀀스의 첫 토큰(신호 0 또는 1)을 마지막에 맞히기. 나머지는 잡음 토큰. 정답 확률 1/2(찍기).
SIGNALS, NOISE = 2, 16  # 토큰 0, 1은 신호(첫 위치에만 등장), 2..17은 잡음 (7장과 같은 과제)
VOCAB = SIGNALS + NOISE
DISTANCES = (1, 5, 10, 20, 50, 100, 200)


def hand_example():
    """첫 화면의 손 계산: 남는 비율 = f^n."""
    out = {}
    for f in (0.5, 0.8, 0.95):
        out[str(f)] = {str(n): f**n for n in (1, 2, 3, 10, 50, 200)}
    out["sigmoid(0,1,3)"] = [float(R.sigmoid(z)) for z in (0.0, 1.0, 3.0)]
    return out


def e1_distance():
    """거리 t에 따른 기울기 노름. RNN은 순환 행렬의 스펙트럼 반경 rho별, LSTM은 forget 편향별로 잰다."""
    # RNN: 7장 experiments.py e1_distance와 같은 난수 순서 (같은 숫자가 나온다)
    rng = np.random.default_rng(0)
    d, n = 32, 200
    x = rng.standard_normal((n, d))
    U = rng.standard_normal((d, d)) * 0.1
    base = rng.standard_normal((d, d))
    # 고유값 절댓값의 최댓값(스펙트럼 반경)으로 나눠 반경 1로 만든 뒤 rho를 곱한다. rho < 1 소실, > 1 폭주 영역.
    base /= np.max(np.abs(np.linalg.eigvals(base)))  # 스펙트럼 반경 1로 정규화
    rnn = {}
    # rho 0.8(소실), 1.0(경계), 1.5(폭주) 세 경우.
    for rho in (0.8, 1.0, 1.5):
        norms = R.rnn_state_jacobian_norms(x, base * rho, U)
        rnn[str(rho)] = {str(t): float(norms[t - 1]) for t in DISTANCES}

    # LSTM: PyTorch 기본 초기화와 같은 균등분포 U(-1/sqrt(H), 1/sqrt(H)), 같은 입력 x
    H = 32
    r = np.random.default_rng(1)
    # PyTorch 기본 LSTM 초기화와 같은 범위 U(-1/sqrt(H), 1/sqrt(H)).
    k = 1 / np.sqrt(H)
    Wx, Wh, b0 = r.uniform(-k, k, (4 * H, d)), r.uniform(-k, k, (4 * H, H)), r.uniform(-k, k, 4 * H)
    lstm = {}
    for name, fb in (("forget_bias=0", 0.0), ("forget_bias=1", 1.0), ("forget_bias=3", 3.0), ("no_forget_gate(f=1)", None)):
        b = b0.copy()
        if fb is None:
            # 게이트 행 순서가 (i, f, o, g)이므로 H:2H가 forget 게이트의 편향.
            b[H : 2 * H] = 40.0  # sigmoid(40) = 1: 1997년 원형처럼 셀이 자기 자신을 1로 잇는다
        else:
            b[H : 2 * H] += fb
        # prod(f)를 시점마다 기록하려고 길이별로 다시 돈다(비용이 작다)
        cell, state, _ = R.lstm_state_jacobian_norms(x, Wx, Wh, b)
        _, cs, caches = R.lstm_forward(x, Wx, Wh, b)
        # cache = (x, h_prev, c_prev, i, f, o, g, c)에서 인덱스 4가 f. 모양 (n, H).
        fs = np.array([cache[4] for cache in caches])  # (n, H) forget 게이트 값
        # 시점별 forget 게이트의 누적곱: 셀 경로만 따라 흐를 때 남는 신호 비율.
        fprod = np.cumprod(fs, axis=0)
        lstm[name] = {
            "mean_f": float(fs.mean()),
            "prod_f_mean_over_units": {str(t): float(fprod[t - 1].mean()) for t in DISTANCES},
            "cell_jacobian_norm": {str(t): float(cell[t - 1]) for t in DISTANCES},
            "state_jacobian_norm": {str(t): float(state[t - 1]) for t in DISTANCES},
            "mean_abs_cell_at_200": float(np.abs(cs[-1]).mean()),
        }
    return {"rnn_jacobian_norm_by_distance": rnn, "lstm_by_forget_bias": lstm}


def make_batch(gen, batch, n):
    """첫 위치에 신호 토큰(0 또는 1), 나머지 n-1개는 잡음. 정답은 첫 토큰. (7장과 같다)"""
    # 정답 = 첫 토큰. generator로 난수를 고정해 시드가 같으면 같은 배치가 나온다.
    y = torch.randint(0, SIGNALS, (batch,), generator=gen)
    noise = torch.randint(SIGNALS, VOCAB, (batch, n - 1), generator=gen)
    return torch.cat([y[:, None], noise], dim=1), y


def train(model, n, seed, steps=600, batch=64, lr=3e-3):
    """7장 experiments.py의 train과 같은 학습 루프(Adam, 배치 64)."""
    # 학습 데이터 난수. 평가(999)와 겹치지 않게 시드를 다르게 둔다.
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, max_grad, first_hit = [], 0.0, None
    t0 = time.perf_counter()
    for step in range(steps):
        x, y = make_batch(gen, batch, n)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        # 클리핑 전 전체 기울기 노름(모든 파라미터).
        g = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        max_grad = max(max_grad, g)
        opt.step()
        losses.append(loss.item())
        # 최근 20스텝 평균 손실이 0.1 아래로 처음 내려간 스텝(수렴 속도 지표). 20스텝 이전엔 평균을 내지 않는다.
        if first_hit is None and step >= 20 and np.mean(losses[-20:]) < 0.1:
            first_hit = step
    sec = time.perf_counter() - t0
    test_gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x, y = make_batch(test_gen, 1000, n)
        acc = float((model(x).argmax(-1) == y).float().mean())
    return {
        "test_acc": acc,
        "final_loss": float(np.mean(losses[-20:])),
        "steps_to_loss_below_0.1": first_hit,
        "max_grad_norm": max_grad,
        "nan_loss": bool(np.isnan(losses).any()),
        # 시간은 기기 의존.
        "ms_per_step": 1000 * sec / steps,
    }


def build(kind, seed, d=32):
    """모델 종류별 생성. 시드 고정 뒤 만들어 초기 가중치가 재현된다. lstm_fb3은 forget 편향 3."""
    torch.manual_seed(seed)
    if kind == "lstm_fb3":
        return RNNClassifier(VOCAB, d, SIGNALS, cell="lstm", forget_bias=3.0)
    return RNNClassifier(VOCAB, d, SIGNALS, cell=kind)


def e2_ch07_baseline():
    """7장 기준 설정(600스텝)으로 n = 10, 50, 200에서 RNN과 LSTM을 시드 3개씩 학습."""
    return {f"n={n},{kind}": [train(build(kind, s), n, s) for s in range(3)] for n in (10, 50, 200) for kind in ("rnn", "lstm")}


def e2b_boundary():
    """학습을 1500스텝으로 늘려 성공과 실패의 경계가 되는 길이를 찾는다."""
    return {
        f"n={n},{kind}": [train(build(kind, s), n, s, steps=1500) for s in range(3)]
        for n in (5, 10, 20, 50)
        for kind in ("rnn", "lstm", "lstm_fb3")
    }


def e2c_open_gate_limit():
    """forget 편향 3 LSTM이 n = 100, 200에서도 버티는지."""
    return {
        "n=100,lstm_fb3,steps=1500": [train(build("lstm_fb3", s), 100, s, steps=1500) for s in range(3)],
        "n=200,lstm_fb3,steps=600": [train(build("lstm_fb3", s), 200, s, steps=600) for s in range(3)],
    }


def e3_position_gradients(n=50):
    """학습 전, 손실이 각 위치의 입력(임베딩)에 주는 기울기 크기. 시드 3개 × 배치 256의 평균."""
    out = {}
    for kind in ("rnn", "lstm", "lstm_fb3"):
        first, last = [], []
        for s in range(3):
            m = build(kind, s)
            gen = torch.Generator().manual_seed(500 + s)
            x, y = make_batch(gen, 256, n)
            logits, e = m(x, return_inputs=True)
            torch.nn.functional.cross_entropy(logits, y).backward()
            # (batch, n, d) -> (batch, n): 시점별 입력 임베딩 기울기의 L2 노름. 학습 전 상태에서 잰다.
            g = e.grad.norm(dim=-1)  # (batch, n)
            first.append(float(g[:, 0].mean()))
            last.append(float(g[:, -1].mean()))
        out[kind] = {"first_token": float(np.mean(first)), "last_token": float(np.mean(last)), "ratio_first_over_last": float(np.mean(first) / np.mean(last))}
    return out


def clip_run(seed, lr, method, c, n=10, steps=800, rho=1.0):
    """SGD(모멘텀 없음)로 RNN을 학습하며 클리핑 방식을 바꾼다. 순환 행렬은 스펙트럼 반경 rho로 맞춘다."""
    m = build("rnn", seed)
    with torch.no_grad():
        W = m.rnn.weight_hh_l0
        # 순환 가중치 W의 스펙트럼 반경을 rho로 맞춘다(E1과 같은 방식).
        W.mul_(rho / torch.linalg.eigvals(W).abs().max())
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    norms, losses, nan = [], [], False
    for _ in range(steps):
        x, y = make_batch(gen, 64, n)
        loss = torch.nn.functional.cross_entropy(m(x), y)
        if not torch.isfinite(loss):
            nan = True
            break
        opt.zero_grad()
        loss.backward()
        # max_norm = inf이면 아무것도 자르지 않고 전체 노름만 돌려준다. 자르기 전 노름 기록용.
        norms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf"))))  # 자르기 전 노름 기록
        if method == "norm":
            torch.nn.utils.clip_grad_norm_(m.parameters(), c)
        elif method == "elementwise":
            # 원소별 클리핑: 각 원소를 [-c, c]로 자른다(방향이 바뀔 수 있다).
            torch.nn.utils.clip_grad_value_(m.parameters(), c)
        opt.step()
        losses.append(loss.item())
    test_gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x, y = make_batch(test_gen, 1000, n)
        acc = float((m(x).argmax(-1) == y).float().mean())
    med = float(np.median(norms))
    # spike_steps: 노름이 중앙값의 10배를 넘은 스텝 수. 폭주 스파이크의 지표(이 실험이 정한 기준).
    return {"test_acc": acc, "max_grad_norm": max(norms), "median_grad_norm": med,
            "spike_steps(>10x_median)": int(sum(v > 10 * med for v in norms)), "final_loss": float(np.mean(losses[-50:])), "nan": nan}


def e4_clipping():
    """큰 학습률 SGD에서 클리핑 없음 / 원소별 / 노름 방식을 문턱값별로 비교한다(시드 10개)."""
    configs = [("none", None), ("elementwise", 1.0), ("elementwise", 0.1), ("elementwise", 0.03), ("norm", 1.0), ("norm", 0.3), ("norm", 0.1)]
    out = {}
    for lr in (1.0, 2.0, 3.0):
        for method, c in configs:
            runs = [clip_run(s, lr, method, c) for s in range(10)]
            out[f"lr={lr},{method},c={c}"] = {
                # 정확도 0.95 이상이면 성공으로 센다(실험의 기준).
                "success(acc>=0.95)": int(sum(r["test_acc"] >= 0.95 for r in runs)),
                "seeds": 10,
                "max_grad_norm": float(max(r["max_grad_norm"] for r in runs)),
                "mean_final_loss": float(np.mean([r["final_loss"] for r in runs])),
                "nan_runs": int(sum(r["nan"] for r in runs)),
                "runs": runs,
            }
    return out


def _median_time(fn, repeats=7):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def e5_compute():
    """같은 토큰 수 1024개를 (B, n)으로 다르게 나눠 시점 직렬 단계 수가 시간에 미치는 영향을 잰다."""
    rng = np.random.default_rng(0)
    d = H = 64
    W, U = rng.standard_normal((H, H)) * 0.1, rng.standard_normal((H, d)) * 0.1
    Wx, Wh, b = rng.standard_normal((4 * H, d)) * 0.1, rng.standard_normal((4 * H, H)) * 0.1, np.zeros(4 * H)
    rows = []
    for B, n in ((1, 1024), (8, 128), (64, 16), (1024, 1)):
        x = rng.standard_normal((B, n, d))
        rows.append({
            "B": B, "n": n, "tokens": B * n, "sequential_steps": n,
            "rnn_ms": 1000 * _median_time(lambda: R.rnn_forward_batched(x, W, U)),
            "lstm_ms": 1000 * _median_time(lambda: R.lstm_forward_batched(x, Wx, Wh, b)),
        })
    return {
        "same_tokens_different_shape": rows,
        "flops_per_step(d=H=64)": {"rnn": R.rnn_flops_per_step(d, H), "lstm": R.lstm_flops_per_step(d, H)},
        "note": "NumPy BLAS 스레드는 고정하지 않았다. 시간 열은 기기마다 다르다.",
    }


def main():
    # 시간 측정과 재현성을 위해 단일 스레드.
    torch.set_num_threads(1)  # 7장과 같이 단일 스레드
    t0 = time.perf_counter()
    results = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "machine": platform.machine(), "torch_threads": 1},
        "hand_example": hand_example(),
        "E1": e1_distance(),
        "E2": e2_ch07_baseline(),
        "E2b": e2b_boundary(),
        "E2c": e2c_open_gate_limit(),
        "E3": e3_position_gradients(),
        "E4": e4_clipping(),
        "E5": e5_compute(),
    }
    results["env"]["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved {OUT} in {results['env']['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
