"""9장 비교 실험. `uv run python llm/ch09_residual_norm/experiments.py` 로 실행한다.

과제: 2차 규칙 말뭉치. 길이 16, 어휘 16. 앞의 두 토큰 (a, b)가 정해지면 다음 토큰은
80% 확률로 고정된 표 T[a, b], 20% 확률로 아무 토큰이다. 이 과제의 이론 최저 손실(엔트로피)을
같이 기록해서, 각 모델이 바닥에서 얼마나 떨어져 있는지 볼 수 있게 한다.

E1 초기화 직후 진단(L=16): 층별 활성값 표준편차, 층별 기울기 노름, 위치 간 평균 코사인 유사도
E1b 깊이별 위치 간 코사인(연습 2의 정답 기준)
E2 깊이 × 배치 방식: L in {1,2,4,8,16}, 5가지 방식, 시드 3개. 고정 학습 세트 전체의 최종 손실
E3 warmup × LN 위치(L=8): Post/Pre × warmup 유무 × 학습률 3개, 시드 3개. 발산 기준은 미리 정한다
E3b Xiong 등 정리 1 점검: 마지막 층 W2의 기대 기울기 Frobenius 노름, 깊이별
E4 BatchNorm이 배치에 기대는 모습: 배치 1, 배치 친구가 바뀔 때
E5 계산량: 잔차·LN이 블록 FLOP에서 차지하는 비율
E6 헤드 수(부록): L=2 Pre-LN에서 h=1 vs h=4

결과는 results/ch09.json 에 저장하고, 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import resnorm_np as rn
from resnorm_torch import TinyLM

OUT = Path(__file__).parent / "results" / "ch09.json"
V, N_CTX, D, H = 16, 16, 32, 4
P_RULE = 0.8
N_TRAIN = 1000
ARCHS = rn.ARCHS
DIVERGE_RULE = "손실에 NaN이 생기거나, 최종 학습 손실이 초기 손실보다 크면 발산으로 센다(실험 전에 정함)."


# ---------------------------------------------------------------- 데이터


def make_corpus(n_seq, seed):
    rng = np.random.default_rng(seed)
    T = np.random.default_rng(12345).integers(0, V, (V, V))  # 규칙 표는 모든 실험에서 같다
    X = np.empty((n_seq, N_CTX), dtype=np.int64)
    X[:, :2] = rng.integers(0, V, (n_seq, 2))
    for t in range(2, N_CTX):
        rule = T[X[:, t - 2], X[:, t - 1]]
        rand = rng.integers(0, V, n_seq)
        X[:, t] = np.where(rng.random(n_seq) < P_RULE, rule, rand)
    return torch.tensor(X)


def entropy_floor():
    """위치 0의 다음 토큰(위치 1)은 균등, 위치 1..14의 다음 토큰은 규칙(0.8) + 잡음(0.2)."""
    p_hit = P_RULE + (1 - P_RULE) / V
    p_miss = (1 - P_RULE) / V
    h_rule = -(p_hit * math.log(p_hit) + (V - 1) * p_miss * math.log(p_miss))
    n_pred = N_CTX - 1
    return {"uniform": math.log(V), "rule": h_rule, "floor": (math.log(V) + (n_pred - 1) * h_rule) / n_pred}


TRAIN = make_corpus(N_TRAIN, seed=0)


def lm_loss(model, X):
    logits = model(X[:, :-1])
    return F.cross_entropy(logits.reshape(-1, V), X[:, 1:].reshape(-1))


def build(arch, L, seed, h=H):
    torch.manual_seed(seed)
    return TinyLM(V, N_CTX, D, L, h, arch)


def full_train_loss(model):
    model.eval()
    with torch.no_grad():
        val = float(lm_loss(model, TRAIN))
    model.train()
    return val


def train(model, seed, steps=400, batch=32, lr=1e-3, warmup=0):
    gen = torch.Generator().manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    init_loss = full_train_loss(model)
    nan, max_grad, t0 = False, 0.0, time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup) if warmup else lr  # 선형 warmup 후 고정
        idx = torch.randint(0, N_TRAIN, (batch,), generator=gen)
        loss = lm_loss(model, TRAIN[idx])
        if not torch.isfinite(loss):
            nan = True
            break
        opt.zero_grad()
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)))
        max_grad = max(max_grad, gn)
        opt.step()
    final = full_train_loss(model)
    nan = nan or not math.isfinite(final)
    return {
        "init_loss": init_loss,
        "final_train_loss": final if math.isfinite(final) else None,
        "diverged": bool(nan or final > init_loss),
        "nan": nan,
        "max_grad_norm": max_grad,
        "ms_per_step": 1000 * (time.perf_counter() - t0) / steps,
    }


# ---------------------------------------------------------------- E1


def init_diagnostics(arch, L, seed):
    model = build(arch, L, seed)
    X = TRAIN[:64]
    logits, hid = model(X[:, :-1], return_hidden=True)
    for h_ in hid:
        h_.retain_grad()
    loss = F.cross_entropy(logits.reshape(-1, V), X[:, 1:].reshape(-1))
    loss.backward()
    act_std = [float(h_.detach().std()) for h_ in hid]
    grad_norm = [float(h_.grad.norm()) for h_ in hid]  # 손실의 층별 입력 기울기 노름
    last = hid[-1].detach()
    u = last / last.norm(dim=-1, keepdim=True)
    cos = u @ u.transpose(1, 2)  # (B, n, n): 위치끼리의 코사인
    n = cos.shape[-1]
    off = (cos.sum((-1, -2)) - n) / (n * (n - 1))
    return {"act_std": act_std, "grad_norm": grad_norm, "pos_cosine_last": float(off.mean()), "init_loss": loss.item()}


def e1():
    out = {}
    for arch in ARCHS:
        runs = [init_diagnostics(arch, 16, s) for s in range(3)]
        out[arch] = {
            "act_std": np.mean([r["act_std"] for r in runs], 0).tolist(),
            "grad_norm": np.mean([r["grad_norm"] for r in runs], 0).tolist(),
            "pos_cosine_last": float(np.mean([r["pos_cosine_last"] for r in runs])),
            "init_loss": float(np.mean([r["init_loss"] for r in runs])),
        }
    # 첫 층 위치 코사인(비교 기준)
    m = build("plain", 1, 0)
    with torch.no_grad():
        _, hid = m(TRAIN[:64, :-1], return_hidden=True)
        u = hid[0] / hid[0].norm(dim=-1, keepdim=True)
        c = u @ u.transpose(1, 2)
        n = c.shape[-1]
        out["embedding_pos_cosine"] = float(((c.sum((-1, -2)) - n) / (n * (n - 1))).mean())
    return out


def e1b():
    """연습 2의 정답 기준: 깊이별 마지막 층 위치 간 코사인(초기화 직후, 시드 0)."""
    return {arch: {str(L): init_diagnostics(arch, L, 0)["pos_cosine_last"] for L in (1, 2, 4, 8, 16)}
            for arch in ("plain", "pre")}


# ---------------------------------------------------------------- E2


def e2():
    out = {}
    for arch in ARCHS:
        for L in (1, 2, 4, 8, 16):
            out[f"{arch},L={L}"] = [train(build(arch, L, s), s) for s in range(3)]
    return out


# ---------------------------------------------------------------- E3


def e3():
    out = {}
    for arch in ("post", "pre"):
        for lr in (1e-3, 3e-3, 1e-2):
            for warmup in (0, 100):
                out[f"{arch},lr={lr},warmup={warmup}"] = [
                    train(build(arch, 8, s), s, lr=lr, warmup=warmup) for s in range(3)
                ]
    return out


def expected_grad_w2(arch, L, seed, batches=20):
    """Xiong 등 Figure 3의 통계: 미니배치마다 기울기를 구해 평균 낸 뒤 Frobenius 노름."""
    model = build(arch, L, seed)
    gen = torch.Generator().manual_seed(77 + seed)
    acc = torch.zeros_like(model.blocks[-1].mlp.W2)
    for _ in range(batches):
        idx = torch.randint(0, N_TRAIN, (64,), generator=gen)
        model.zero_grad()
        lm_loss(model, TRAIN[idx]).backward()
        acc += model.blocks[-1].mlp.W2.grad
    return float((acc / batches).norm())


def e3b():
    return {
        f"{arch},L={L}": float(np.mean([expected_grad_w2(arch, L, s) for s in range(3)]))
        for arch in ("post", "pre")
        for L in (2, 4, 8, 16)
    }


# ---------------------------------------------------------------- E4


def e4():
    rng = np.random.default_rng(0)
    d = 4
    g, b = np.ones(d), np.full(d, 0.5)
    sample = rng.standard_normal((1, d)) * 3
    bn_single = rn.batch_norm(sample, g, b)
    torch_err = None
    try:
        torch.nn.BatchNorm1d(d).train()(torch.tensor(sample, dtype=torch.float32))
    except ValueError as e:
        torch_err = str(e)
    mates_a = np.vstack([sample, rng.standard_normal((7, d))])
    mates_b = np.vstack([sample, rng.standard_normal((7, d)) + 2.0])
    bn_a, bn_b = rn.batch_norm(mates_a, g, b)[0], rn.batch_norm(mates_b, g, b)[0]
    ln_a, ln_b = rn.layer_norm(mates_a, g, b)[0], rn.layer_norm(mates_b, g, b)[0]
    return {
        "batch1_bn_output": bn_single[0].tolist(),
        "batch1_bn_beta": b.tolist(),
        "torch_batchnorm1d_batch1_error": torch_err,
        "same_sample_bn_with_mates_a": bn_a.tolist(),
        "same_sample_bn_with_mates_b": bn_b.tolist(),
        "bn_max_abs_change": float(np.max(np.abs(bn_a - bn_b))),
        "ln_max_abs_change": float(np.max(np.abs(ln_a - ln_b))),
    }


# ---------------------------------------------------------------- E5


def e5():
    rows = []
    for n, d in ((16, 32), (1024, 768)):
        blk = rn.block_flops(n, d)
        res, ln = rn.residual_flops(n, d), rn.layer_norm_flops(n, d)
        rows.append({"n": n, "d": d, "block_flops": blk, "residual_flops": res, "ln_flops": ln,
                     "extra_ratio": (res + ln) / blk})
    return rows


# ---------------------------------------------------------------- E6


def e6():
    return {f"h={h}": [train(build("pre", 2, s, h=h), s) for s in range(3)] for h in (1, 4)}


# ---------------------------------------------------------------- 손 계산 예(첫 화면)


def hand_examples():
    x = np.array([1.0, 3.0, 5.0, 7.0])
    ln = rn.layer_norm(x, 1.0, 0.0, eps=0.0)
    return {
        "ln_input": x.tolist(),
        "ln_mean": float(x.mean()),
        "ln_var": float(x.var()),
        "ln_output": ln.round(4).tolist(),
        "plain_0.5_pow10": 0.5**10,
        "residual_1.5_pow10": 1.5**10,
    }


def main():
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    results = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": 1,
        },
        "setup": {"vocab": V, "n_ctx": N_CTX, "d": D, "heads": H, "p_rule": P_RULE, "n_train_seq": N_TRAIN,
                  "steps": 400, "batch": 32, "optimizer": "Adam", "lr_default": 1e-3,
                  "diverge_rule": DIVERGE_RULE, "entropy": entropy_floor()},
        "hand": hand_examples(),
        "E1": e1(),
        "E1b": e1b(),
        "E2": e2(),
        "E3": e3(),
        "E3b": e3b(),
        "E4": e4(),
        "E5": e5(),
        "E6": e6(),
    }
    results["runtime_sec"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in ("setup", "hand", "E3b", "E4", "E5")}, indent=1, ensure_ascii=False))
    print("runtime_sec", results["runtime_sec"])


if __name__ == "__main__":
    main()
