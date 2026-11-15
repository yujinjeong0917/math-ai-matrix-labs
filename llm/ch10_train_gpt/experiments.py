"""10장 비교 실험. `uv run python llm/ch10_train_gpt/experiments.py` 로 실행한다.
일부만 다시 돌릴 때는 `... experiments.py E1 E3` 처럼 이름을 주면 그 결과만 ch10.json에 덮어쓴다.

모델: 9장 Pre-LN 블록 4층, d=128, 헤드 4, 문맥 64토큰. 데이터: ch10_data.py의 문법 말뭉치
(학습 창 1024개 = 65,536토큰. 600스텝이면 약 9번 반복해 보게 되어 과적합이 시작된다.
검증 창 256개는 다른 시드로 만든다).

E0 데이터: 어휘 크기, 이론 바닥 손실(엔트로피), 문맥을 안 보는 모델(유니그램)의 손실
E1b 비교 기준 · 같은 학습률들(+3, 10)에서 모멘텀 없는 SGD
E1 실패 1 · 학습률 3배씩: Adam(감쇠·클리핑 없음), 학습률 9개 × 시드 3개, 200스텝
E2 실패 2 · Adam+L2 vs AdamW: 같은 λ ∈ {0.001, 0.01, 0.1, 1}에서 검증 손실과 가중치 노름, 시드 3개
E2b 메커니즘: 임베딩 행마다 줄어든 비율. L2에서는 기울기가 큰(자주 나오는) 토큰이 덜 줄어드는가
E3 실패 3 + 비교 측정: {Adam+L2, AdamW} × {클리핑 없음, c=1} × 학습률 3개 × 시드 3개
E4 warmup: E1에서 실패한 학습률에서 상수 학습률 vs 선형 warmup + 코사인 감쇠
E5 계산량: 스텝당 시간, 토큰/초, 옵티마이저 상태 메모리
E6 NumPy 2층 MLP 언어모델을 NumPy AdamW로 학습, 같은 초기값의 torch 학습과 손실 곡선 대조
E7 체크포인트 저장과 탐욕 생성(11장으로 넘기는 관찰)

판정 규칙은 실험 전에 RULES에 정해 두고 결과 JSON에도 함께 기록한다. 전체 실행은 단일 스레드 CPU에서 약 25분 걸린다.

입력: 명령행 인자로 실험 이름(E0, E1, ...)을 줄 수 있다. 없으면 전부 실행. 출력: results/ch10.json, results/ch10_ckpt.pt(E7).
손실 단위는 nat(자연로그), 기호: V = 어휘 크기, N_CTX = 문맥 길이, D = 모델 차원, L = 블록 수, H = 헤드 수.
"""

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ch10_data as data
import ch10_np as cnp
from ch10_torch import MLPLM, TinyGPT, load_checkpoint, make_optimizer, save_checkpoint

OUT = Path(__file__).parent / "results" / "ch10.json"
CKPT = Path(__file__).parent / "results" / "ch10_ckpt.pt"

N_CTX, D, L, H = 64, 128, 4, 4
# 배치 16개 창. 한 스텝에 쓰는 토큰은 16 * 64 = 1024개(E5의 tokens_per_step)
BATCH, N_TRAIN, N_VAL = 16, 1024, 256
STEPS = 600
SEEDS = (0, 1, 2)
WD_MAIN = 0.1
# 3배씩 늘리는 학습률 격자(1e-4 ~ 1). RULES['lr_grid_revision']에 이 범위를 넓힌 이유가 적혀 있다
SWEEP_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

# 학습·검증 말뭉치는 시드(0, 1)가 달라 서로 다른 창이다. FLOOR는 각 말뭉치의 엔트로피 바닥
TRAIN_NP, FLOOR_TRAIN = data.make_windows(N_TRAIN, N_CTX, seed=0)
VAL_NP, FLOOR_VAL = data.make_windows(N_VAL, N_CTX, seed=1)
TRAIN, VAL = torch.tensor(TRAIN_NP), torch.tensor(VAL_NP)


def unigram_loss():
    # 학습 창의 '예측 대상' 토큰(첫 토큰 제외) 빈도. 1e-12는 log(0) 방지용 아주 작은 값
    counts = np.bincount(TRAIN_NP[:, 1:].reshape(-1), minlength=data.V) + 1e-12
    p = counts / counts.sum()
    # 문맥을 전혀 안 보는 모델이 검증 토큰에 주는 평균 -log p. 이보다 나쁘면 '학습 실패'로 센다
    vy = VAL_NP[:, 1:].reshape(-1)
    return float(-np.log(p[vy]).mean())


UNIGRAM = unigram_loss()

RULES = {
    "failure": f"손실에 NaN·inf가 생기거나, 마지막 20스텝 평균 학습 손실이 문맥을 안 보는 유니그램 모델의 손실({UNIGRAM:.3f})보다 크면 '학습 실패'로 센다.",
    "spike": "50번째 스텝부터, 기울기 노름이 직전 50스텝 중앙값의 3배를 넘은 스텝을 '튐'으로 센다(클리핑 전 노름).",
    "lr_grid": "E1에서 검증 손실 평균이 가장 낮은 학습률 lr_best, 시드 3개 모두 성공한 가장 큰 학습률 lr*, 그 3배인 3·lr* 를 E3의 학습률로 쓴다.",
    "lr_grid_revision": "처음 규칙은 lr*/3, lr*, 3·lr* 였다. 학습률 1e-4~0.1만 훑은 첫 실행에서 실패가 하나도 없어 lr*=0.1이 되었고, 세 학습률이 모두 손실이 나쁜 높은 쪽에 몰렸다. 그래서 E3를 돌리기 전에 훑는 범위를 1.0까지 늘리고 규칙을 위처럼 바꿨다.",
    "decay_params": "감쇠는 2차원 이상 텐서(임베딩·행렬)에만 건다. LN의 γ·β와 편향은 제외.",
    "best_config": "E3에서 실패가 없는 설정 중 최종 검증 손실의 시드 평균이 가장 낮은 설정을 체크포인트로 저장한다.",
}


# ---------------------------------------------------------------- 학습 한 번


def lm_loss(model, X):
    logits = model(X[:, :-1])
    return F.cross_entropy(logits.reshape(-1, data.V), X[:, 1:].reshape(-1))


# 고정 데이터 전체(학습 1024창 또는 검증 256창)에 대한 손실. 드롭아웃이 없어 eval/train 차이는 없지만 습관적으로 전환한다
def eval_loss(model, X):
    model.eval()
    with torch.no_grad():
        v = float(lm_loss(model, X))
    model.train()
    return v


def lr_at(step, lr, schedule, steps, warmup=60):
    if schedule == "const":
        return lr
    # 선형 warmup: 첫 60스텝은 lr * (step+1)/60으로 올린다
    if step < warmup:  # 선형 warmup: 0에서 lr까지
        return lr * (step + 1) / warmup
    # 코사인 감쇠 진행도 0..1. lr에서 lr/10까지(0.1 + 0.9 * 코사인 절반)
    prog = (step - warmup) / max(1, steps - warmup)  # 코사인: lr에서 lr/10까지
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def decayed_norm(model):
    # 감쇠 대상(2차원 이상) 파라미터의 L2 노름
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters() if p.dim() >= 2)))


def count_spikes(norms):
    s = 0
    # RULES['spike']와 같은 기준: 직전 50스텝 중앙값의 3배 초과. 앞 50스텝은 비교할 기준이 없어 세지 않는다
    for t in range(50, len(norms)):
        if norms[t] > 3 * np.median(norms[t - 50 : t]):
            s += 1
    return s


_CACHE = {}


def train_gpt(opt_kind, lr, wd, clip, seed, steps=STEPS, schedule="const", keep=False):
    key = (opt_kind, lr, wd, clip, seed, steps, schedule)
    # 결정적 학습이라 같은 키면 결과가 같다는 가정(시드 고정, 단일 스레드). keep=True는 모델이 필요해 캐시를 쓰지 않는다
    if not keep and key in _CACHE:  # 같은 설정을 두 실험에서 쓰면 한 번만 학습한다(결정적이라 결과가 같다)
        return _CACHE[key]
    torch.manual_seed(seed)
    model = TinyGPT(data.V, N_CTX, D, L, H)
    opt = make_optimizer(opt_kind, model, lr, wd)
    # 배치 추출용 난수는 모델 초기화 시드와 분리(1000 + seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    init_loss = eval_loss(model, TRAIN)
    losses, norms, val_curve = [], [], {}
    nan = False
    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            # 스텝마다 학습률을 덮어쓴다(const이면 항상 lr)
            g["lr"] = lr_at(step, lr, schedule, steps)
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=gen)
        loss = lm_loss(model, TRAIN[idx])
        opt.zero_grad()
        loss.backward()
        # clip_grad_norm_은 자르기 전 노름을 반환한다. clip이 없으면 무한대 상한이라 자르지 않고 노름만 기록
        total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf")))
        opt.step()
        lv = loss.item()
        losses.append(lv)
        norms.append(total)
        # NaN 또는 inf가 나오면 즉시 중단하고 발산으로 기록
        if not math.isfinite(lv) or not math.isfinite(total):
            nan = True
            break
        if (step + 1) % 100 == 0:
            val_curve[step + 1] = eval_loss(model, VAL)
    sec = time.perf_counter() - t0
    last = float(np.mean(losses[-20:]))
    out = {
        "init_train_loss": init_loss,
        "final_batch_loss_last20": last,
        # RULES['failure']: NaN이거나 마지막 20스텝 평균 학습 손실이 유니그램 손실보다 크다
        "failed": bool(nan or not math.isfinite(last) or last > UNIGRAM),
        "nan": nan,
        "train_loss": eval_loss(model, TRAIN) if not nan else None,
        "val_loss": eval_loss(model, VAL) if not nan else None,
        "val_curve": val_curve,
        "decayed_weight_norm": decayed_norm(model) if not nan else None,
        # 노름은 자르기 전 값(RULES['spike'])
        "max_grad_norm": float(np.nanmax(norms)),
        "median_grad_norm": float(np.nanmedian(norms)),
        "spikes": count_spikes(norms),
        "ms_per_step": 1000 * sec / len(losses),
        "steps_run": len(losses),
    }
    _CACHE[key] = out
    if keep:
        return out, model, opt
    return out


def summarize(runs):
    # 실패한 실행은 평균에서 뺀다(fail_rate에 따로 기록). 모두 실패하면 None
    vals = [r["val_loss"] for r in runs if not r["failed"]]
    return {
        "fail_rate": f"{sum(r['failed'] for r in runs)}/{len(runs)}",
        "val_mean": float(np.mean(vals)) if vals else None,
        "val_by_seed": [r["val_loss"] for r in runs],
        "val_best_mean": float(np.mean([min(r["val_curve"].values()) for r in runs if not r["failed"]])) if vals else None,
        "train_mean": float(np.mean([r["train_loss"] for r in runs if not r["failed"]])) if vals else None,
        "weight_norm_mean": float(np.mean([r["decayed_weight_norm"] for r in runs if not r["failed"]])) if vals else None,
        "spikes_by_seed": [r["spikes"] for r in runs],
        "max_grad_by_seed": [r["max_grad_norm"] for r in runs],
        "final_batch_loss_by_seed": [r["final_batch_loss_last20"] for r in runs],
    }


# ---------------------------------------------------------------- 실험


def e0():
    # 학습 창의 예측 대상 토큰별 등장 횟수
    counts = np.bincount(TRAIN_NP[:, 1:].reshape(-1), minlength=data.V)
    return {
        "vocab": data.VOCAB,
        "V": data.V,
        "uniform_loss": math.log(data.V),
        "unigram_loss": UNIGRAM,
        "entropy_floor_train": FLOOR_TRAIN,
        "entropy_floor_val": FLOOR_VAL,
        "train_tokens": int(N_TRAIN * N_CTX),
        "token_counts_train": {w: int(c) for w, c in zip(data.VOCAB, counts)},
        "example_window": " ".join(data.VOCAB[i] for i in TRAIN_NP[0][:24]),
    }


def e1():
    out = {}
    for lr in SWEEP_LRS:
        runs = [train_gpt("adamw", lr, 0.0, None, s, steps=200) for s in SEEDS]
        out[f"{lr:g}"] = summarize(runs)
    # fail_rate가 '0/'로 시작 = 시드 3개 모두 성공한 학습률
    ok = [lr for lr in SWEEP_LRS if out[f"{lr:g}"]["fail_rate"].startswith("0/")]
    lr_star = None
    for lr in SWEEP_LRS:  # 작은 쪽부터 연속으로 성공한 마지막 값
        if lr in ok:
            lr_star = lr
        else:
            break
    lr_best = min(SWEEP_LRS, key=lambda lr: out[f"{lr:g}"]["val_mean"] if out[f"{lr:g}"]["val_mean"] is not None else 1e9)
    # grid = [검증 손실이 가장 좋은 lr, 연속 성공 마지막 lr*, 3 * lr*]. 주의: 가장 작은 lr부터 실패하면 lr_star가 None이라 lr_star * 3에서 오류가 난다
    return {"runs": out, "lr_best": lr_best, "lr_star": lr_star, "grid": [lr_best, lr_star, lr_star * 3]}


def e1b():
    """비교 기준: 같은 모델을 모멘텀 없는 SGD로. Adam 없이도 같은 학습률 범위에서 학습되는가."""
    out = {}
    # SGD는 Adam보다 큰 학습률이 필요할 수 있어 3, 10을 더 시험한다
    for lr in SWEEP_LRS + [3.0, 10.0]:
        out[f"{lr:g}"] = summarize([train_gpt("sgd", lr, 0.0, None, s, steps=200) for s in SEEDS])
    return {"runs": out}


def e2(lr):
    out = {"lr": lr, "baseline_wd0": summarize([train_gpt("adamw", lr, 0.0, None, s) for s in SEEDS])}
    for wd in (0.001, 0.01, 0.1, 1.0):
        for kind in ("adam_l2", "adamw"):
            out[f"{kind},wd={wd:g}"] = summarize([train_gpt(kind, lr, wd, None, s) for s in SEEDS])
    return out


def e2b(lr, wd=WD_MAIN):
    """같은 λ에서 임베딩 행이 처음보다 얼마나 줄었나. 자주 나온 토큰 vs 드문 토큰."""
    # 입력으로 들어가는 토큰(마지막 제외)의 빈도. 임베딩 행은 입력으로 쓰일 때 기울기를 받는다
    counts = np.bincount(TRAIN_NP[:, :-1].reshape(-1), minlength=data.V)  # 입력으로 들어간 횟수
    # 시드 0으로 같은 초기화를 다시 만들어 '처음 행 노름'을 얻는다(학습에 쓴 시드 0 모델과 같은 값)
    torch.manual_seed(0)
    init_rows = TinyGPT(data.V, N_CTX, D, L, H).tok.weight.detach().norm(dim=1).numpy()
    res = {}
    for kind in ("adam_l2", "adamw"):
        _, model, opt = train_gpt(kind, lr, wd, None, 0, keep=True)
        rows = model.tok.weight.detach().norm(dim=1).numpy()
        # Adam이 기록한 임베딩의 상태: exp_avg(m), exp_avg_sq(v), step
        st = opt.state[model.tok.weight]
        t = st["step"].item()
        # 식: sqrt(v_hat), v_hat = v / (1 - beta2^t). 임베딩 행(토큰)마다 d개 성분의 평균 -> (V,)
        vhat = (st["exp_avg_sq"] / (1 - 0.999**t)).sqrt().mean(1).numpy()
        res[kind] = {"row_norm_ratio": (rows / init_rows).tolist(), "sqrt_vhat_row_mean": vhat.tolist()}
    # 빈도 내림차순 정렬. 상위 5개 = 자주 나온 토큰, 하위 5개 = 드문 토큰
    order = np.argsort(-counts)
    top, bottom = order[:5], order[-5:]
    pick = lambda k, idx: float(np.mean(np.array(res[k]["row_norm_ratio"])[idx]))  # noqa: E731
    l2_ratio, w_ratio = np.array(res["adam_l2"]["row_norm_ratio"]), np.array(res["adamw"]["row_norm_ratio"])
    return {
        "lr": lr,
        "wd": wd,
        "input_counts": {w: int(c) for w, c in zip(data.VOCAB, counts)},
        "frequent5": [data.VOCAB[i] for i in top],
        "rare5": [data.VOCAB[i] for i in bottom],
        "ratio_frequent5": {"adam_l2": pick("adam_l2", top), "adamw": pick("adamw", top)},
        "ratio_rare5": {"adam_l2": pick("adam_l2", bottom), "adamw": pick("adamw", bottom)},
        "corr_logcount_vs_ratio": {
            "adam_l2": float(np.corrcoef(np.log(counts + 1), l2_ratio)[0, 1]),
            "adamw": float(np.corrcoef(np.log(counts + 1), w_ratio)[0, 1]),
        },
        # L2에서 한 스텝에 실제로 줄어드는 비율 ≈ lr·λ/√v̂ (AdamW는 모든 원소가 lr·λ)
        # v_hat이 큰(기울기가 큰) 토큰일수록 L2 감쇠가 정규화되어 덜 줄어든다는 가설을 검증하는 지표들
        "sqrt_vhat_frequent5": {k: float(np.mean(np.array(res[k]["sqrt_vhat_row_mean"])[top])) for k in res},
        "sqrt_vhat_rare5": {k: float(np.mean(np.array(res[k]["sqrt_vhat_row_mean"])[bottom])) for k in res},
        "per_step_shrink_l2_frequent5": float(np.mean(lr * wd / np.array(res["adam_l2"]["sqrt_vhat_row_mean"])[top])),
        "per_step_shrink_l2_rare5": float(np.mean(lr * wd / np.array(res["adam_l2"]["sqrt_vhat_row_mean"])[bottom])),
        "per_step_shrink_adamw": lr * wd,
        "per_token": res,
    }


def e3(grid):
    out, raw = {}, {}
    for lr in grid:
        for kind in ("adam_l2", "adamw"):
            for clip in (None, 1.0):
                # 키 문자열을 _best()가 다시 split(',')로 해석한다. 이 형식을 바꾸면 _best가 깨진다(함정)
                key = f"{kind},clip={clip},lr={lr:g}"
                runs = [train_gpt(kind, lr, WD_MAIN, clip, s) for s in SEEDS]
                raw[key] = runs
                out[key] = summarize(runs)
    return out, raw


def e4(lr):
    return {
        "lr": lr,
        "const": summarize([train_gpt("adamw", lr, WD_MAIN, None, s) for s in SEEDS]),
        "warmup60_cosine": summarize([train_gpt("adamw", lr, WD_MAIN, None, s, schedule="warmup_cosine") for s in SEEDS]),
    }


def e5(raw):
    # 실패하지 않은 모든 실행의 스텝당 시간 중앙값을 쓴다(기기마다 다름)
    ms = [r["ms_per_step"] for runs in raw.values() for r in runs if not r["failed"]]
    torch.manual_seed(0)
    model = TinyGPT(data.V, N_CTX, D, L, H)
    opt = make_optimizer("adamw", model, 1e-3, WD_MAIN)
    # 옵티마이저 상태(m, v)가 생기도록 한 스텝만 돌린다
    lm_loss(model, TRAIN[:BATCH]).backward()
    opt.step()
    n_params = sum(p.numel() for p in model.parameters())
    # Adam은 파라미터마다 m, v 두 개를 저장해 상태 크기가 파라미터의 약 2배
    state_floats = sum(v.numel() for s in opt.state.values() for k, v in s.items() if k in ("exp_avg", "exp_avg_sq"))
    med = float(np.median(ms))
    return {
        "params": n_params,
        "optimizer_state_floats": state_floats,
        "state_over_params": state_floats / n_params,
        # fp32 = 값 하나당 4바이트, 2**20바이트 = 1MiB
        "params_mb_fp32": n_params * 4 / 2**20,
        "optimizer_state_mb_fp32": state_floats * 4 / 2**20,
        "ms_per_step_median": med,
        "tokens_per_step": BATCH * N_CTX,
        "tokens_per_sec": BATCH * N_CTX / (med / 1000),
        "note": "단일 스레드 CPU. 시간은 기기마다 다르다.",
    }


def e6():
    # 앞 3개 토큰을 보는 작은 MLP 언어모델(문맥 전체를 보는 GPT와 비교 불가한 약한 모델)
    k, emb, hid = 3, 16, 64
    ctx, y = cnp.ngram_pairs(TRAIN_NP, k)
    vctx, vy = cnp.ngram_pairs(VAL_NP, k)
    p0 = cnp.init_mlp(data.V, k, emb, hid, seed=0)
    params = [p.copy() for p in p0]
    steps, batch, lr, wd = 400, 64, 3e-3, 0.01
    t0 = time.perf_counter()
    losses, _ = cnp.train_loop(params, ctx, y, steps, batch, lr, opt="adamw", wd=wd, clip=1.0, seed=0)
    np_ms = 1000 * (time.perf_counter() - t0) / steps
    # 같은 초기값·같은 미니배치를 torch.optim.AdamW + clip_grad_norm_로 학습한다
    m = MLPLM(p0)
    # NumPy의 DECAY_MASK와 같은 분할(E, W1, W2만 감쇠)
    groups = [{"params": [m.E, m.W1, m.W2], "weight_decay": wd}, {"params": [m.b1, m.b2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, foreach=False)
    # NumPy train_loop와 같은 시드·같은 호출 순서로 미니배치를 뽑아 두 구현이 같은 배치를 본다
    rng = np.random.default_rng(0)
    t_losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        loss = F.cross_entropy(m(torch.tensor(ctx[idx])), torch.tensor(y[idx]))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        t_losses.append(loss.item())
    final = [p.detach().numpy() for p in (m.E, m.W1, m.b1, m.W2, m.b2)]
    return {
        "config": {"k": k, "emb": emb, "hidden": hid, "steps": steps, "batch": batch, "lr": lr, "wd": wd, "clip": 1.0},
        "params": int(sum(p.size for p in p0)),
        "first_loss": losses[0],
        "final_loss_last20": float(np.mean(losses[-20:])),
        "val_loss": cnp.estimate_loss(params, vctx, vy),
        # 두 구현의 스텝별 손실·최종 파라미터 최대 차이. 작을수록 NumPy 옵티마이저가 torch와 같다는 뜻
        "max_abs_loss_diff_vs_torch": float(np.max(np.abs(np.array(losses) - np.array(t_losses)))),
        "max_abs_param_diff_vs_torch": float(max(np.abs(a - b).max() for a, b in zip(params, final))),
        "numpy_ms_per_step": np_ms,
        "note": "앞 3토큰만 보는 MLP라 문맥 전체를 보는 GPT보다 바닥에서 멀다.",
    }


def e7(best):
    # E3에서 고른 최적 설정을 키 문자열에서 해석한 값
    kind, clip, lr = best["kind"], best["clip"], best["lr"]
    res, model, _ = train_gpt(kind, lr, WD_MAIN, clip, 0, keep=True)
    save_checkpoint(CKPT, model, data.VOCAB, {"opt": kind, "lr": lr, "wd": WD_MAIN, "clip": clip, "steps": STEPS, "batch": BATCH, "seed": 0})
    model2, ck = load_checkpoint(CKPT)
    reload_val = eval_loss(model2, VAL)

    def generate(prompt, n_new, greedy, seed=0):
        g = torch.Generator().manual_seed(seed)
        ids = [data.STOI[w] for w in prompt]
        model2.eval()
        with torch.no_grad():
            for _ in range(n_new):
                # 문맥은 최근 N_CTX개 토큰만 사용(위치 임베딩이 N_CTX까지만 있다). [0, -1]: 첫 배치의 마지막 위치 로짓 (V,)
                logits = model2(torch.tensor([ids[-N_CTX:]]))[0, -1]
                # greedy는 최대 확률 토큰, 아니면 온도 1(T=1)로 softmax 표본 추출
                nxt = int(logits.argmax()) if greedy else int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
                ids.append(nxt)
        return ids

    out = {"config": best, "val_loss": res["val_loss"], "reloaded_val_loss": reload_val, "format": ck["format"], "path": "results/ch10_ckpt.pt"}
    for name, greedy in (("greedy", True), ("sample_T1", False)):
        # 마침표 하나에서 출발해 120토큰 생성. 첫 토큰은 프롬프트라 ids[1:]부터 센다
        ids = generate(["."], 120, greedy)
        sents = data.split_sentences(ids[1:])
        out[name] = {
            "text": " ".join(data.VOCAB[i] for i in ids[1:61]),
            "sentences": len(sents),
            "unique_sentences": len(set(sents)),
            # 문법에 맞는 문장의 비율. 문장이 없으면 None
            "valid_fraction": float(np.mean([data.is_valid_sentence(s) for s in sents])) if sents else None,
        }
    return out


def main(which):
    torch.set_num_threads(1)
    # 일부 실험만 다시 돌릴 때 기존 JSON을 읽어 해당 항목만 덮어쓴다
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["env"] = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "machine": platform.machine(), "torch_threads": 1,
    }
    results["setup"] = {"n_ctx": N_CTX, "d": D, "L": L, "h": H, "batch": BATCH, "n_train_windows": N_TRAIN, "n_val_windows": N_VAL, "steps": STEPS, "seeds": list(SEEDS), "wd_main": WD_MAIN}
    results["rules"] = RULES

    def run(name, fn):
        if not which or name in which:
            t0 = time.perf_counter()
            results[name] = fn()
            print(f"{name} done in {time.perf_counter() - t0:.0f}s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    run("E0", e0)
    run("E1", e1)
    run("E1b", e1b)
    # 주의: E1을 한 번도 안 돌린 상태에서 E2 등만 요청하면 results['E1']이 없어 KeyError
    lr_star, grid = results["E1"]["lr_star"], results["E1"]["grid"]
    lr_best = results["E1"]["lr_best"]
    run("E2", lambda: e2(lr_best))
    run("E2b", lambda: e2b(lr_best))
    raw_holder = {}

    def _e3():
        out, raw = e3(grid)
        raw_holder["raw"] = raw
        return out

    run("E3", _e3)
    # E1에서 실패하기 시작하는 첫 학습률(lr*보다 큰 값 중 가장 작은 것). warmup 효과를 보려는 설정
    fail_lr = next((lr for lr in SWEEP_LRS if lr > lr_star), grid[-1])
    run("E4", lambda: e4(fail_lr))
    if "raw" in raw_holder:
        run("E5", lambda: e5(raw_holder["raw"]))

    def _best():
        cands = []
        for key, s in results["E3"].items():
            if s["fail_rate"].startswith("0/"):
                kind, clip, lr = key.split(",")
                clip = None if clip == "clip=None" else float(clip.split("=")[1])
                cands.append((s["val_mean"], {"kind": kind, "clip": clip, "lr": float(lr.split("=")[1]), "key": key}))
        return min(cands, key=lambda c: c[0])[1]

    run("E6", e6)
    run("E7", lambda: e7(_best()))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
