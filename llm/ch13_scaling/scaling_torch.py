"""13장 PyTorch 대응 구현.

- TinyLM: 9장 resnorm_torch.py의 Pre-LN 모델을 복사했다(장끼리 import하지 않는 저장소 규칙).
  장들을 동시에 쓰고 장끼리 import하지 않으므로, 학습 루프도 이 파일에 따로 둔다.
- train_run: AdamW + 선형 warmup(12스텝 고정) + 코사인 감쇠(최대 학습률의 1/10까지). 코사인 길이는 인자로 받아서,
  '데이터 양에 맞춘 스케줄'과 '가장 긴 스케줄 하나를 중간에서 읽기'를 같은 코드로 비교한다.
- fit_chinchilla_torch: Hoffmann 등(2022) 식 (3)·(11)처럼 log 손실의 Huber 손실을 L-BFGS로 최소화한다.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ch13_data as cd


# ---------------------------------------------------------------- 모델 (9장에서 복사, Pre-LN만 남김)


def causal_bool_mask(n, device=None):
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Parameter(torch.randn(d, d) / math.sqrt(d)) for _ in range(4))

    def forward(self, x):
        B, n, d = x.shape
        dk = d // self.h

        def split(t):
            return t.view(B, n, self.h, dk).transpose(1, 2)

        Q, K, V = split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
        scores = scores.masked_fill(causal_bool_mask(n, x.device), float("-inf"))
        heads = torch.softmax(scores, dim=-1) @ V
        return heads.transpose(1, 2).reshape(B, n, d) @ self.Wo


class MLP(nn.Module):
    def __init__(self, d, ff=4):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, ff * d) * math.sqrt(2.0 / d))
        self.b1 = nn.Parameter(torch.zeros(ff * d))
        self.W2 = nn.Parameter(torch.randn(ff * d, d) / math.sqrt(ff * d))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return torch.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn, self.mlp = MultiHeadAttention(d, h), MLP(d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, n_ctx, d, L, h):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(n_ctx, d))
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.final_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1.0 / math.sqrt(d))
            self.head.bias.zero_()

    def forward(self, x):
        h = self.tok(x) + self.pos[: x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.head(self.final_ln(h))


def build(cfg, seed):
    torch.manual_seed(seed)
    return TinyLM(cd.V, cd.N_CTX, cfg["d"], cfg["L"], cfg["h"])


def torch_param_counts(model):
    """(임베딩 제외 N, 전체). Kaplan 등 Table 1처럼 토큰·위치 임베딩을 N에서 뺀다."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok.weight.numel() + model.pos.numel()
    return total - emb, total


# ---------------------------------------------------------------- 학습


SCORED = torch.tensor(cd.scored_positions())


def lm_loss(model, X):
    """값 자리만 센다. 열쇠 토큰은 지프 분포에서 무작위로 뽑혀서 맞힐 구조가 없다."""
    logits = model(X[:, :-1])[:, SCORED]
    return F.cross_entropy(logits.reshape(-1, cd.V), X[:, 1:][:, SCORED].reshape(-1))


def lr_at(step, peak, warmup, schedule_steps):
    """선형 warmup 뒤 코사인으로 peak/10까지 내린다(Hoffmann 등의 10배 감쇠). schedule_steps 이후는 바닥 유지."""
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, schedule_steps - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


WARMUP_STEPS = 12  # 가장 짧은 학습(256스텝)의 약 5%

VAL = torch.tensor(cd.sample_sequences(2048, seed=424242))


def eval_loss(model, X=None):
    model.eval()
    with torch.no_grad():
        out = float(lm_loss(model, VAL if X is None else X))
    model.train()
    return out


def train_run(cfg, tokens, seed, lr, batch=8, schedule_tokens=None, eval_at_tokens=(), warmup_frac=None):
    """tokens개의 새 토큰을 한 번씩 보며 학습하고 검증 손실을 돌려준다.

    tokens는 입력 토큰 수(스텝 × batch × N_CTX)다. schedule_tokens를 주면 코사인 길이를 그 값에 맞춘다
    (기본은 tokens와 같다). eval_at_tokens의 지점마다 중간 검증 손실도 기록한다.
    warmup_frac을 주면 warmup을 스케줄 길이의 비율로 잡는다(7절 '흔한 실패' 재현용).
    """
    per_step = batch * cd.N_CTX
    steps = tokens // per_step
    sched_steps = (schedule_tokens or tokens) // per_step
    # 기본은 모든 학습에서 같은 절대 길이. 그래야 두 프로토콜의 차이가 코사인 길이 하나로 좁혀진다
    warmup = WARMUP_STEPS if warmup_frac is None else max(1, int(sched_steps * warmup_frac))
    data = torch.tensor(cd.training_stream(steps * batch, seed=10_000 + seed))  # 모델 크기와 무관하게 같은 토큰열
    model = build(cfg, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    checkpoints = {int(t) // per_step: int(t) for t in eval_at_tokens}
    mid, nan = {}, False
    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr, warmup, sched_steps)
        loss = lm_loss(model, data[step * batch : (step + 1) * batch])
        if not torch.isfinite(loss):
            nan = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step + 1 in checkpoints:
            mid[str(checkpoints[step + 1])] = eval_loss(model)
    sec = time.perf_counter() - t0
    n_ne, n_tot = torch_param_counts(model)
    return {
        "val_loss": float("nan") if nan else eval_loss(model),
        "nan": nan,
        "intermediate_val_loss": mid,
        "N": n_ne,
        "N_total": n_tot,
        "D": steps * per_step,
        "seconds": sec,
    }


# ---------------------------------------------------------------- 피팅 (Hoffmann 등 식 (11)의 LSE 꼴)


def huber(r, delta):
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def fit_chinchilla_torch(N, D, L, delta=1e-3, inits=None, E_fixed=None):
    """log L ≈ LSE(a - α log N, b - β log D, e) 를 Huber_δ로 맞춘다. A=e^a, B=e^b, E=e^e.

    여러 초기값에서 L-BFGS를 돌려 목적함수가 가장 작은 해를 고른다(Hoffmann 등 §3.3).
    E_fixed를 주면 e = log E_fixed로 고정한다.
    """
    N, D, L = (torch.tensor(np.asarray(v, dtype=np.float64)) for v in (N, D, L))
    lN, lD, lL = N.log(), D.log(), L.log()
    if inits is None:
        inits = [(a, b, e, al, be) for a in (0.0, 5.0, 10.0) for b in (0.0, 5.0, 10.0)
                 for e in (-1.0, 0.0, 1.0) for al in (0.2, 0.5) for be in (0.2, 0.5)]
    best = None
    for init in inits:
        if E_fixed is not None:
            init = (init[0], init[1], math.log(E_fixed), init[3], init[4])
        th = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        fix = torch.tensor([1.0, 1.0, 0.0 if E_fixed is not None else 1.0, 1.0, 1.0], dtype=torch.float64)
        th.register_hook(lambda g: g * fix)  # 고정한 성분은 기울기를 0으로
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            a, b, e, al, be = th
            pred = torch.logsumexp(torch.stack([a - al * lN, b - be * lD, e.expand_as(lN)]), dim=0)
            obj = huber(pred - lL, delta).sum()
            obj.backward()
            return obj

        try:
            opt.step(closure)
        except RuntimeError:
            continue
        with torch.no_grad():
            a, b, e, al, be = th
            pred = torch.logsumexp(torch.stack([a - al * lN, b - be * lD, e.expand_as(lN)]), dim=0)
            obj = float(huber(pred - lL, delta).sum())
        if np.isfinite(obj) and (best is None or obj < best[0]):
            best = (obj, [float(v) for v in th.detach()])
    obj, (a, b, e, al, be) = best
    return {"A": math.exp(a), "B": math.exp(b), "E": math.exp(e), "alpha": al, "beta": be, "objective": obj}
