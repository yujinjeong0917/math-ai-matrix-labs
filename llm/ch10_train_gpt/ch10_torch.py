"""10장 PyTorch 대응 구현.

- MLPLM: ch10_np.py의 2층 MLP 언어모델과 같은 수식. NumPy 옵티마이저 궤적을 torch.optim과 대조할 때 쓴다.
- TinyGPT: 9장 Pre-LN 블록 L개를 쌓은 작은 GPT. 9장 resnorm_torch.py에서 Pre-LN 경로만 복사했다
  (장끼리 import하지 않는다). 출력층 앞에 LN을 하나 더 둔다.
- make_optimizer: "adam_l2"는 torch.optim.Adam(weight_decay=λ), "adamw"는 torch.optim.AdamW(weight_decay=λ).
  PyTorch 2.8 문서에서 Adam의 weight_decay는 "weight decay (L2 penalty)"로 적혀 있고, 알고리즘 상자에서
  g ← g + λθ로 기울기에 더해진다. AdamW는 θ ← θ − γλθ로 따로 줄인다.
- 감쇠는 2차원 이상 텐서(행렬·임베딩)에만 건다. LN의 γ·β와 편향은 감쇠하지 않는다.
- 체크포인트 형식(11~13장이 같은 모델을 불러 쓴다): {"format": "ch10-tinygpt-v1", "config", "vocab", "state_dict"}
"""

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------- 2층 MLP (NumPy 대조용)


class MLPLM(nn.Module):
    def __init__(self, params_np):
        super().__init__()
        self.E, self.W1, self.b1, self.W2, self.b2 = (nn.Parameter(torch.tensor(p, dtype=torch.float64)) for p in params_np)

    def forward(self, ctx):
        e = self.E[ctx].reshape(len(ctx), -1)
        return torch.tanh(e @ self.W1 + self.b1) @ self.W2 + self.b2


# ---------------------------------------------------------------- 9장에서 복사한 Pre-LN 블록
# 출처: llm/ch09_residual_norm/resnorm_torch.py (arch="pre" 경로만)


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
    """Pre-LN: x + F(LN(x))."""

    def __init__(self, d, h):
        super().__init__()
        self.attn, self.mlp = MultiHeadAttention(d, h), MLP(d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, n_ctx, d, L, h):
        super().__init__()
        self.config = {"vocab": vocab, "n_ctx": n_ctx, "d": d, "L": L, "h": h}
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


# ---------------------------------------------------------------- 옵티마이저·체크포인트


def param_groups(model, wd):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(kind, model, lr, wd, foreach=None):
    groups = param_groups(model, wd)
    if kind == "adam_l2":
        return torch.optim.Adam(groups, lr=lr, foreach=foreach)
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=lr, foreach=foreach)
    if kind == "sgd":  # 비교용: 모멘텀 없는 SGD, 모든 숫자에 같은 학습률
        return torch.optim.SGD(groups, lr=lr, foreach=foreach)
    raise ValueError(kind)


CKPT_FORMAT = "ch10-tinygpt-v1"


def save_checkpoint(path, model, vocab, train_config):
    torch.save(
        {"format": CKPT_FORMAT, "config": model.config, "train_config": train_config, "vocab": list(vocab), "state_dict": model.state_dict()},
        path,
    )


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["format"] == CKPT_FORMAT
    c = ck["config"]
    model = TinyGPT(c["vocab"], c["n_ctx"], c["d"], c["L"], c["h"])
    model.load_state_dict(ck["state_dict"])
    return model, ck
