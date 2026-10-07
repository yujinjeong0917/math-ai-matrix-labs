"""11장 PyTorch 대응 구현.

- apply_temperature / top_k_filter / top_p_filter: decoding_np.py와 같은 수식을 텐서로 옮긴 것.
  같은 로짓에서 남기는 칸(마스크)이 NumPy와 같아야 한다.
- TinyLM: 9장 `ch09_residual_norm/resnorm_torch.py`의 Pre-LN 소형 GPT를 그대로 복사했다
  (장끼리 import하지 않는 저장소 규칙). 10장 체크포인트가 아직 없어서 이 장에서 직접 학습한다.
- numpy_model: 학습된 TinyLM을 decoding_np가 쓰는 `model(ids) -> logits` 함수로 감싼다.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 필터


def apply_temperature(logits, T):
    if T == 0:
        out = torch.full_like(logits, float("-inf"))
        return out.scatter(-1, logits.argmax(-1, keepdim=True), 0.0)
    return logits / T


def top_k_filter(logits, k):
    if k is None or k >= logits.shape[-1]:
        return logits.clone()
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    if p is None or p >= 1.0:
        return logits.clone()
    probs = torch.softmax(logits, dim=-1)
    sorted_p, order = torch.sort(probs, dim=-1, descending=True)
    cum = sorted_p.cumsum(-1)
    keep_sorted = (cum - sorted_p) < p  # p를 처음 넘기게 만든 토큰까지 포함
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    return logits.masked_fill(~keep, float("-inf"))


def shape_logits(logits, T=1.0, k=None, p=None):
    return top_p_filter(top_k_filter(apply_temperature(logits, T), k), p)


def sample_from_logits(logits, generator, T=1.0, k=None, p=None):
    probs = torch.softmax(shape_logits(logits, T, k, p), dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------- 9장에서 복사한 Pre-LN 소형 GPT


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
    """Pre-LN: x + f(LN(x))."""

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
        self.n_ctx = n_ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(n_ctx, d))  # 학습되는 위치 임베딩: n_ctx칸까지만 있다
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


def numpy_model(model):
    """`ids (B, L) -> 다음 토큰 로짓 (B, V)`. 문맥이 n_ctx보다 길면 마지막 n_ctx개만 넣는다.

    위치 임베딩 표가 n_ctx칸뿐이라, 자르지 않으면 n_ctx+1번째 토큰에서 인덱스 오류가 난다(7절 흔한 실패 1).
    """
    model.eval()

    def f(ids):
        ids = np.atleast_2d(ids)[:, -model.n_ctx :]
        with torch.no_grad():
            return model(torch.as_tensor(ids, dtype=torch.long))[:, -1].double().numpy()

    return f
