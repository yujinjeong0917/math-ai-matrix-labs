"""9장 PyTorch 대응 구현.

resnorm_np.py와 같은 수식을 텐서로 옮기고, 깊이·배치 방식 비교에 쓸 작은 언어모델을 정의한다.
- MultiHeadAttention / Block: NumPy 구현과 한 줄씩 대응한다(배치 차원만 붙는다).
- TinyLM: 토큰 임베딩 + 학습되는 위치 임베딩 + 블록 L개 + 출력층.
  arch="pre"이면 출력층 앞에 LN을 하나 더 둔다(Xiong 등 2020이 정의한 Pre-LN).
"""

import math

import torch
import torch.nn as nn


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

        def split(t):  # (B, n, d) -> (B, h, n, dk)
            return t.view(B, n, self.h, dk).transpose(1, 2)

        Q, K, V = split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
        scores = scores.masked_fill(causal_bool_mask(n, x.device), float("-inf"))
        heads = torch.softmax(scores, dim=-1) @ V  # (B, h, n, dk)
        return heads.transpose(1, 2).reshape(B, n, d) @ self.Wo  # concat 후 W_O


class MLP(nn.Module):
    def __init__(self, d, ff=4):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, ff * d) * math.sqrt(2.0 / d))  # ReLU 앞이라 He 초기화
        self.b1 = nn.Parameter(torch.zeros(ff * d))
        self.W2 = nn.Parameter(torch.randn(ff * d, d) / math.sqrt(ff * d))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return torch.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class Block(nn.Module):
    def __init__(self, d, h, arch):
        super().__init__()
        self.arch = arch
        self.attn, self.mlp = MultiHeadAttention(d, h), MLP(d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def _sub(self, x, f, ln):
        a = self.arch
        if a == "plain":
            return f(x)
        if a == "norm":
            return ln(f(x))
        if a == "residual":
            return x + f(x)
        if a == "post":
            return ln(x + f(x))
        if a == "pre":
            return x + f(ln(x))
        raise ValueError(a)

    def forward(self, x):
        x = self._sub(x, self.attn, self.ln1)
        return self._sub(x, self.mlp, self.ln2)


class TinyLM(nn.Module):
    def __init__(self, vocab, n_ctx, d, L, h, arch):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)  # 기본 초기화 N(0, 1)
        self.pos = nn.Parameter(torch.randn(n_ctx, d))
        self.blocks = nn.ModuleList(Block(d, h, arch) for _ in range(L))
        self.final_ln = nn.LayerNorm(d) if arch == "pre" else nn.Identity()
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1.0 / math.sqrt(d))
            self.head.bias.zero_()

    def forward(self, x, return_hidden=False):
        hid = [self.tok(x) + self.pos[: x.shape[1]]]
        for b in self.blocks:
            hid.append(b(hid[-1]))
        logits = self.head(self.final_ln(hid[-1]))
        return (logits, hid) if return_hidden else logits


def block_from_numpy(p, d, h, arch):
    """resnorm_np.block과 같은 파라미터를 가진 float64 Block을 만든다(수치 대조용)."""
    blk = Block(d, h, arch).double()
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    with torch.no_grad():
        for k in ("Wq", "Wk", "Wv", "Wo"):
            getattr(blk.attn, k).copy_(t(p[k]))
        for k in ("W1", "b1", "W2", "b2"):
            getattr(blk.mlp, k).copy_(t(p[k]))
        blk.ln1.weight.copy_(t(p["g1"])), blk.ln1.bias.copy_(t(p["be1"]))
        blk.ln2.weight.copy_(t(p["g2"])), blk.ln2.bias.copy_(t(p["be2"]))
    return blk
