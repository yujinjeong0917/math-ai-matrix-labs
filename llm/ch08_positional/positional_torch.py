"""8장 PyTorch 대응 구현.

positional_np.py와 같은 수식을 텐서로 옮기고, 학습 비교에 쓸 모델 하나를 정의한다.
- RoPE는 두 가지로 쓴다: 회전행렬(짝마다 cos/sin) 구현과 복소수 곱 구현. 테스트에서 서로 대조한다.
- PosAttentionModel: 7장 AttentionClassifier를 멀티헤드로 늘리고 위치 방식(kind)만 바꿔 끼운다.
"""

import math

import torch
import torch.nn as nn

KINDS = ("none", "sin", "learned", "rope", "alibi")


def sinusoidal(n, d, base=10000.0, dtype=torch.float32):
    pos = torch.arange(n, dtype=dtype)[:, None]
    freq = base ** (-torch.arange(0, d, 2, dtype=dtype) / d)
    pe = torch.zeros(n, d, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def rope_rotate(x, positions, base=10000.0):
    """회전행렬 구현. x: (..., n, d), positions: (n,)."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = positions.to(x.dtype)[:, None] * theta
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)


def rope_rotate_complex(x, positions, base=10000.0):
    """복소수 곱 구현. 짝 (x0, x1)을 복소수 x0 + i x1로 보고 e^{i m theta}를 곱한다 (RoFormer 식 (12))."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = positions.to(x.dtype)[:, None] * theta
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2).contiguous())
    return torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).flatten(-2)


def alibi_slopes(n_heads):
    start = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([start ** (h + 1) for h in range(n_heads)])


def alibi_bias(n, slopes):
    """(heads, n, n). 칸 (i, j)는 -slope * (i - j)."""
    dist = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).to(slopes.dtype)
    return -slopes[:, None, None] * dist


def attention_with_pos(x, Wq, Wk, Wv, kind="none", pos_table=None, slope=0.25, causal=True):
    """positional_np.attention_with_pos와 한 줄씩 대응하는 단일 헤드 버전."""
    n = x.shape[-2]
    if kind == "sin":
        x = x + sinusoidal(n, x.shape[-1], dtype=x.dtype)
    elif kind == "learned":
        x = x + pos_table[:n]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    if kind == "rope":
        pos = torch.arange(n)
        Q, K = rope_rotate(Q, pos), rope_rotate(K, pos)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    if kind == "alibi":
        scores = scores + alibi_bias(n, torch.tensor([slope], dtype=x.dtype))[0]
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(scores, dim=-1) @ V


class PosAttentionModel(nn.Module):
    """임베딩 -> (위치) -> causal 멀티헤드 self-attention 한 층 -> 위치마다 어휘 로짓.

    위치 방식만 다르고 나머지(크기, 초기화, 헤드 수)는 모두 같다. 7장처럼 잔차·정규화는 없다(9장의 주제).
    """

    def __init__(self, vocab, d=64, n_heads=4, kind="none", n_max=64):
        super().__init__()
        assert kind in KINDS
        self.kind, self.h, self.dh, self.n_max = kind, n_heads, d // n_heads, n_max
        self.emb = nn.Embedding(vocab, d)  # N(0, 1)
        if kind == "learned":
            self.pos = nn.Embedding(n_max, d)  # 같은 N(0, 1) 초기화. 위치 n_max 이상의 칸은 없다
        self.Wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.out = nn.Linear(d, vocab)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def forward(self, x, return_weights=False):
        B, n = x.shape
        h = self.emb(x)
        if self.kind == "sin":
            h = h + sinusoidal(n, h.shape[-1])
        elif self.kind == "learned":
            if n > self.n_max:
                raise ValueError("learned: 학습 길이를 넘는 위치에는 벡터가 없다")
            h = h + self.pos.weight[:n]
        split = lambda t: t.view(B, n, self.h, self.dh).transpose(1, 2)  # (B, heads, n, dh)
        Q, K, V = split(h @ self.Wq), split(h @ self.Wk), split(h @ self.Wv)
        if self.kind == "rope":
            pos = torch.arange(n)
            Q, K = rope_rotate(Q, pos), rope_rotate(K, pos)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dh)
        if self.kind == "alibi":
            scores = scores + alibi_bias(n, self.slopes)
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        A = torch.softmax(scores, dim=-1)
        y = (A @ V).transpose(1, 2).reshape(B, n, -1)
        logits = self.out(y)
        return (logits, A) if return_weights else logits
