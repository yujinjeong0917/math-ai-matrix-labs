"""8장 PyTorch 대응 구현.

positional_np.py와 같은 수식을 텐서로 옮기고, 학습 비교에 쓸 모델 하나를 정의한다.
- RoPE는 두 가지로 쓴다: 회전행렬(짝마다 cos/sin) 구현과 복소수 곱 구현. 테스트에서 서로 대조한다.
- PosAttentionModel: 7장 AttentionClassifier를 멀티헤드로 늘리고 위치 방식(kind)만 바꿔 끼운다.

텐서 모양: 입력 토큰 id (B, n), 출력 로짓 (B, n, vocab). 어텐션 가중치 A는 (B, heads, n, n).
B = 배치 크기, n = 시퀀스 길이, dh = 헤드당 차원 = d / heads.
"""

import math

import torch
import torch.nn as nn

KINDS = ("none", "sin", "learned", "rope", "alibi")


def sinusoidal(n, d, base=10000.0, dtype=torch.float32):
    pos = torch.arange(n, dtype=dtype)[:, None]
    # NumPy 구현과 같은 식: 1 / base^(2i/d)
    freq = base ** (-torch.arange(0, d, 2, dtype=dtype) / d)
    pe = torch.zeros(n, d, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def rope_rotate(x, positions, base=10000.0):
    """회전행렬 구현. x: (..., n, d), positions: (n,)."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    # (n, 1) * (d/2,) -> (n, d/2). Q의 (..., n, d/2)와 마지막 두 축이 맞아 브로드캐스트
    ang = positions.to(x.dtype)[:, None] * theta
    c, s = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    # 회전 결과 두 성분을 마지막 축에 쌓아 (..., n, d/2, 2)로 만든 뒤 flatten해 원래 (..., n, d) 짝 순서를 복원
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)


def rope_rotate_complex(x, positions, base=10000.0):
    """복소수 곱 구현. 짝 (x0, x1)을 복소수 x0 + i x1로 보고 e^{i m theta}를 곱한다 (RoFormer 식 (12))."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    ang = positions.to(x.dtype)[:, None] * theta
    # (..., d) -> (..., d/2, 2) -> 복소수 (..., d/2). 마지막 축이 연속(contiguous)이어야 view_as_complex가 된다
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2).contiguous())
    # 극형식 e^{i m theta} = polar(크기 1, 각 m*theta)를 곱하면 위 회전행렬과 같은 결과
    return torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).flatten(-2)


def alibi_slopes(n_heads):
    start = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([start ** (h + 1) for h in range(n_heads)])


def alibi_bias(n, slopes):
    """(heads, n, n). 칸 (i, j)는 -slope * (i - j)."""
    # dist[i, j] = i - j, (n, n). slopes (heads,) -> (heads, 1, 1)로 늘려 (heads, n, n)
    dist = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).to(slopes.dtype)
    return -slopes[:, None, None] * dist


def attention_with_pos(x, Wq, Wk, Wv, kind="none", pos_table=None, slope=0.25, causal=True):
    """positional_np.attention_with_pos와 한 줄씩 대응하는 단일 헤드 버전."""
    n = x.shape[-2]
    if kind == "sin":
        x = x + sinusoidal(n, x.shape[-1], dtype=x.dtype)
    elif kind == "learned":
        x = x + pos_table[:n]
    # 식: Q = X W_Q 등. 이 단일 헤드 버전은 (n, d) 또는 (batch, n, d) 입력을 받는다
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    if kind == "rope":
        pos = torch.arange(n)
        Q, K = rope_rotate(Q, pos), rope_rotate(K, pos)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(K.shape[-1])
    if kind == "alibi":
        # 기울기 하나를 (1,)로 만들어 첫 헤드 (n, n)만 꺼낸다
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
        # d를 헤드 수로 나눠 dh를 정한다. d가 헤드 수의 배수여야 view가 성립(가정)
        self.kind, self.h, self.dh, self.n_max = kind, n_heads, d // n_heads, n_max
        self.emb = nn.Embedding(vocab, d)  # N(0, 1)
        if kind == "learned":
            self.pos = nn.Embedding(n_max, d)  # 같은 N(0, 1) 초기화. 위치 n_max 이상의 칸은 없다
        self.Wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.out = nn.Linear(d, vocab)
        # slopes는 학습하지 않는 상수. persistent=False라 state_dict에는 저장되지 않는다
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
        # (B, n, d) -> (B, n, heads, dh) -> (B, heads, n, dh)
        split = lambda t: t.view(B, n, self.h, self.dh).transpose(1, 2)  # (B, heads, n, dh)
        Q, K, V = split(h @ self.Wq), split(h @ self.Wk), split(h @ self.Wv)
        if self.kind == "rope":
            pos = torch.arange(n)
            Q, K = rope_rotate(Q, pos), rope_rotate(K, pos)
        # 식: Q K^T / sqrt(d_h): (B, heads, n, n). 헤드별 차원 dh로 나눈다
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dh)
        if self.kind == "alibi":
            scores = scores + alibi_bias(n, self.slopes)
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        A = torch.softmax(scores, dim=-1)
        # (B, heads, n, dh) -> (B, n, heads, dh) -> (B, n, d): 헤드를 이어 붙인다(concat). 출력 투영 W_O는 없다(모형의 단순화)
        y = (A @ V).transpose(1, 2).reshape(B, n, -1)
        logits = self.out(y)
        return (logits, A) if return_weights else logits
