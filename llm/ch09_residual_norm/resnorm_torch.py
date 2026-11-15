"""9장 PyTorch 대응 구현.

resnorm_np.py와 같은 수식을 텐서로 옮기고, 깊이·배치 방식 비교에 쓸 작은 언어모델을 정의한다.
- MultiHeadAttention / Block: NumPy 구현과 한 줄씩 대응한다(배치 차원만 붙는다).
- TinyLM: 토큰 임베딩 + 학습되는 위치 임베딩 + 블록 L개 + 출력층.
  arch="pre"이면 출력층 앞에 LN을 하나 더 둔다(Xiong 등 2020이 정의한 Pre-LN).

텐서 모양: 토큰 id (B, n) -> 로짓 (B, n, vocab). 블록 안 은닉 상태는 (B, n, d).
B = 배치 크기, n = 시퀀스 길이, d = 모델 차원, h = 헤드 수, L = 층 수. 초기화 규모는 experiments.py E1의 층별 활성 진단과 맞물린다.
"""

import math

import torch
import torch.nn as nn


def causal_bool_mask(n, device=None):
    # 위쪽 삼각(j > i)이 True인 마스크. True인 칸을 -inf로 채운다
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        # 네 행렬 모두 (d, d), 분산 1/d로 초기화(입력 분산 보존)
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Parameter(torch.randn(d, d) / math.sqrt(d)) for _ in range(4))

    def forward(self, x):
        # B: 배치, n: 길이, d: 모델 차원
        B, n, d = x.shape
        dk = d // self.h

        def split(t):  # (B, n, d) -> (B, h, n, dk)
            return t.view(B, n, self.h, dk).transpose(1, 2)

        Q, K, V = split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv)
        # 식: Q K^T / sqrt(d_k): (B, h, n, n)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
        scores = scores.masked_fill(causal_bool_mask(n, x.device), float("-inf"))
        # 식: softmax(scores) V -> (B, h, n, dk)
        heads = torch.softmax(scores, dim=-1) @ V  # (B, h, n, dk)
        # (B, h, n, dk) -> (B, n, h, dk) -> (B, n, d). concat 후 W_O를 곱한다
        return heads.transpose(1, 2).reshape(B, n, d) @ self.Wo  # concat 후 W_O


class MLP(nn.Module):
    def __init__(self, d, ff=4):
        super().__init__()
        # He 초기화: 분산 2/d(ReLU가 절반을 0으로 만드는 점을 보정)
        self.W1 = nn.Parameter(torch.randn(d, ff * d) * math.sqrt(2.0 / d))  # ReLU 앞이라 He 초기화
        self.b1 = nn.Parameter(torch.zeros(ff * d))
        # 분산 1/(ff*d)
        self.W2 = nn.Parameter(torch.randn(ff * d, d) / math.sqrt(ff * d))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        # 식: ReLU(x W1 + b1) W2 + b2. (B, n, d) -> (B, n, ff*d) -> (B, n, d)
        return torch.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class Block(nn.Module):
    def __init__(self, d, h, arch):
        super().__init__()
        self.arch = arch
        self.attn, self.mlp = MultiHeadAttention(d, h), MLP(d)
        # 부분층마다 LayerNorm 하나씩. nn.LayerNorm 기본 eps는 1e-5로 NumPy 구현과 같다
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    # resnorm_np.sublayer와 같은 다섯 규칙
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
        # 학습되는 위치 임베딩 (n_ctx, d), N(0, 1)
        self.pos = nn.Parameter(torch.randn(n_ctx, d))
        self.blocks = nn.ModuleList(Block(d, h, arch) for _ in range(L))
        self.final_ln = nn.LayerNorm(d) if arch == "pre" else nn.Identity()
        self.head = nn.Linear(d, vocab)
        # 출력층 가중치를 분산 1/d로 다시 뽑는다(초기 로짓 크기를 d에 무관하게)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1.0 / math.sqrt(d))
            self.head.bias.zero_()

    def forward(self, x, return_hidden=False):
        # hid[0]은 (임베딩 + 위치) 입력, hid[l]은 l번째 블록 출력. 모두 (B, n, d). E1 진단에서 층별로 쓴다
        hid = [self.tok(x) + self.pos[: x.shape[1]]]
        for b in self.blocks:
            hid.append(b(hid[-1]))
        # pre에서만 final_ln이 LayerNorm, 나머지는 항등
        logits = self.head(self.final_ln(hid[-1]))
        return (logits, hid) if return_hidden else logits


def block_from_numpy(p, d, h, arch):
    """resnorm_np.block과 같은 파라미터를 가진 float64 Block을 만든다(수치 대조용)."""
    blk = Block(d, h, arch).double()
    # torch에 맞춘 float64 변환. 테스트가 NumPy와 1e-10으로 대조하려고
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    with torch.no_grad():
        for k in ("Wq", "Wk", "Wv", "Wo"):
            getattr(blk.attn, k).copy_(t(p[k]))
        for k in ("W1", "b1", "W2", "b2"):
            getattr(blk.mlp, k).copy_(t(p[k]))
        # NumPy의 gamma, beta를 LayerNorm의 weight, bias로 대응
        blk.ln1.weight.copy_(t(p["g1"])), blk.ln1.bias.copy_(t(p["be1"]))
        blk.ln2.weight.copy_(t(p["g2"])), blk.ln2.bias.copy_(t(p["be2"]))
    return blk
