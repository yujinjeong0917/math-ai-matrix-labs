"""7장 PyTorch 대응 구현.

attention_np.py와 같은 수식을 텐서로 옮기고, 학습 비교에 쓸 세 모델을 정의한다.
- self_attention: NumPy 구현과 한 줄씩 대응한다.
- RNNClassifier / LSTMClassifier / AttentionClassifier: 같은 입력에서 학습 안정성을 비교한다.

텐서 모양: 임베딩 x (batch, n, d), 가중치 W* (d, d_k). 분류기 입력 토큰 id (batch, n), 출력 로짓 (batch, n_classes).
experiments.py의 E4(첫 토큰 기억 과제), E5(스케일 유무)에서 쓴다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def self_attention(x, Wq, Wk, Wv, causal=False, scale=True, return_weights=False):
    # 식: Q = X W_Q 등. (..., n, d) @ (d, d_k) -> (..., n, d_k), 앞쪽 batch 차원은 그대로 따라간다
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    # 식: Q K^T. 마지막 두 축만 바꿔야 해서 transpose(-2, -1): (..., n, n)
    scores = Q @ K.transpose(-2, -1)
    if scale:
        # 식: / sqrt(d_k)
        scores = scores / math.sqrt(K.shape[-1])
    if causal:
        n = x.shape[-2]
        # 위쪽 삼각(j > i)을 -inf로 채운다. NumPy 구현의 causal_mask 덧셈과 같은 효과
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    # A: (..., n, n), 마지막 축(키 방향)으로 합 1
    A = torch.softmax(scores, dim=-1)
    return (A @ V, A) if return_weights else A @ V


def self_attention_builtin(x, Wq, Wk, Wv, causal=False):
    """PyTorch 내장 커널. 직접 구현과 수치가 같아야 한다."""
    # 내장 커널은 scale 기본값이 1/sqrt(d_k)라 위 직접 구현(scale=True)과 대응한다
    return F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=causal)


class RNNClassifier(nn.Module):
    def __init__(self, vocab, d, n_classes, cell="rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(d, d, batch_first=True)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        # h: (batch, n, d) -> 마지막 시점 h[:, -1]: (batch, d) -> 로짓 (batch, n_classes)
        return self.out(h[:, -1])  # 마지막 시점의 상태만으로 답한다


class AttentionClassifier(nn.Module):
    """causal self-attention 한 층. 위치 정보는 일부러 넣지 않는다(8장의 주제)."""

    def __init__(self, vocab, d, n_classes, scale=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)  # 기본 초기화 N(0, 1): 각주 4의 분산 1 가정과 맞춘다
        self.Wq = nn.Parameter(torch.randn(d, d) / math.sqrt(d))  # 분산을 보존하는 초기화
        self.Wk = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.Wv = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.out = nn.Linear(d, n_classes)
        self.scale = scale

    def forward(self, x):
        h = self.emb(x)
        # y: (batch, n, d). causal이라 마지막 위치 y[:, -1]은 모든 이전 위치를 볼 수 있다
        y = self_attention(h, self.Wq, self.Wk, self.Wv, causal=True, scale=self.scale)
        return self.out(y[:, -1])

    def last_position_weights(self, x):
        """마지막 위치가 각 위치에 준 어텐션 가중치 (batch, n)."""
        _, A = self_attention(self.emb(x), self.Wq, self.Wk, self.Wv, causal=True, scale=self.scale, return_weights=True)
        return A[:, -1]
