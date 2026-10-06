"""7장 PyTorch 대응 구현.

attention_np.py와 같은 수식을 텐서로 옮기고, 학습 비교에 쓸 세 모델을 정의한다.
- self_attention: NumPy 구현과 한 줄씩 대응한다.
- RNNClassifier / LSTMClassifier / AttentionClassifier: 같은 입력에서 학습 안정성을 비교한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def self_attention(x, Wq, Wk, Wv, causal=False, scale=True, return_weights=False):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(K.shape[-1])
    if causal:
        n = x.shape[-2]
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    A = torch.softmax(scores, dim=-1)
    return (A @ V, A) if return_weights else A @ V


def self_attention_builtin(x, Wq, Wk, Wv, causal=False):
    """PyTorch 내장 커널. 직접 구현과 수치가 같아야 한다."""
    return F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=causal)


class RNNClassifier(nn.Module):
    def __init__(self, vocab, d, n_classes, cell="rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(d, d, batch_first=True)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
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
        y = self_attention(h, self.Wq, self.Wk, self.Wv, causal=True, scale=self.scale)
        return self.out(y[:, -1])

    def last_position_weights(self, x):
        """마지막 위치가 각 위치에 준 어텐션 가중치 (batch, n)."""
        _, A = self_attention(self.emb(x), self.Wq, self.Wk, self.Wv, causal=True, scale=self.scale, return_weights=True)
        return A[:, -1]
