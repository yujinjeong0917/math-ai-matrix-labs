"""5장 PyTorch 대응 구현.

- rnn_manual / lstm_manual: rnn_np.py와 같은 식을 텐서로 옮긴 직접 구현.
- to_torch_lstm_order: NumPy 쪽 게이트 순서 (i, f, o, g)를 nn.LSTM의 (i, f, g, o)로 바꾼다.
- RNNClassifier: 첫 토큰 맞히기 학습용 모델(7장 attention_torch.py에서 복사, forget 편향 옵션 추가).
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_manual(x, W, U, b):
    """rnn_np.rnn_forward와 같은 식. x: (n, d_in) -> (n, H)."""
    h = torch.zeros(W.shape[0], dtype=x.dtype)
    hs = []
    for t in range(x.shape[0]):
        h = torch.tanh(W @ h + U @ x[t] + b)
        hs.append(h)
    return torch.stack(hs)


def lstm_manual(x, Wx, Wh, b):
    """게이트 순서 (i, f, o, g). rnn_np.lstm_cell_forward와 한 줄씩 대응한다."""
    H = Wh.shape[1]
    h = torch.zeros(H, dtype=x.dtype)
    c = torch.zeros(H, dtype=x.dtype)
    hs = []
    for t in range(x.shape[0]):
        z = Wx @ x[t] + Wh @ h + b
        i, f, o = torch.sigmoid(z[:H]), torch.sigmoid(z[H : 2 * H]), torch.sigmoid(z[2 * H : 3 * H])
        g = torch.tanh(z[3 * H :])
        c = f * c + i * g
        h = o * torch.tanh(c)
        hs.append(h)
    return torch.stack(hs)


def to_torch_lstm_order(M):
    """(i, f, o, g) 순서로 쌓인 4H 행을 nn.LSTM의 (i, f, g, o) 순서로 바꾼다."""
    H = M.shape[0] // 4
    i, f, o, g = M[:H], M[H : 2 * H], M[2 * H : 3 * H], M[3 * H :]
    # M이 텐서/배열 어느 쪽이든 행(4H) 방향으로 이어 붙인다. 편향 (4H,)에도 같은 함수를 쓴다.
    return torch.cat([i, f, g, o]) if isinstance(M, torch.Tensor) else np.concatenate([i, f, g, o])


def builtin_rnn(x, W, U, b):
    """nn.RNN에 같은 가중치를 넣는다. bias_hh는 0으로 두고 b를 bias_ih에 넣는다."""
    # nn.RNN은 기본 tanh. 입력 차원 = U의 열, 은닉 = W의 행. 시퀀스 (n, d_in)는 배치 없는 입력으로 받는다.
    m = nn.RNN(U.shape[1], W.shape[0], dtype=x.dtype)
    with torch.no_grad():
        m.weight_hh_l0.copy_(W)
        m.weight_ih_l0.copy_(U)
        m.bias_ih_l0.copy_(b)
        # nn.RNN/LSTM은 편향이 둘(ih, hh)이라 합이 b가 되도록 hh를 0으로 둔다.
        m.bias_hh_l0.zero_()
    out, _ = m(x)
    return out


def builtin_lstm(x, Wx, Wh, b):
    """nn.LSTM에 같은 가중치를 게이트 순서만 바꿔 넣는다."""
    m = nn.LSTM(Wx.shape[1], Wh.shape[1], dtype=x.dtype)
    with torch.no_grad():
        m.weight_ih_l0.copy_(to_torch_lstm_order(Wx))
        m.weight_hh_l0.copy_(to_torch_lstm_order(Wh))
        m.bias_ih_l0.copy_(to_torch_lstm_order(b))
        m.bias_hh_l0.zero_()
    out, _ = m(x)
    return out


class RNNClassifier(nn.Module):
    """임베딩 → RNN 또는 LSTM → 마지막 시점 상태로 분류. (7장 attention_torch.py에서 복사)"""

    def __init__(self, vocab, d, n_classes, cell="rnn", forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        # hidden 크기를 임베딩 차원 d와 같게 둔다. batch_first=True라 입력은 (B, n, d).
        self.rnn = (nn.LSTM if cell == "lstm" else nn.RNN)(d, d, batch_first=True)
        self.out = nn.Linear(d, n_classes)
        if cell == "lstm" and forget_bias is not None:
            with torch.no_grad():  # nn.LSTM 게이트 순서 (i, f, g, o): forget 편향만 바꾼다
                # nn.LSTM 순서 (i, f, g, o)에서 두 번째 블록이 forget 게이트. 큰 값을 주면 처음엔 기억을 거의 지우지 않는다.
                self.rnn.bias_ih_l0[d : 2 * d].fill_(forget_bias)
                self.rnn.bias_hh_l0[d : 2 * d].fill_(0.0)

    def forward(self, x, return_inputs=False):
        e = self.emb(x)
        if return_inputs:
            # 입력 임베딩이 리프가 아니므로 backward 뒤에도 기울기가 남게 한다(시점별 기울기 크기 측정용).
            e.retain_grad()
        h, _ = self.rnn(e)
        # h: (B, n, d)에서 마지막 시점만 -> (B, d) -> (B, n_classes).
        logits = self.out(h[:, -1])  # 마지막 시점의 상태만으로 답한다
        return (logits, e) if return_inputs else logits
