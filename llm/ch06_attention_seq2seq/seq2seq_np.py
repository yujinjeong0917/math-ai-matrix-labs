"""6장 NumPy 최소 구현: 인코더-디코더에 붙이는 어텐션.

원리만 드러나도록 배치 차원 없이 다룬다.
- 인코더 상태 H: (L, d). 디코더 상태 s: (d,).
- 점수 함수 셋: 덧셈형(Bahdanau 등 2014, 부록 A.1.2), 곱셈형 dot·general(Luong 등 2015, 3.1절).
- attend: 점수를 softmax로 비율 alpha로 바꾸고, 인코더 상태를 그 비율로 섞어 문맥 c를 만든다.
- seq2seq_attention_step: 디코더 한 걸음. GRU 셀로 s_t를 만든 뒤 s_t로 점수를 매긴다(Luong 흐름).
  Bahdanau 등은 직전 상태 s_{t-1}로 점수를 매겼다. 이 장은 점수 함수만 바꿔 비교하려고 흐름을 하나로 고정했다.
"""

import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # 값은 그대로, 지수가 넘치는 것만 막는다 (7장과 같은 줄)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------- 점수 함수: 디코더 상태 s 하나와 인코더 상태 L개를 비교해 점수 L개를 낸다 ----------

def additive_score(s, H, Ws, Wh, v, HWh=None):
    """e_i = v^T tanh(Ws s + Wh h_i). s: (d,), H: (L, d), Ws, Wh: (d_a, d), v: (d_a,). 반환 (L,).

    H @ Wh.T 는 디코더 상태 s와 무관하니 문장마다 한 번만 계산해 HWh로 넘길 수 있다(Bahdanau 등 부록 A.1.2).
    """
    HWh = H @ Wh.T if HWh is None else HWh
    return np.tanh(HWh + Ws @ s) @ v


def dot_score(s, H):
    """e_i = s^T h_i. 학습할 가중치가 없다."""
    return H @ s


def general_score(s, H, W):
    """e_i = s^T W h_i. W: (d, d)."""
    return H @ (W.T @ s)


def attend(scores, H):
    """alpha = softmax(scores), c = sum_i alpha_i h_i. 반환 (c, alpha)."""
    alpha = softmax(scores)
    return alpha @ H, alpha


# ---------- 순환 부품: PyTorch nn.GRUCell과 같은 식, 같은 게이트 순서(r, z, n) ----------

def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """W_ih: (3d, d_in), W_hh: (3d, d). PyTorch 문서의 GRUCell 식 그대로."""
    d = h.shape[0]
    gi, gh = W_ih @ x + b_ih, W_hh @ h + b_hh
    r = sigmoid(gi[:d] + gh[:d])
    z = sigmoid(gi[d : 2 * d] + gh[d : 2 * d])
    n = np.tanh(gi[2 * d :] + r * gh[2 * d :])
    return (1.0 - z) * n + z * h


def encode(xs, gru):
    """xs: (L, d_in). 반환: 모든 시점의 인코더 상태 H (L, d). gru = (W_ih, W_hh, b_ih, b_hh)."""
    d = gru[1].shape[1]
    h = np.zeros(d)
    H = np.empty((xs.shape[0], d))
    for i in range(xs.shape[0]):  # 순차 루프: h_i는 h_{i-1}을 기다린다
        h = gru_cell(xs[i], h, *gru)
        H[i] = h
    return H


def seq2seq_attention_step(y_prev_emb, s_prev, H, gru, Wc, Wo, score="dot", score_params=()):
    """디코더 한 걸음.

    s_t = GRU(y_{t-1}, s_{t-1})
    c_t = sum_i alpha_ti h_i (score="fixed"이면 c_t = h_L, 어텐션 없음)
    logits = Wo tanh(Wc [c_t; s_t])
    반환: (logits, s_t, alpha_t). 고정 문맥이면 alpha_t는 None.
    """
    s = gru_cell(y_prev_emb, s_prev, *gru)
    if score == "fixed":
        c, alpha = H[-1], None
    else:
        fn = {"additive": additive_score, "dot": dot_score, "general": general_score}[score]
        c, alpha = attend(fn(s, H, *score_params), H)
    logits = Wo @ np.tanh(Wc @ np.concatenate([c, s]))
    return logits, s, alpha


def score_flops(L, d, d_a=None, kind="dot"):
    """디코딩 한 걸음에서 점수 L개를 내는 곱셈·덧셈 수(곱 1회 + 덧셈 1회 = 2로 센다).

    dot: 내적 L번 = 2Ld. general: W^T s 한 번(2d^2) + 내적 L번(2Ld).
    additive: Ws s 한 번(2 d_a d) + tanh 안 덧셈 L d_a + v 내적 L번(2 L d_a). Wh h_i 는 미리 계산(2 L d_a d, 문장당 한 번).
    """
    d_a = d if d_a is None else d_a
    if kind == "dot":
        return 2 * L * d
    if kind == "general":
        return 2 * d * d + 2 * L * d
    if kind == "additive":
        return 2 * d_a * d + L * d_a + 2 * L * d_a
    raise ValueError(kind)
