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
    """수치 안정 softmax. 지정한 축(기본 마지막)에서 합이 1이 되게 한다."""
    z = z - z.max(axis=axis, keepdims=True)  # 값은 그대로, 지수가 넘치는 것만 막는다 (7장과 같은 줄)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    """1/(1+e^-z). GRU의 reset·update 게이트에 쓴다."""
    return 1.0 / (1.0 + np.exp(-z))


# ---------- 점수 함수: 디코더 상태 s 하나와 인코더 상태 L개를 비교해 점수 L개를 낸다 ----------

def additive_score(s, H, Ws, Wh, v, HWh=None):
    """e_i = v^T tanh(Ws s + Wh h_i). s: (d,), H: (L, d), Ws, Wh: (d_a, d), v: (d_a,). 반환 (L,).

    H @ Wh.T 는 디코더 상태 s와 무관하니 문장마다 한 번만 계산해 HWh로 넘길 수 있다(Bahdanau 등 부록 A.1.2).
    """
    # (L, d) @ (d, d_a) -> (L, d_a). 인코더 쪽 항은 디코더 걸음마다 같아서 미리 계산해 넘길 수 있다.
    HWh = H @ Wh.T if HWh is None else HWh
    # (L, d_a) + (d_a,) 브로드캐스트 -> tanh -> v와 내적 -> 점수 (L,). 식 e_i = v^T tanh(Ws s + Wh h_i).
    return np.tanh(HWh + Ws @ s) @ v


def dot_score(s, H):
    """e_i = s^T h_i. 학습할 가중치가 없다."""
    # (L, d) @ (d,) -> (L,). 곱셈형 dot. s와 h의 크기가 그대로 점수에 들어간다(1/sqrt(d) 스케일 없음).
    return H @ s


def general_score(s, H, W):
    """e_i = s^T W h_i. W: (d, d)."""
    # s^T W h_i = (W^T s)^T h_i 이므로 W^T s를 먼저 계산해 H에 곱한다.
    return H @ (W.T @ s)


def attend(scores, H):
    """alpha = softmax(scores), c = sum_i alpha_i h_i. 반환 (c, alpha)."""
    # alpha_i = exp(e_i) / sum_j exp(e_j). 합이 1이고 모두 양수.
    alpha = softmax(scores)
    # c = sum_i alpha_i h_i: (L,) @ (L, d) -> (d,). 인코더 상태의 가중평균.
    return alpha @ H, alpha


# ---------- 순환 부품: PyTorch nn.GRUCell과 같은 식, 같은 게이트 순서(r, z, n) ----------

def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """W_ih: (3d, d_in), W_hh: (3d, d). PyTorch 문서의 GRUCell 식 그대로."""
    d = h.shape[0]
    # 입력 쪽과 은닉 쪽 선형변환을 따로 둔다(PyTorch 구현과 같은 형태). 각각 (3d,)이고 d씩 (r, z, n) 순서.
    gi, gh = W_ih @ x + b_ih, W_hh @ h + b_hh
    # reset 게이트 r. 이전 상태를 얼마나 쓸지.
    r = sigmoid(gi[:d] + gh[:d])
    # update 게이트 z. 이전 상태를 얼마나 유지할지.
    z = sigmoid(gi[d : 2 * d] + gh[d : 2 * d])
    # 후보 상태 n. r이 은닉 쪽 항(편향 포함)에만 곱해지는 것이 PyTorch GRUCell 식(원 논문 변형과 다를 수 있음).
    n = np.tanh(gi[2 * d :] + r * gh[2 * d :])
    # h' = (1 - z) * n + z * h. z가 클수록 이전 상태를 유지.
    return (1.0 - z) * n + z * h


def encode(xs, gru):
    """xs: (L, d_in). 반환: 모든 시점의 인코더 상태 H (L, d). gru = (W_ih, W_hh, b_ih, b_hh)."""
    # gru = (W_ih, W_hh, b_ih, b_hh). W_hh: (3d, d)의 열 수가 d.
    d = gru[1].shape[1]
    h = np.zeros(d)
    # H[i] = i번째 입력까지 읽은 인코더 상태. 모양 (L, d).
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
    # Luong 흐름: 새 디코더 상태 s_t를 먼저 만들고, s_t로 점수를 매긴다.
    s = gru_cell(y_prev_emb, s_prev, *gru)
    if score == "fixed":
        # 어텐션 없음 기준선: 인코더 마지막 상태 하나가 문맥.
        c, alpha = H[-1], None
    else:
        fn = {"additive": additive_score, "dot": dot_score, "general": general_score}[score]
        c, alpha = attend(fn(s, H, *score_params), H)
    # [c; s]: (2d,) -> Wc (d, 2d) -> tanh -> Wo (V, d) -> 로짓 (V,).
    logits = Wo @ np.tanh(Wc @ np.concatenate([c, s]))
    return logits, s, alpha


def score_flops(L, d, d_a=None, kind="dot"):
    """디코딩 한 걸음에서 점수 L개를 내는 곱셈·덧셈 수(곱 1회 + 덧셈 1회 = 2로 센다).

    dot: 내적 L번 = 2Ld. general: W^T s 한 번(2d^2) + 내적 L번(2Ld).
    additive: Ws s 한 번(2 d_a d) + tanh 안 덧셈 L d_a + v 내적 L번(2 L d_a). Wh h_i 는 미리 계산(2 L d_a d, 문장당 한 번).
    """
    d_a = d if d_a is None else d_a
    # 점수 계산만의 비용(문맥 합 alpha @ H 등은 제외).
    if kind == "dot":
        return 2 * L * d
    if kind == "general":
        return 2 * d * d + 2 * L * d
    if kind == "additive":
        return 2 * d_a * d + L * d_a + 2 * L * d_a
    raise ValueError(kind)
