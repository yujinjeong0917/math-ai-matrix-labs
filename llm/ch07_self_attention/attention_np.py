"""7장 NumPy 최소 구현: 순환 신경망(RNN)의 한 걸음과 self-attention 한 층.

원리만 드러나도록 배치 차원 없이 (n, d) 행렬 하나를 다룬다.
- rnn_forward: h_t = tanh(W h_{t-1} + U x_t). 시점 t를 계산하려면 t-1이 끝나야 한다.
- self_attention: softmax(Q K^T / sqrt(d_k) + mask) V. 모든 시점을 행렬곱 한 번에 계산한다.
"""

import numpy as np


def rnn_forward(x, W, U, h0=None):
    """x: (n, d_in), W: (d_h, d_h), U: (d_h, d_in). 반환: 모든 시점의 은닉 상태 (n, d_h)."""
    n = x.shape[0]
    h = np.zeros(W.shape[0]) if h0 is None else h0
    hs = np.empty((n, W.shape[0]))
    for t in range(n):  # 이 루프는 병렬화할 수 없다: h_t가 h_{t-1}을 필요로 한다
        h = np.tanh(W @ h + U @ x[t])
        hs[t] = h
    return hs


def rnn_state_jacobian_norms(x, W, U):
    """||dh_t / dh_0||_2 를 t = 1..n 에 대해 정확히 계산한다.

    dh_t/dh_{t-1} = diag(1 - h_t^2) W 이므로, 거리 t만큼 떨어진 신호의 기울기는
    이 행렬을 t번 곱한 것이다. 곱이 줄어들면 소실, 커지면 폭주한다.
    """
    hs = rnn_forward(x, W, U)
    J = np.eye(W.shape[0])
    norms = []
    for t in range(x.shape[0]):
        J = (np.diag(1.0 - hs[t] ** 2) @ W) @ J
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화: 값은 그대로, overflow만 막는다
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """미래 위치(j > i)를 -inf로 막는 (n, n) 가산 마스크."""
    return np.triu(np.full((n, n), -np.inf), k=1)


def self_attention(x, Wq, Wk, Wv, causal=False, scale=True, return_weights=False):
    """x: (n, d_model), W*: (d_model, d_k). 반환: (n, d_k)."""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T  # (n, n): 모든 위치 쌍을 한 번에 비교한다. 여기서 O(n^2)이 생긴다
    if scale:
        scores = scores / np.sqrt(K.shape[-1])
    if causal:
        scores = scores + causal_mask(x.shape[0])
    A = softmax(scores, axis=-1)
    out = A @ V
    return (out, A) if return_weights else out


def attention_flops(n, d):
    """한 층의 곱셈-덧셈 연산 수(곱셈과 덧셈을 각각 1로 센다)."""
    projections = 3 * 2 * n * d * d  # Q, K, V
    scores = 2 * n * n * d  # Q K^T
    mix = 2 * n * n * d  # A V
    return projections + scores + mix


def rnn_flops(n, d):
    return n * (2 * d * d + 2 * d * d)  # W h + U x, 시점마다


def softmax_saturation(d, n=16, trials=200, scale=False, seed=0):
    """Vaswani 등(2017) 각주 4의 가정(q, k 성분이 평균 0, 분산 1)으로 표본을 뽑아
    점수 분산, softmax 최대 가중치, softmax 야코비안 노름을 잰다."""
    rng = np.random.default_rng(seed)
    var, peak, jac = [], [], []
    for _ in range(trials):
        q = rng.standard_normal(d)
        K = rng.standard_normal((n, d))
        s = K @ q
        if scale:
            s = s / np.sqrt(d)
        a = softmax(s)
        J = np.diag(a) - np.outer(a, a)  # d softmax / d score
        var.append(s.var())
        peak.append(a.max())
        jac.append(np.linalg.norm(J, 2))
    return float(np.mean(var)), float(np.mean(peak)), float(np.mean(jac))
