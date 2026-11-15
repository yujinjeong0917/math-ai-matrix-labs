"""7장 NumPy 최소 구현: 순환 신경망(RNN)의 한 걸음과 self-attention 한 층.

원리만 드러나도록 배치 차원 없이 (n, d) 행렬 하나를 다룬다.
- rnn_forward: h_t = tanh(W h_{t-1} + U x_t). 시점 t를 계산하려면 t-1이 끝나야 한다.
- self_attention: softmax(Q K^T / sqrt(d_k) + mask) V. 모든 시점을 행렬곱 한 번에 계산한다.

입력: 토큰 임베딩 행렬 x (n, d_model)과 가중치 행렬들. 출력: 은닉 상태 (n, d_h) 또는 어텐션 출력 (n, d_k)과 가중치 A (n, n).
실험 E1~E3(기울기 거리, 계산량, sqrt(d_k) 스케일)이 이 파일의 함수를 호출한다. 장 본문 숫자와의 대응은 experiments.py 머리말 참고.
기호: n = 시퀀스 길이, d = 모델 차원, d_k = 질의·키 차원(여기서는 d와 같게 쓴다).
"""

import numpy as np


def rnn_forward(x, W, U, h0=None):
    """x: (n, d_in), W: (d_h, d_h), U: (d_h, d_in). 반환: 모든 시점의 은닉 상태 (n, d_h)."""
    n = x.shape[0]
    h = np.zeros(W.shape[0]) if h0 is None else h0
    hs = np.empty((n, W.shape[0]))
    for t in range(n):  # 이 루프는 병렬화할 수 없다: h_t가 h_{t-1}을 필요로 한다
        # 식: h_t = tanh(W h_{t-1} + U x_t).  W @ h: (d_h,), U @ x[t]: (d_h,)
        h = np.tanh(W @ h + U @ x[t])
        hs[t] = h
    return hs


def rnn_state_jacobian_norms(x, W, U):
    """||dh_t / dh_0||_2 를 t = 1..n 에 대해 정확히 계산한다.

    dh_t/dh_{t-1} = diag(1 - h_t^2) W 이므로, 거리 t만큼 떨어진 신호의 기울기는
    이 행렬을 t번 곱한 것이다. 곱이 줄어들면 소실, 커지면 폭주한다.
    """
    hs = rnn_forward(x, W, U)
    # J는 dh_t/dh_0. 시작값은 단위행렬 (d_h, d_h)
    J = np.eye(W.shape[0])
    norms = []
    for t in range(x.shape[0]):
        # 연쇄법칙: J_t = diag(1 - h_t^2) W J_{t-1}.  tanh'(a) = 1 - tanh(a)^2 이고 h_t = tanh(a)이므로 hs[t]로 쓴다
        J = (np.diag(1.0 - hs[t] ** 2) @ W) @ J
        # ord=2: 행렬 스펙트럼 노름(최대 특이값)
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)


def softmax(z, axis=-1):
    # 식: softmax(z)_i = exp(z_i) / sum_j exp(z_j). 최댓값을 빼도 분자·분모의 e^{-max}가 약분되어 결과는 같다
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화: 값은 그대로, overflow만 막는다
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """미래 위치(j > i)를 -inf로 막는 (n, n) 가산 마스크."""
    return np.triu(np.full((n, n), -np.inf), k=1)


def self_attention(x, Wq, Wk, Wv, causal=False, scale=True, return_weights=False):
    """x: (n, d_model), W*: (d_model, d_k). 반환: (n, d_k)."""
    # 식: Q = X W_Q, K = X W_K, V = X W_V.  각각 (n, d_k)
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T  # (n, n): 모든 위치 쌍을 한 번에 비교한다. 여기서 O(n^2)이 생긴다
    if scale:
        # 식: Q K^T / sqrt(d_k). 점수 분산이 d_k에 비례해 커지는 것을 보정한다(softmax_saturation 참고)
        scores = scores / np.sqrt(K.shape[-1])
    if causal:
        # 가산 마스크: 미래 위치에 -inf를 더하면 softmax 뒤 가중치가 정확히 0
        scores = scores + causal_mask(x.shape[0])
    # A = softmax(scores): 행(질의 위치)마다 합이 1인 (n, n) 가중치. axis=-1이 키 방향
    A = softmax(scores, axis=-1)
    # 출력 = A V: 값 벡터의 가중 평균, (n, n) @ (n, d_k) -> (n, d_k)
    out = A @ V
    return (out, A) if return_weights else out


def attention_flops(n, d):
    """한 층의 곱셈-덧셈 연산 수(곱셈과 덧셈을 각각 1로 센다)."""
    # Q, K, V 투영 3개: 각각 (n, d) @ (d, d)가 2*n*d*d. d_k = d로 가정
    projections = 3 * 2 * n * d * d  # Q, K, V
    scores = 2 * n * n * d  # Q K^T
    mix = 2 * n * n * d  # A V
    return projections + scores + mix


def rnn_flops(n, d):
    # 한 시점당 W h(2d^2)와 U x(2d^2). d_h = d_in = d 가정, 활성함수·덧셈 비용은 무시
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
        # 점수가 클수록 softmax가 한 칸에 쏠려(peak -> 1) 야코비안이 0에 가까워진다: 기울기 소실의 원인
        a = softmax(s)
        J = np.diag(a) - np.outer(a, a)  # d softmax / d score
        var.append(s.var())
        peak.append(a.max())
        jac.append(np.linalg.norm(J, 2))
    return float(np.mean(var)), float(np.mean(peak)), float(np.mean(jac))
