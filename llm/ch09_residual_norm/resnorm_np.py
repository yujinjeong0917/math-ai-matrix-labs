"""9장 NumPy 최소 구현: 정규화(LayerNorm, BatchNorm), 멀티헤드 어텐션, MLP, 그리고 블록 하나.

원리만 드러나도록 배치 차원 없이 (n, d) 행렬 하나를 다룬다(BatchNorm만 (batch, d)).
- layer_norm: 한 위치의 특징 d개로 평균·분산을 구한다. 배치와 무관하다.
- batch_norm: 같은 특징을 배치 안의 표본끼리 모아 평균·분산을 구한다. 배치에 따라 출력이 바뀐다.
- block: 배치 방식 4가지
    "plain"    x_{l+1} = F(x_l)                  (잔차·정규화 없음)
    "norm"     x_{l+1} = LN(F(x_l))              (정규화만, 잔차 없음: He 등의 BN 평범한 망에 대응)
    "residual" x_{l+1} = x_l + F(x_l)            (잔차만)
    "post"     x_{l+1} = LN(x_l + F(x_l))        (Post-LN, Vaswani 등 2017)
    "pre"      x_{l+1} = x_l + F(LN(x_l))        (Pre-LN, 마지막 출력 앞에 LN을 하나 더 둔다)
  F는 어텐션 부분층과 MLP 부분층 두 개이고, 위 규칙을 부분층마다 적용한다.

입력: x (n, d)와 파라미터 dict p. 출력: 같은 모양 (n, d)의 블록 출력.
비교 기준: arch 5가지의 활성값·기울기 크기(experiments.py E1~E3). 기호: n = 시퀀스 길이, d = 모델 차원, h = 헤드 수, ff = MLP 확장 배수.
"""

import numpy as np

# ---------------------------------------------------------------- 7장에서 복사한 부분
# 출처: llm/ch07_self_attention/attention_np.py (장끼리 import하지 않고 복사한다)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    return np.triu(np.full((n, n), -np.inf), k=1)


def self_attention(x, Wq, Wk, Wv, causal=False, scale=True):
    """7장의 헤드 하나짜리 self-attention. x: (n, d_model), W*: (d_model, d_k)."""
    # 식: Q = X W_Q, K = X W_K, V = X W_V, 각각 (n, d_k)
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(K.shape[-1])
    if causal:
        scores = scores + causal_mask(x.shape[0])
    return softmax(scores, axis=-1) @ V


# ---------------------------------------------------------------- 정규화


def layer_norm(x, gamma, beta, eps=1e-5):
    """특징 축(마지막 축)으로 정규화한다. nn.LayerNorm과 같게 eps를 제곱근 안에 둔다."""
    # 식: mu = (1/d) sum_i x_i. 마지막 축(특징 d개)에 대해 위치마다 따로 구한다
    mu = x.mean(-1, keepdims=True)
    # 식: sigma^2 = (1/d) sum (x_i - mu)^2. np.var 기본값 ddof=0이 편향 분산(1/d)
    var = x.var(-1, keepdims=True)  # 편향 분산(1/d)
    # 식: xhat = (x - mu) / sqrt(sigma^2 + eps).  eps는 분산 0일 때의 0 나누기 방지
    xhat = (x - mu) / np.sqrt(var + eps)
    # 식: y = gamma * xhat + beta. gamma, beta: (d,), 특징마다 학습되는 크기·이동
    return gamma * xhat + beta


def layer_norm_backward(dy, x, gamma, eps=1e-5):
    """LN의 역전파. 반환: (dx, dgamma, dbeta).

    xhat = (x - mu) / s 이고 mu, s가 모두 x에 기대므로, 그냥 1/s를 곱하면 틀린다.
    dx = (g - mean(g) - xhat * mean(g * xhat)) / s,  g = dy * gamma
    """
    mu = x.mean(-1, keepdims=True)
    # s = sqrt(sigma^2 + eps), 위치마다 (n, 1)
    s = np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) / s
    g = dy * gamma
    dx = (g - g.mean(-1, keepdims=True) - xhat * (g * xhat).mean(-1, keepdims=True)) / s
    # gamma, beta의 기울기는 위치(앞 축)들을 모두 합한다. dy.ndim - 1개의 앞 축
    red = tuple(range(dy.ndim - 1))
    # dgamma = sum(dy * xhat), dbeta = sum(dy)
    return dx, (dy * xhat).sum(red), dy.sum(red)


def batch_norm(x, gamma, beta, eps=1e-5):
    """학습 모드 BN. x: (batch, d). 같은 특징을 배치 축으로 모아 정규화한다."""
    # BatchNorm: 축 0(배치)으로 평균. LN과 달리 같은 특징을 다른 표본끼리 모은다
    mu = x.mean(0, keepdims=True)
    var = x.var(0, keepdims=True)
    # 학습 모드만 구현. 추론 시 쓰는 이동평균 통계(running mean/var)는 여기 없다(모형의 단순화)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ---------------------------------------------------------------- 부분층


def multi_head_attention(x, Wq, Wk, Wv, Wo, h, causal=True):
    """concat(head_1..head_h) W_O. 헤드마다 d_k = d / h 차원을 쓴다. W*: (d, d)."""
    n, d = x.shape
    # d가 h의 배수라는 가정. 헤드별로 열을 dk개씩 나눈다
    dk = d // h
    # Q, K, V: (n, d). 아래에서 열 구간 sl로 헤드 i를 꺼낸다
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    heads = []
    for i in range(h):
        sl = slice(i * dk, (i + 1) * dk)
        # 헤드 i의 점수 (n, n): Q_i K_i^T / sqrt(d_k)
        scores = Q[:, sl] @ K[:, sl].T / np.sqrt(dk)
        if causal:
            scores = scores + causal_mask(n)
        # (n, n) @ (n, dk) -> (n, dk)
        heads.append(softmax(scores) @ V[:, sl])
    # 식: concat(head_1..head_h) W_O. 이어 붙이면 (n, d), W_O: (d, d)
    return np.concatenate(heads, axis=-1) @ Wo


def mlp(x, W1, b1, W2, b2):
    # 식: MLP(x) = ReLU(x W1 + b1) W2 + b2. (n, d) -> (n, ff*d) -> (n, d)
    return np.maximum(0.0, x @ W1 + b1) @ W2 + b2


# ---------------------------------------------------------------- 블록

ARCHS = ("plain", "norm", "residual", "post", "pre")


def sublayer(x, f, arch, gamma, beta):
    """부분층 하나에 잔차·정규화 규칙을 적용한다. f는 x -> F(x)."""
    if arch == "plain":
        return f(x)
    if arch == "norm":
        # Post-LN이 아님에 주의: 잔차 없이 출력에만 LN
        return layer_norm(f(x), gamma, beta)
    if arch == "residual":
        return x + f(x)
    if arch == "post":
        # Post-LN: 잔차를 더한 뒤 정규화. 잔차 경로에도 LN이 들어간다
        return layer_norm(x + f(x), gamma, beta)
    if arch == "pre":
        # Pre-LN: 부분층 입력만 정규화하고 잔차 경로는 그대로 둔다(항등 경로 유지)
        return x + f(layer_norm(x, gamma, beta))
    raise ValueError(arch)


def block(x, p, arch, h):
    """p: dict(Wq, Wk, Wv, Wo, W1, b1, W2, b2, g1, be1, g2, be2)."""
    # 부분층 1: 어텐션(causal). 정규화 파라미터는 g1, be1
    x = sublayer(x, lambda z: multi_head_attention(z, p["Wq"], p["Wk"], p["Wv"], p["Wo"], h), arch, p["g1"], p["be1"])
    # 부분층 2: MLP. 정규화 파라미터는 g2, be2
    x = sublayer(x, lambda z: mlp(z, p["W1"], p["b1"], p["W2"], p["b2"]), arch, p["g2"], p["be2"])
    return x


# ---------------------------------------------------------------- 계산량


def block_flops(n, d, h=1, ff=4):
    """블록 하나의 곱셈·덧셈 수(곱셈과 덧셈을 각각 1로 센다). h는 총량을 바꾸지 않는다."""
    # 투영 4개(Q, K, V, O) 각각 2nd^2, 점수와 혼합 각각 2n^2 d. 헤드를 나눠도 합은 같다
    attn = 4 * 2 * n * d * d + 2 * 2 * n * n * d  # Q,K,V,O 투영 + QK^T, AV
    # MLP 두 행렬곱: n x d -> ff*d, ff*d -> d. 각 2*n*d*(ff*d). 활성함수·편향 비용은 무시
    ffn = 2 * 2 * n * d * (ff * d)
    return attn + ffn


def residual_flops(n, d):
    # 부분층 2개 * 덧셈 n*d. 곱셈은 없다
    return 2 * n * d  # 부분층 2개, 각각 덧셈 n*d번


def layer_norm_flops(n, d):
    """평균(d), 빼기(d), 제곱·평균(2d), 나누기(d), gamma·beta(2d): 약 7d. 부분층 2개."""
    return 2 * 7 * n * d
