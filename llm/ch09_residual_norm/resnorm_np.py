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
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)  # 편향 분산(1/d)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta


def layer_norm_backward(dy, x, gamma, eps=1e-5):
    """LN의 역전파. 반환: (dx, dgamma, dbeta).

    xhat = (x - mu) / s 이고 mu, s가 모두 x에 기대므로, 그냥 1/s를 곱하면 틀린다.
    dx = (g - mean(g) - xhat * mean(g * xhat)) / s,  g = dy * gamma
    """
    mu = x.mean(-1, keepdims=True)
    s = np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mu) / s
    g = dy * gamma
    dx = (g - g.mean(-1, keepdims=True) - xhat * (g * xhat).mean(-1, keepdims=True)) / s
    red = tuple(range(dy.ndim - 1))
    return dx, (dy * xhat).sum(red), dy.sum(red)


def batch_norm(x, gamma, beta, eps=1e-5):
    """학습 모드 BN. x: (batch, d). 같은 특징을 배치 축으로 모아 정규화한다."""
    mu = x.mean(0, keepdims=True)
    var = x.var(0, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ---------------------------------------------------------------- 부분층


def multi_head_attention(x, Wq, Wk, Wv, Wo, h, causal=True):
    """concat(head_1..head_h) W_O. 헤드마다 d_k = d / h 차원을 쓴다. W*: (d, d)."""
    n, d = x.shape
    dk = d // h
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    heads = []
    for i in range(h):
        sl = slice(i * dk, (i + 1) * dk)
        scores = Q[:, sl] @ K[:, sl].T / np.sqrt(dk)
        if causal:
            scores = scores + causal_mask(n)
        heads.append(softmax(scores) @ V[:, sl])
    return np.concatenate(heads, axis=-1) @ Wo


def mlp(x, W1, b1, W2, b2):
    return np.maximum(0.0, x @ W1 + b1) @ W2 + b2


# ---------------------------------------------------------------- 블록

ARCHS = ("plain", "norm", "residual", "post", "pre")


def sublayer(x, f, arch, gamma, beta):
    """부분층 하나에 잔차·정규화 규칙을 적용한다. f는 x -> F(x)."""
    if arch == "plain":
        return f(x)
    if arch == "norm":
        return layer_norm(f(x), gamma, beta)
    if arch == "residual":
        return x + f(x)
    if arch == "post":
        return layer_norm(x + f(x), gamma, beta)
    if arch == "pre":
        return x + f(layer_norm(x, gamma, beta))
    raise ValueError(arch)


def block(x, p, arch, h):
    """p: dict(Wq, Wk, Wv, Wo, W1, b1, W2, b2, g1, be1, g2, be2)."""
    x = sublayer(x, lambda z: multi_head_attention(z, p["Wq"], p["Wk"], p["Wv"], p["Wo"], h), arch, p["g1"], p["be1"])
    x = sublayer(x, lambda z: mlp(z, p["W1"], p["b1"], p["W2"], p["b2"]), arch, p["g2"], p["be2"])
    return x


# ---------------------------------------------------------------- 계산량


def block_flops(n, d, h=1, ff=4):
    """블록 하나의 곱셈·덧셈 수(곱셈과 덧셈을 각각 1로 센다). h는 총량을 바꾸지 않는다."""
    attn = 4 * 2 * n * d * d + 2 * 2 * n * n * d  # Q,K,V,O 투영 + QK^T, AV
    ffn = 2 * 2 * n * d * (ff * d)
    return attn + ffn


def residual_flops(n, d):
    return 2 * n * d  # 부분층 2개, 각각 덧셈 n*d번


def layer_norm_flops(n, d):
    """평균(d), 빼기(d), 제곱·평균(2d), 나누기(d), gamma·beta(2d): 약 7d. 부분층 2개."""
    return 2 * 7 * n * d
