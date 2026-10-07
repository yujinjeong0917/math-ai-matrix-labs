"""10장 NumPy 최소 구현: 옵티마이저와 학습 루프.

- adam_step:   Kingma·Ba(2014) Algorithm 1. l2 > 0이면 기울기에 l2*θ를 더한다(손실에 L2 벌점을 더한 것과 같다).
- adamw_step:  Loshchilov·Hutter(2019)의 분리형 weight decay. 감쇠를 기울기 통계 밖에서 따로 한다.
               논문 Algorithm 2 12행은 θ ← θ − η_t(α·m̂/(√v̂+ε) + λθ)로 감쇠에 α를 곱하지 않지만,
               여기서는 PyTorch 2.8 AdamW와 같게 θ ← θ − lr·λ·θ (lr을 곱한다)로 쓴다.
- sgd_step:    L2 방식과 분리형 감쇠를 둘 다 지원한다(SGD에서는 둘이 같아지는 것을 테스트로 확인).
- clip_global_norm: 모든 기울기를 한 벡터로 이어 붙인 길이가 c를 넘으면 같은 비율로 줄인다.
                    PyTorch clip_grad_norm_과 같게 c / (norm + 1e-6)을 곱한다.
- 2층 MLP 언어모델(앞 k개 토큰 → 다음 토큰)의 순전파·역전파. NumPy로는 이 크기까지만 학습하고,
  4층 Transformer는 PyTorch(ch10_torch.py)로 학습한다.
"""

import numpy as np

# ---------------------------------------------------------------- 옵티마이저


def adam_init(params):
    return {"t": 0, "m": [np.zeros_like(p) for p in params], "v": [np.zeros_like(p) for p in params]}


def adam_step(state, params, grads, lr, betas=(0.9, 0.999), eps=1e-8, l2=0.0, decay_mask=None):
    """Adam 한 스텝(제자리 갱신). l2는 손실에 (l2/2)·||θ||²를 더한 것과 같다."""
    b1, b2 = betas
    state["t"] += 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        if l2 and (decay_mask is None or decay_mask[i]):
            g = g + l2 * p  # L2: 감쇠 항이 기울기에 섞여 아래의 m, v를 함께 지난다
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        m_hat = state["m"][i] / (1 - b1**t)  # 편향 보정: 0에서 시작한 평균을 키워 준다
        v_hat = state["v"][i] / (1 - b2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def adamw_step(state, params, grads, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.0, decay_mask=None):
    """AdamW 한 스텝. 감쇠 θ ← θ − lr·wd·θ를 먼저 하고, 손실 기울기로만 Adam 갱신을 한다."""
    b1, b2 = betas
    state["t"] += 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        if wd and (decay_mask is None or decay_mask[i]):
            p *= 1 - lr * wd  # 모든 가중치를 같은 비율로 줄인다. m, v에는 들어가지 않는다
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        m_hat = state["m"][i] / (1 - b1**t)
        v_hat = state["v"][i] / (1 - b2**t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def sgd_step(params, grads, lr, l2=0.0, wd=0.0):
    """모멘텀 없는 SGD. l2: 기울기에 l2·θ를 더한다. wd: θ ← θ − wd·θ (논문 식 1의 분리형)."""
    for p, g in zip(params, grads):
        p -= lr * (g + l2 * p) + wd * p


def global_norm(grads):
    return float(np.sqrt(sum(float((g * g).sum()) for g in grads)))


def clip_global_norm(grads, c):
    """방향은 그대로, 전체 길이만 c 이하로. 반환: (새 기울기 목록, 자르기 전 노름)."""
    norm = global_norm(grads)
    coef = min(1.0, c / (norm + 1e-6))
    return [g * coef for g in grads], norm


# ---------------------------------------------------------------- 손실


def cross_entropy_loss(logits, targets):
    """평균 교차 엔트로피와 logits에 대한 기울기. logits: (N, V), targets: (N,)"""
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    N = len(targets)
    loss = -logp[np.arange(N), targets].mean()
    d = np.exp(logp)
    d[np.arange(N), targets] -= 1.0
    return float(loss), d / N


# ---------------------------------------------------------------- 2층 MLP 언어모델

PARAM_NAMES = ("E", "W1", "b1", "W2", "b2")
DECAY_MASK = (True, True, False, True, False)  # 행렬만 감쇠한다(편향은 제외)


def init_mlp(vocab, k, emb, hidden, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((vocab, emb)) * 0.5,
        rng.standard_normal((k * emb, hidden)) / np.sqrt(k * emb),
        np.zeros(hidden),
        rng.standard_normal((hidden, vocab)) / np.sqrt(hidden),
        np.zeros(vocab),
    ]


def ngram_pairs(X, k):
    """길이 L 창들에서 (앞 k개 토큰, 다음 토큰) 쌍을 모두 뽑는다."""
    ctx = np.stack([X[:, t - k : t] for t in range(k, X.shape[1])], 1).reshape(-1, k)
    y = X[:, k:].reshape(-1)
    return ctx, y


def mlp_forward(params, ctx):
    E, W1, b1, W2, b2 = params
    e = E[ctx].reshape(len(ctx), -1)  # (N, k*emb): 앞 k개 토큰의 임베딩을 이어 붙인다
    h = np.tanh(e @ W1 + b1)
    return e, h, h @ W2 + b2


def mlp_loss_and_grads(params, ctx, y):
    E, W1, b1, W2, b2 = params
    e, h, logits = mlp_forward(params, ctx)
    loss, dlogits = cross_entropy_loss(logits, y)
    dW2, db2 = h.T @ dlogits, dlogits.sum(0)
    dpre = (dlogits @ W2.T) * (1 - h * h)  # tanh' = 1 − tanh²
    dW1, db1 = e.T @ dpre, dpre.sum(0)
    de = (dpre @ W1.T).reshape(len(ctx), ctx.shape[1], -1)
    dE = np.zeros_like(E)
    np.add.at(dE, ctx, de)  # 같은 토큰이 여러 번 나오면 기울기를 더한다
    return loss, [dE, dW1, db1, dW2, db2]


def estimate_loss(params, ctx, y):
    return cross_entropy_loss(mlp_forward(params, ctx)[2], y)[0]


def train_loop(params, ctx, y, steps, batch, lr, opt="adamw", wd=0.0, clip=None, seed=0):
    """고정 학습 쌍에서 미니배치를 뽑아 학습한다. 반환: 스텝별 학습 손실과 기울기 노름."""
    rng = np.random.default_rng(seed)
    state = adam_init(params)
    losses, norms = [], []
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        loss, grads = mlp_loss_and_grads(params, ctx[idx], y[idx])
        if clip is not None:
            grads, n = clip_global_norm(grads, clip)
        else:
            n = global_norm(grads)
        if opt == "adamw":
            adamw_step(state, params, grads, lr, wd=wd, decay_mask=DECAY_MASK)
        else:
            adam_step(state, params, grads, lr, l2=wd, decay_mask=DECAY_MASK)
        losses.append(loss)
        norms.append(n)
    return losses, norms
