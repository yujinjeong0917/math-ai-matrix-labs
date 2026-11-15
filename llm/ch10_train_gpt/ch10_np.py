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

입력: 파라미터 리스트(numpy 배열)와 기울기. 출력: 파라미터가 제자리에서 갱신되고, 학습 손실·기울기 노름 목록이 반환된다.
기호: m, v = 기울기의 1·2차 이동평균, t = 스텝 번호, lr = 학습률, wd(또는 l2) = 감쇠 계수, N = 미니배치 크기, V = 어휘 크기, k = 문맥 토큰 수.
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
    # p는 배열이라 `p -=` 가 제자리 갱신이다(params 리스트의 원소가 직접 바뀐다)
    for i, (p, g) in enumerate(zip(params, grads)):
        if l2 and (decay_mask is None or decay_mask[i]):
            g = g + l2 * p  # L2: 감쇠 항이 기울기에 섞여 아래의 m, v를 함께 지난다
        # 식: m_t = beta1 m_{t-1} + (1 - beta1) g_t
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        # 식: v_t = beta2 v_{t-1} + (1 - beta2) g_t^2 (원소별 제곱)
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        # 식: m_hat = m_t / (1 - beta1^t). 0에서 시작한 평균은 초반에 작게 치우친다
        m_hat = state["m"][i] / (1 - b1**t)  # 편향 보정: 0에서 시작한 평균을 키워 준다
        v_hat = state["v"][i] / (1 - b2**t)
        # 식: theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
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
        # l2 항은 기울기 안에서 lr이 곱해지고, wd 항은 lr 없이 p에 직접 곱한다(위 docstring의 정의)
        p -= lr * (g + l2 * p) + wd * p


def global_norm(grads):
    # 모든 배열의 제곱합을 합친 L2 노름(하나의 긴 벡터로 본 길이)
    return float(np.sqrt(sum(float((g * g).sum()) for g in grads)))


def clip_global_norm(grads, c):
    """방향은 그대로, 전체 길이만 c 이하로. 반환: (새 기울기 목록, 자르기 전 노름)."""
    norm = global_norm(grads)
    # 노름이 c 이하면 1(그대로), 넘으면 c/norm으로 축소. 1e-6은 0 나누기 방지(PyTorch와 같은 값)
    coef = min(1.0, c / (norm + 1e-6))
    return [g * coef for g in grads], norm


# ---------------------------------------------------------------- 손실


def cross_entropy_loss(logits, targets):
    """평균 교차 엔트로피와 logits에 대한 기울기. logits: (N, V), targets: (N,)"""
    # log-sum-exp 안정화: 행마다 최댓값을 빼도 log softmax는 같다
    z = logits - logits.max(-1, keepdims=True)
    # 식: log p = z - log sum_j exp(z_j). (N, V)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    N = len(targets)
    # 식: L = -(1/N) sum_i log p_i[y_i]. 정답 칸만 골라 평균
    loss = -logp[np.arange(N), targets].mean()
    # 식: dL/dz = (softmax(z) - onehot(y)) / N. 먼저 softmax를 만들고 정답 칸에서 1을 뺀다
    d = np.exp(logp)
    d[np.arange(N), targets] -= 1.0
    return float(loss), d / N


# ---------------------------------------------------------------- 2층 MLP 언어모델

PARAM_NAMES = ("E", "W1", "b1", "W2", "b2")
# 순서는 PARAM_NAMES: E(임베딩), W1, b1, W2, b2. 편향(b1, b2)은 감쇠 제외
DECAY_MASK = (True, True, False, True, False)  # 행렬만 감쇠한다(편향은 제외)


def init_mlp(vocab, k, emb, hidden, seed=0):
    rng = np.random.default_rng(seed)
    return [
        # 임베딩 (V, emb): 표준편차 0.5(모형의 선택)
        rng.standard_normal((vocab, emb)) * 0.5,
        # W1 (k*emb, hidden): 1/sqrt(입력 수) 스케일
        rng.standard_normal((k * emb, hidden)) / np.sqrt(k * emb),
        np.zeros(hidden),
        rng.standard_normal((hidden, vocab)) / np.sqrt(hidden),
        np.zeros(vocab),
    ]


def ngram_pairs(X, k):
    """길이 L 창들에서 (앞 k개 토큰, 다음 토큰) 쌍을 모두 뽑는다."""
    # 위치 t마다 앞 k개 [t-k, t)를 잡는다. stack 후 (창 수, 위치 수, k) -> (N, k). 행 순서는 창 우선
    ctx = np.stack([X[:, t - k : t] for t in range(k, X.shape[1])], 1).reshape(-1, k)
    # 정답: 위 ctx의 행 순서와 같게 창 우선으로 펼친다 (N,)
    y = X[:, k:].reshape(-1)
    return ctx, y


def mlp_forward(params, ctx):
    E, W1, b1, W2, b2 = params
    # E[ctx]: (N, k, emb) -> (N, k*emb)
    e = E[ctx].reshape(len(ctx), -1)  # (N, k*emb): 앞 k개 토큰의 임베딩을 이어 붙인다
    # 은닉층 (N, hidden)
    h = np.tanh(e @ W1 + b1)
    # 로짓 (N, V). 역전파에서 e, h를 재사용하려고 함께 반환
    return e, h, h @ W2 + b2


def mlp_loss_and_grads(params, ctx, y):
    E, W1, b1, W2, b2 = params
    e, h, logits = mlp_forward(params, ctx)
    loss, dlogits = cross_entropy_loss(logits, y)
    # dW2 = h^T dlogits (hidden, V), db2는 행 합
    dW2, db2 = h.T @ dlogits, dlogits.sum(0)
    # 역전파: 로짓 -> 은닉 -> tanh 앞. tanh'(a) = 1 - tanh(a)^2 = 1 - h^2
    dpre = (dlogits @ W2.T) * (1 - h * h)  # tanh' = 1 − tanh²
    dW1, db1 = e.T @ dpre, dpre.sum(0)
    # (N, k*emb)를 (N, k, emb)로 되돌려 각 문맥 위치의 임베딩 기울기로 본다
    de = (dpre @ W1.T).reshape(len(ctx), ctx.shape[1], -1)
    dE = np.zeros_like(E)
    # E[ctx] 인덱싱은 같은 행을 여러 번 가져올 수 있다. dE[ctx] += de는 중복을 한 번만 더하므로 add.at이 필요(함정)
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
        # 복원 추출 미니배치: 같은 쌍이 한 배치에 두 번 나올 수 있다
        idx = rng.integers(0, len(y), batch)
        loss, grads = mlp_loss_and_grads(params, ctx[idx], y[idx])
        # 클리핑은 옵티마이저 갱신 직전에 한다. n은 자르기 전 노름
        if clip is not None:
            grads, n = clip_global_norm(grads, clip)
        else:
            n = global_norm(grads)
        if opt == "adamw":
            adamw_step(state, params, grads, lr, wd=wd, decay_mask=DECAY_MASK)
        else:
            # opt가 adamw가 아니면 Adam + L2 (wd를 l2로 전달)
            adam_step(state, params, grads, lr, l2=wd, decay_mask=DECAY_MASK)
        losses.append(loss)
        norms.append(n)
    return losses, norms
