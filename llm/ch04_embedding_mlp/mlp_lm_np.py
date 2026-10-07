"""4장 NumPy 최소 구현: Bengio 등(2003)의 신경 확률 언어모델(식 1)과 손으로 쓴 역전파.

기호는 원 논문을 따른다. m = 단어 벡터 차원, h = 은닉 유닛 수, k = n-1 = 문맥 단어 수.
    x = [C[w_{t-2}]; C[w_{t-1}]]                 (B, k*m)   공유 임베딩 C를 문맥 자리마다 다시 쓴다
    y = b + x W^T + tanh(d + x H^T) U^T          (B, |V|)   논문 식 (1). 은닉 편향 d는 코드에서 `hb`
    P(w_t | 문맥) = softmax(y)
원 논문은 x를 (C(w_{t-1}), C(w_{t-2}), ...) 순서로 이어 붙였다. 여기서는 오래된 단어부터 붙였는데,
H와 W의 열 순서만 바뀔 뿐 모델은 같다.
"""

import numpy as np

DECAYED = ("C", "H", "U", "W")  # 원 논문처럼 weight decay는 가중치와 C에만, 편향에는 주지 않는다


def init_params(V, m, k, h, rng, direct=True):
    p = {
        "C": rng.standard_normal((V, m)) * 0.1,
        "H": rng.standard_normal((h, k * m)) / np.sqrt(k * m),
        "hb": np.zeros(h),
        "U": rng.standard_normal((V, h)) / np.sqrt(h),
        "b": np.zeros(V),
    }
    p["W"] = rng.standard_normal((V, k * m)) / np.sqrt(k * m) if direct else np.zeros((V, k * m))
    return p


def softmax(y):
    y = y - y.max(axis=-1, keepdims=True)  # 값은 그대로, 지수가 넘치는 것만 막는다
    e = np.exp(y)
    return e / e.sum(axis=-1, keepdims=True)


def forward(params, ctx_ids):
    """ctx_ids: (B, k) 정수. 반환: (확률 (B, |V|), 역전파용 cache)."""
    C = params["C"]
    x = C[ctx_ids].reshape(len(ctx_ids), -1)  # 같은 행 C[w]를 어느 문맥 자리에서든 재사용한다
    z = np.tanh(params["hb"] + x @ params["H"].T)
    y = params["b"] + x @ params["W"].T + z @ params["U"].T
    P = softmax(y)
    return P, {"ctx": ctx_ids, "x": x, "z": z, "P": P}


def loss(params, ctx_ids, targets):
    P, _ = forward(params, ctx_ids)
    with np.errstate(divide="ignore"):  # 확률이 정확히 0이면 inf를 그대로 돌려준다(발산 판정에 쓴다)
        return float(-np.mean(np.log(P[np.arange(len(targets)), targets])))


def backward(params, cache, targets, direct=True):
    """평균 교차엔트로피의 기울기. dl/dy = softmax(y) - onehot(target) 에서 시작해 연쇄법칙으로 C까지."""
    x, z, P, ctx = cache["x"], cache["z"], cache["P"], cache["ctx"]
    B = len(targets)
    dy = P.copy()
    dy[np.arange(B), targets] -= 1.0
    dy /= B
    g = {
        "b": dy.sum(0),
        "U": dy.T @ z,
        "W": dy.T @ x if direct else np.zeros_like(params["W"]),
    }
    da = (dy @ params["U"]) * (1.0 - z**2)  # tanh'(a) = 1 - tanh(a)^2
    g["hb"] = da.sum(0)
    g["H"] = da.T @ x
    dx = da @ params["H"] + (dy @ params["W"] if direct else 0.0)  # (B, k*m)
    m = params["C"].shape[1]
    gC = np.zeros_like(params["C"])
    np.add.at(gC, ctx.reshape(-1), dx.reshape(-1, m))  # 같은 단어가 여러 자리·여러 예제에 나오면 기울기가 더해진다
    g["C"] = gC
    return g


def sgd_step(params, grads, lr, wd=0.0):
    for name in params:
        step = grads[name] + (wd * params[name] if name in DECAYED else 0.0)
        params[name] -= lr * step
    return params


def train(params, ctx, tgt, lr, steps, batch, rng, wd=0.0, direct=True, log_every=0, valid=None):
    """미니배치 SGD. 반환: 기록 리스트 [(step, train_loss_on_batch, valid_loss)]. 발산하면 일찍 멈춘다."""
    log = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(tgt), size=batch)
        P, cache = forward(params, ctx[idx])
        batch_loss = float(-np.mean(np.log(P[np.arange(batch), tgt[idx]] + 1e-300)))
        if not np.isfinite(batch_loss) or batch_loss > 50.0:  # 처음 손실은 log|V| = 3.04 근처. 50을 넘으면 발산으로 본다
            log.append((step, float("nan"), float("nan")))
            break
        sgd_step(params, backward(params, cache, tgt[idx], direct), lr, wd)
        if not all(np.all(np.isfinite(v)) for v in params.values()):
            log.append((step, float("nan"), float("nan")))
            break
        if log_every and (step % log_every == 0 or step == 1):
            log.append((step, batch_loss, loss(params, *valid) if valid is not None else None))
    return log


def n_params(V, m, k, h):
    """원 논문 2절: |V|(1 + n m + h) + h(1 + (n-1) m), 여기서 n = k + 1."""
    return V * (1 + (k + 1) * m + h) + h * (1 + k * m)


def step_flops(V, m, k, h, batch):
    """한 스텝의 행렬곱 FLOP(곱셈·덧셈 각각 1). 순전파 2(hkm + Vh + Vkm), 역전파는 그 두 배."""
    fwd = 2 * (h * k * m + V * h + V * k * m)
    return 3 * fwd * batch


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
