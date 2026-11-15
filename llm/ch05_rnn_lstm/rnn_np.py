"""5장 NumPy 최소 구현: 순환망(RNN)과 LSTM, 손으로 쓴 시간 역전파(BPTT), 기울기 클리핑.

원리만 드러나도록 배치 차원 없이 시퀀스 하나 (n, d_in)를 다룬다.
- rnn_forward / rnn_bptt: h_t = tanh(W h_{t-1} + U x_t + b), 마지막 상태로 분류.
- lstm_cell_forward / lstm_cell_backward: c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t).
  게이트는 이 파일에서 (i, f, o, g) 순서로 쌓는다. PyTorch nn.LSTM은 (i, f, g, o) 순서라
  test_ch05.py에서 순서를 명시적으로 바꿔 맞춘다.
- clip_by_norm (Pascanu 등 2013 Algorithm 1) vs clip_elementwise (같은 논문이 Mikolov 2012의 방법이라고 밝힌 방식).
"""

import numpy as np


# ---------------------------------------------------------------- 순환망
def rnn_forward(x, W, U, b=None, h0=None):
    """x: (n, d_in), W: (H, H), U: (H, d_in). 반환: 모든 시점의 은닉 상태 (n, H)."""
    n, H = x.shape[0], W.shape[0]
    b = np.zeros(H) if b is None else b
    h = np.zeros(H) if h0 is None else h0
    # hs[t] = 시점 t의 은닉 상태. 모양 (n, H).
    hs = np.empty((n, H))
    for t in range(n):  # 이 루프는 병렬화할 수 없다: h_t가 h_{t-1}을 필요로 한다
        # 식: h_t = tanh(W h_{t-1} + U x_t + b). W (H, H), U (H, d_in).
        h = np.tanh(W @ h + U @ x[t] + b)
        hs[t] = h
    return hs


def rnn_state_jacobian_norms(x, W, U):
    """||dh_t / dh_0||_2 를 t = 1..n 에 대해 정확히 계산한다. (7장 attention_np.py에서 복사)

    dh_t/dh_{t-1} = diag(1 - h_t^2) W 이므로, 거리 t만큼 떨어진 신호의 기울기는
    이 행렬을 t번 곱한 것이다. 곱이 줄어들면 소실, 커지면 폭주한다.
    """
    hs = rnn_forward(x, W, U)
    J = np.eye(W.shape[0])
    norms = []
    for t in range(x.shape[0]):
        # J = dh_t/dh_0 를 한 걸음씩 곱해 만든다. tanh' = 1 - h^2.
        J = (np.diag(1.0 - hs[t] ** 2) @ W) @ J
        # ord=2는 행렬의 스펙트럼 노름(최대 특이값).
        norms.append(np.linalg.norm(J, 2))
    return np.array(norms)


def softmax(z):
    """수치 안정 softmax(1차원 로짓 벡터). 최댓값을 빼도 결과는 같고 exp 오버플로만 막는다."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rnn_loss(params, x, y):
    """마지막 상태 h_n 하나로 클래스 y를 맞히는 교차엔트로피."""
    hs = rnn_forward(x, params["W"], params["U"], params["b"])
    # Wo: (클래스 수, H). 마지막 시점 상태 하나만 분류에 쓴다.
    logits = params["Wo"] @ hs[-1] + params["bo"]
    return -np.log(softmax(logits)[y])


def rnn_bptt(params, x, y):
    """시간 역전파. 반환: (손실, 파라미터 기울기 dict, 입력 기울기 dx (n, d_in))."""
    W, U, b, Wo, bo = (params[k] for k in ("W", "U", "b", "Wo", "bo"))
    n = x.shape[0]
    hs = rnn_forward(x, W, U, b)
    p = softmax(Wo @ hs[-1] + bo)
    loss = -np.log(p[y])
    dlogits = p.copy()
    dlogits[y] -= 1.0  # softmax + 교차엔트로피의 기울기: p - e_y
    g = {"Wo": np.outer(dlogits, hs[-1]), "bo": dlogits, "W": np.zeros_like(W), "U": np.zeros_like(U), "b": np.zeros_like(b)}
    dx = np.zeros_like(x)
    # 손실에서 마지막 h_n으로 오는 기울기 (H,).
    dh = Wo.T @ dlogits
    for t in range(n - 1, -1, -1):  # 앞으로 n걸음 왔으니 뒤로도 n걸음 간다
        da = dh * (1.0 - hs[t] ** 2)  # tanh의 미분
        # t = 0이면 이전 상태는 h0 = 0(rnn_forward 기본값과 같다).
        h_prev = hs[t - 1] if t > 0 else np.zeros_like(hs[0])
        # 같은 W를 모든 시점이 공유하므로 시점마다 기울기를 더한다(BPTT의 핵심).
        g["W"] += np.outer(da, h_prev)
        g["U"] += np.outer(da, x[t])
        g["b"] += da
        dx[t] = U.T @ da
        dh = W.T @ da  # 한 걸음 거슬러 갈 때마다 W^T diag(1-h^2)가 한 번씩 곱해진다
    return loss, g, dx


# ---------------------------------------------------------------- LSTM
def sigmoid(z):
    """시그모이드 1/(1+e^-z). 게이트 값을 (0, 1)로 만든다. z가 매우 작으면 exp가 오버플로 경고를 낼 수 있다."""
    return 1.0 / (1.0 + np.exp(-z))


def lstm_cell_forward(x, h_prev, c_prev, Wx, Wh, b):
    """한 시점. Wx: (4H, d_in), Wh: (4H, H), b: (4H,). 게이트 순서 (i, f, o, g)."""
    H = h_prev.shape[0]
    # 네 게이트의 pre-activation을 한 번에: (4H,). 아래에서 H씩 잘라 (i, f, o, g)로 쓴다.
    z = Wx @ x + Wh @ h_prev + b
    i, f, o = sigmoid(z[:H]), sigmoid(z[H : 2 * H]), sigmoid(z[2 * H : 3 * H])
    g = np.tanh(z[3 * H :])
    # 식: c_t = f * c_{t-1} + i * g. 덧셈 경로라 기울기가 f만 곱해져 흐른다.
    c = f * c_prev + i * g  # 덧셈 경로: c_prev는 f를 곱한 채로 그대로 넘어간다
    # 식: h_t = o * tanh(c_t).
    h = o * np.tanh(c)
    return h, c, (x, h_prev, c_prev, i, f, o, g, c)


def lstm_cell_backward(dh, dc, cache, Wx, Wh):
    """한 시점의 역전파. 반환: (dx, dh_prev, dc_prev, dWx, dWh, db)."""
    x, h_prev, c_prev, i, f, o, g, c = cache
    tc = np.tanh(c)
    # h = o * tanh(c)에서 dh -> do. tc = tanh(c).
    do = dh * tc
    # c로 오는 기울기 = 다음 시점에서 온 dc + h를 통해 오는 몫.
    dc = dc + dh * o * (1.0 - tc**2)
    di, df, dg = dc * g, dc * c_prev, dc * i
    dc_prev = dc * f  # 셀 경로의 야코비안은 diag(f) 하나뿐이다
    # 각 게이트의 pre-activation 기울기. 시그모이드' = s(1-s), tanh' = 1 - g^2. 순서 (i, f, o, g).
    dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g**2)])
    return Wx.T @ dz, Wh.T @ dz, dc_prev, np.outer(dz, x), np.outer(dz, h_prev), dz


def lstm_forward(x, Wx, Wh, b, h0=None, c0=None):
    """x: (n, d_in) -> (hs (n, H), cs (n, H), 시점별 cache 리스트). 게이트 순서 (i, f, o, g)."""
    H = Wh.shape[1]
    h = np.zeros(H) if h0 is None else h0
    c = np.zeros(H) if c0 is None else c0
    hs, cs, caches = [], [], []
    for t in range(x.shape[0]):
        h, c, cache = lstm_cell_forward(x[t], h, c, Wx, Wh, b)
        hs.append(h)
        cs.append(c)
        caches.append(cache)
    return np.array(hs), np.array(cs), caches


def lstm_loss(params, x, y):
    """마지막 은닉 상태 h_n으로 클래스 y를 맞히는 교차엔트로피."""
    hs, _, _ = lstm_forward(x, params["Wx"], params["Wh"], params["b"])
    return -np.log(softmax(params["Wo"] @ hs[-1] + params["bo"])[y])


def lstm_bptt(params, x, y):
    """LSTM 시간 역전파. 반환: (손실, 파라미터 기울기 dict, 입력 기울기 dx (n, d_in))."""
    Wx, Wh, b, Wo, bo = (params[k] for k in ("Wx", "Wh", "b", "Wo", "bo"))
    hs, _, caches = lstm_forward(x, Wx, Wh, b)
    p = softmax(Wo @ hs[-1] + bo)
    loss = -np.log(p[y])
    dlogits = p.copy()
    dlogits[y] -= 1.0
    g = {"Wo": np.outer(dlogits, hs[-1]), "bo": dlogits, "Wx": np.zeros_like(Wx), "Wh": np.zeros_like(Wh), "b": np.zeros_like(b)}
    dx = np.zeros_like(x)
    dh, dc = Wo.T @ dlogits, np.zeros(Wh.shape[1])
    for t in range(x.shape[0] - 1, -1, -1):
        dx[t], dh, dc, dWx, dWh, db = lstm_cell_backward(dh, dc, caches[t], Wx, Wh)
        g["Wx"] += dWx
        g["Wh"] += dWh
        g["b"] += db
    return loss, g, dx


def lstm_state_jacobian_norms(x, Wx, Wh, b):
    """LSTM 상태 s_t = (h_t, c_t)에 대해 ||dc_t/dc_0||_2 와 ||ds_t/ds_0||_2 를 t = 1..n 에서 정확히 계산한다.

    셀 경로만 보면 dc_t/dc_{t-1} = diag(f_t)이지만, h를 거쳐 게이트로 돌아오는 경로도 있어 전부 더해 센다.
    """
    H = Wh.shape[1]
    hs, cs, caches = lstm_forward(x, Wx, Wh, b)
    # 상태 s = (h, c) 2H차원에 대한 전체 야코비안. 항등으로 시작해 시점마다 A를 곱한다.
    J = np.eye(2 * H)  # 행/열 순서: (h, c)
    Wi, Wf, Wo_, Wg = Wh[:H], Wh[H : 2 * H], Wh[2 * H : 3 * H], Wh[3 * H :]
    cell_norms, state_norms, f_prod = [], [], np.ones(H)
    for t in range(x.shape[0]):
        _, _, c_prev, i, f, o, g, c = caches[t]
        tc = np.tanh(c)
        # c가 h_{t-1}에 의존하는 경로: f, i, g 게이트를 거친다. 각 항이 (H, H).
        dc_dh = (c_prev * f * (1 - f))[:, None] * Wf + (g * i * (1 - i))[:, None] * Wi + (i * (1 - g**2))[:, None] * Wg
        # 셀 경로 야코비안은 diag(f). f가 1에 가까우면 기울기가 보존된다.
        dc_dc = np.diag(f)
        dh_dh = (tc * o * (1 - o))[:, None] * Wo_ + (o * (1 - tc**2))[:, None] * dc_dh
        dh_dc = (o * (1 - tc**2) * f)[:, None] * np.eye(H)
        # 한 시점 야코비안 d(h_t, c_t) / d(h_{t-1}, c_{t-1}) 을 (2H, 2H) 블록으로 조립.
        A = np.block([[dh_dh, dh_dc], [dc_dh, dc_dc]])
        J = A @ J
        cell_norms.append(np.linalg.norm(J[H:, H:], 2))
        state_norms.append(np.linalg.norm(J, 2))
        # 셀 경로만의 근사(forget 게이트의 누적곱). 비교용 값.
        f_prod = f_prod * f
    return np.array(cell_norms), np.array(state_norms), f_prod


# ---------------------------------------------------------------- 클리핑
def global_norm(grads):
    """모든 기울기 배열을 하나로 이은 벡터의 L2 노름."""
    return float(np.sqrt(sum(float((g**2).sum()) for g in grads)))


def clip_by_norm(grads, c):
    """Pascanu 등(2013) Algorithm 1: 전체 노름이 c를 넘으면 c가 되게 같은 비율로 줄인다. 방향은 그대로."""
    norm = global_norm(grads)
    # 노름이 c 이하면 s = 1로 그대로 둔다. 방향은 보존하고 크기만 줄인다.
    s = min(1.0, c / norm) if norm > 0 else 1.0
    return [g * s for g in grads]


def clip_elementwise(grads, c):
    """원소마다 [-c, c]로 자른다. 큰 성분만 깎이므로 방향이 바뀔 수 있다."""
    return [np.clip(g, -c, c) for g in grads]


# ---------------------------------------------------------------- 계산량
def rnn_flops_per_step(d_in, H):
    """한 시점 FLOP(부동소수점 연산 수)."""
    return 2 * H * H + 2 * H * d_in  # W h + U x (곱셈·덧셈을 각각 1로 센다)


def lstm_flops_per_step(d_in, H):
    """게이트 4개가 각각 행렬곱을 하므로 RNN의 4배."""
    return 4 * (2 * H * H + 2 * H * d_in)  # 게이트 4개가 각자 W h + U x를 한다


def rnn_forward_batched(x, W, U):
    """x: (B, n, d_in). 서로 독립인 B개 시퀀스를 한꺼번에 돈다. 시점 루프(n번)는 그대로 남는다."""
    B, n, _ = x.shape
    h = np.zeros((B, W.shape[0]))
    for t in range(n):
        # (B, H) @ (H, H) -> (B, H). 배치 차원은 행렬곱으로 묶이지만 t 루프는 순차다. b는 쓰지 않는 단순화.
        h = np.tanh(h @ W.T + x[:, t] @ U.T)  # B개는 행렬곱 한 번에 처리된다
    return h


def lstm_forward_batched(x, Wx, Wh, b):
    """x: (B, n, d_in) -> 마지막 시점 h (B, H). 시점 루프는 직렬이다."""
    B, n, _ = x.shape
    H = Wh.shape[1]
    h, c = np.zeros((B, H)), np.zeros((B, H))
    for t in range(n):
        z = x[:, t] @ Wx.T + h @ Wh.T + b
        i, f, o = sigmoid(z[:, :H]), sigmoid(z[:, H : 2 * H]), sigmoid(z[:, 2 * H : 3 * H])
        c = f * c + i * np.tanh(z[:, 3 * H :])
        h = o * np.tanh(c)
    return h
