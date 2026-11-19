"""14장 NumPy 최소 구현: 전문가 혼합(MoE) 라우팅, 부하분산 손실, 파라미터·KV 세기.

- softmax, topk_gate: Shazeer 등(2017) 식 (3), (5)의 G(x) = Softmax(KeepTopK(x·Wg, k)). 뽑히지 않은 전문가의 게이트는 0이다.
- moe_forward: y = Σ_i G(x)_i · E_i(x). 게이트가 0인 전문가는 계산하지 않는다.
- balance_loss: Fedus 등(2021) 식 (4)~(6)의 보조 손실 α·N·Σ f_i·P_i. f는 1순위 전문가로 간 토큰 비율, P는 라우터 확률의 평균.
- expert_capacity, dropped_tokens: 식 (3)의 전문가 용량 = (배치 토큰 수 / 전문가 수) × 용량 계수. 넘친 토큰은 세어서 버린다.
- params_dense_block, params_moe_block, model_params: 블록 하나와 모델 전체의 총 파라미터와 토큰당 활성 파라미터.
- kv_bytes_per_token: 층마다 K, V를 쌓는 어텐션의 토큰당 KV cache 바이트 수 = 2 · L · n_kv · d_head · 자료형 바이트.

입력: 토큰 행렬 x (T×d), 라우터 가중치 Wg (d×E), 전문가 함수 목록, 모델 설정 dict.
출력: 혼합 출력, 게이트 행렬, 손실 값, 파라미터 수와 바이트 수.
기호: T = 토큰 수, d = 모델 차원, E = 전문가 수, k = 토큰당 선택 전문가 수, f_i = i번 전문가로 간 토큰 비율, P_i = i번 전문가의 라우터 확률 평균.
"""

import numpy as np


def softmax(z, axis=-1):
    # 최댓값을 빼서 exp가 넘치지 않게 한다. 결과는 같다
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def topk_gate(logits, k):
    """각 행에서 큰 k개만 남기고 나머지를 -inf로 바꾼 뒤 softmax. 식 (3), (5)."""
    T, E = logits.shape
    # 행마다 k번째로 큰 값의 위치들. argpartition은 정렬 없이 상위 k개를 뽑는다
    idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    masked = np.full_like(logits, -np.inf)
    rows = np.arange(T)[:, None]
    masked[rows, idx] = logits[rows, idx]
    # exp(-inf) = 0이라 뽑히지 않은 전문가의 게이트는 정확히 0이다
    return softmax(masked), idx


def switch_gate(logits, k):
    """Fedus 등(2021) 식 (1)(2): 전체 softmax 확률을 그대로 게이트로 쓰고 상위 k개 밖은 0. 다시 나누지 않는다."""
    T, E = logits.shape
    p = softmax(logits)
    idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    G = np.zeros_like(p)
    rows = np.arange(T)[:, None]
    G[rows, idx] = p[rows, idx]
    return G, idx


def moe_forward(x, Wg, experts, k, gate="renorm"):
    """y_t = Σ_i G(x_t)_i · E_i(x_t). 게이트가 0이 아닌 토큰만 그 전문가에 보낸다."""
    logits = x @ Wg
    G, idx = (switch_gate if gate == "switch" else topk_gate)(logits, k)
    y = np.zeros((x.shape[0], experts[0](x[:1]).shape[1]))
    calls = np.zeros(len(experts), dtype=int)
    for i, f in enumerate(experts):
        sel = np.where(G[:, i] > 0)[0]
        calls[i] = len(sel)
        if len(sel):
            y[sel] += G[sel, i][:, None] * f(x[sel])
    return y, G, calls


def balance_loss(probs, alpha=1e-2):
    """Fedus 등(2021) 식 (4)~(6). probs는 라우터 softmax 확률(T×E). 1순위(argmax)로 f를 센다.

    균등 라우팅이면 f_i = P_i = 1/E라서 E·Σ f·P = 1이 되어 α와 같다. f와 P가 비슷할 때는 쏠릴수록 커진다(f와 P가 다르면 1보다 작아질 수도 있다).
    """
    T, E = probs.shape
    f = np.bincount(probs.argmax(axis=1), minlength=E) / T
    P = probs.mean(axis=0)
    return alpha * E * float(np.sum(f * P)), f, P


def expert_capacity(tokens, E, capacity_factor):
    """식 (3): 전문가 한 명이 받을 수 있는 토큰 수."""
    return int(np.ceil(tokens / E * capacity_factor))


def dropped_tokens(assign, E, capacity):
    """1순위 전문가별로 용량을 넘은 토큰 수의 합. 앞에 온 토큰부터 받는다고 가정한다."""
    counts = np.bincount(assign, minlength=E)
    return int(np.maximum(counts - capacity, 0).sum())


def params_attn(d, n_heads, n_kv, d_head):
    # W_Q(d × h·dh) + W_K, W_V(d × kv·dh 각각) + W_O(h·dh × d). 편향 없음
    return d * n_heads * d_head + 2 * d * n_kv * d_head + n_heads * d_head * d


def params_ffn(d, hidden, swiglu=True):
    # SwiGLU는 d→hidden 두 개(게이트, 업)와 hidden→d 한 개로 행렬 3개. 일반 FFN은 2개
    return (3 if swiglu else 2) * d * hidden


def model_params(cfg, E=1, k=1, swiglu=True):
    """(총, 토큰당 활성) 파라미터 수. 임베딩과 출력층은 따로 두고(공유 안 함), 정규화 층은 뺀다.

    E = 1이면 보통의 밀집 모델이다. 라우터 d×E는 E > 1일 때만 있다.
    """
    d, L = cfg["d"], cfg["L"]
    attn = params_attn(d, cfg["h"], cfg["kv"], cfg["dh"])
    ffn = params_ffn(d, cfg["ff"], swiglu)
    gate = d * E if E > 1 else 0
    total_blk = attn + E * ffn + gate
    active_blk = attn + k * ffn + gate
    emb = 2 * cfg["V"] * d
    return L * total_blk + emb, L * active_blk + emb


def kv_bytes_per_token(L, n_kv, d_head, bytes_per=2):
    # K와 V 두 벌, 층마다, 헤드마다
    return 2 * L * n_kv * d_head * bytes_per
