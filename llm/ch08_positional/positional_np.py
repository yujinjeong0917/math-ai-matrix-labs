"""8장 NumPy 최소 구현: self-attention에 위치 정보를 넣는 네 가지 방법.

원리만 드러나도록 배치 차원 없이 (n, d) 행렬 하나, 헤드 하나를 다룬다.
- none: 7장의 self-attention 그대로. 마스크가 없으면 순열 등변이다.
- sin / learned: 입력에 위치 벡터를 더한다. x_i <- x_i + p_i  (Vaswani 등 2017 §3.5)
- rope: 질의·키를 위치만큼 회전한다. 점수가 거리 m - n에만 의존한다  (Su 등 2021)
- alibi: 점수에서 거리에 비례하는 값을 뺀다. e_ij = q_i·k_j - m (i - j)  (Press 등 2022)

softmax, causal_mask, self_attention의 기본 꼴은 7장(ch07_self_attention/attention_np.py)에서 복사했다.
"""

import numpy as np

KINDS = ("none", "sin", "learned", "rope", "alibi")


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # 7장과 같은 log-sum-exp 안정화
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """미래 위치(j > i)를 -inf로 막는 (n, n) 가산 마스크. 7장에서 복사."""
    return np.triu(np.full((n, n), -np.inf), k=1)


def sinusoidal(n, d, base=10000.0):
    """PE[pos, 2i] = sin(pos / base^(2i/d)), PE[pos, 2i+1] = cos(같은 값). 위치는 0부터 센다."""
    pos = np.arange(n)[:, None]
    freq = base ** (-np.arange(0, d, 2) / d)  # (d/2,)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe


def learned_pos(n_max, d, rng):
    """학습형 위치 표: 위치마다 d차원 벡터 하나. 파라미터 n_max * d개, n_max 너머의 칸은 없다."""
    return rng.standard_normal((n_max, d))


def rope_angles(positions, d, base=10000.0):
    """RoFormer 식 (15): theta_k = base^(-2k/d), 위치 m의 k번째 짝은 m * theta_k 만큼 돈다."""
    theta = base ** (-np.arange(0, d, 2) / d)
    return np.asarray(positions)[:, None] * theta  # (n, d/2)


def rope_rotate(x, positions, base=10000.0):
    """x의 성분을 (0,1), (2,3), ... 짝으로 묶어 2차원 회전을 적용한다. 길이(노름)는 변하지 않는다."""
    ang = rope_angles(positions, x.shape[-1], base)
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def apply_rope(q, k, positions, base=10000.0):
    return rope_rotate(q, positions, base), rope_rotate(k, positions, base)


def alibi_slopes(n_heads):
    """Press 등(2022) §3: 2^(-8/h)에서 시작해 같은 값을 공비로 하는 등비수열."""
    start = 2.0 ** (-8.0 / n_heads)
    return start ** np.arange(1, n_heads + 1)


def alibi_bias(n, slope):
    """(n, n) 편향 행렬. 칸 (i, j)는 -slope * (i - j). 미래 칸(j > i)은 마스크가 따로 막는다."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return -slope * (i - j).astype(float)


def attention_with_pos(x, Wq, Wk, Wv, kind="none", pos_table=None, slope=0.25, causal=True, return_weights=False):
    """x: (n, d_model), W*: (d_model, d_k). kind는 KINDS 중 하나.

    sin/learned는 입력에, rope는 질의·키에, alibi는 점수에 위치를 넣는다. 나머지는 7장과 같다.
    """
    n = x.shape[0]
    if kind == "sin":
        x = x + sinusoidal(n, x.shape[1])
    elif kind == "learned":
        if n > pos_table.shape[0]:
            raise ValueError(f"학습형 위치 표에는 위치 {pos_table.shape[0]} 이상의 칸이 없어요 (n={n})")
        x = x + pos_table[:n]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    if kind == "rope":
        Q, K = apply_rope(Q, K, np.arange(n))
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if kind == "alibi":
        scores = scores + alibi_bias(n, slope)
    if causal:
        scores = scores + causal_mask(n)
    A = softmax(scores, axis=-1)
    out = A @ V
    return (out, A) if return_weights else out


def extra_cost(kind, n, d, n_heads=1, n_max=None):
    """위치 정보 때문에 한 층에 더해지는 연산 수와 파라미터 수(곱셈·덧셈을 각각 1로 센다).

    sin/learned: 입력에 더하기 n*d. rope: 질의·키 두 개, 성분 짝마다 곱 4 + 덧셈 2 = 성분당 3, 즉 2*3*n*d.
    alibi: 헤드마다 점수 n*n칸에 더하기. 회전 각도(cos, sin)와 사인 표는 미리 계산해 둔다고 본다.
    """
    flops = {"none": 0, "sin": n * d, "learned": n * d, "rope": 6 * n * d, "alibi": n_heads * n * n}[kind]
    params = n_max * d if kind == "learned" else 0
    return flops, params


def first_screen_toy():
    """첫 화면의 손 계산 예. 낱말 벡터는 (값, 0), 위치 벡터는 90도씩 도는 (sin, cos)이다.

    "3 빼기 5"와 "5 빼기 3"은 위치가 없으면 같은 벡터 묶음이고, 위치를 더하면 다른 묶음이 된다.
    """
    word = {"3": (3, 0), "빼기": (1, 0), "5": (5, 0)}
    deg = [90 * i for i in range(3)]
    pos = [(round(float(np.sin(np.radians(a)))), round(float(np.cos(np.radians(a))))) for a in deg]
    out = {"position_vectors_deg90": pos}
    for sent in (("3", "빼기", "5"), ("5", "빼기", "3")):
        plain = [word[w] for w in sent]
        added = [(word[w][0] + p[0], word[w][1] + p[1]) for w, p in zip(sent, pos)]
        out[" ".join(sent)] = {"no_position": plain, "with_position": added}
    a, b = out["3 빼기 5"], out["5 빼기 3"]
    out["same_set_without_position"] = sorted(a["no_position"]) == sorted(b["no_position"])
    out["same_set_with_position"] = sorted(a["with_position"]) == sorted(b["with_position"])
    out["sinusoidal_d2_radians_pos0to3"] = [[round(float(v), 4) for v in r] for r in sinusoidal(4, 2)]
    return out
