"""12장 NumPy 최소 구현: 작은 GPT의 생성 루프를 캐시 없이 / KV cache로 돌리기.

모델은 9장의 Pre-LN 블록과 같은 구조다(토큰 임베딩 + 학습되는 위치 임베딩 + 블록 L개 + 마지막 LN + 출력층).
10장의 학습 루프가 아직 이 저장소에 없으므로, 블록은 9장 코드에서 복사해 생성용으로 고쳤다.
출처: llm/ch09_residual_norm/resnorm_np.py (장끼리 import하지 않고 복사한다)

핵심 함수
- forward(params, tokens, start, cache): 위치 start..start+n-1의 토큰 n개를 처리한다.
  cache가 None이면 이 n개만으로 어텐션한다(캐시 없는 생성은 매 스텝 접두사 전체를 넣는다).
  cache가 있으면 새 위치의 k, v만 계산해 캐시 뒤에 붙이고, 캐시 전체에 어텐션한다.
- generate_no_cache / generate_with_cache: 같은 결과를 내야 하는 두 생성 루프.
- h_kv: 키·값 헤드 수. h_kv == h 이면 멀티헤드(MHA), h_kv == 1 이면 multi-query(MQA, Shazeer 2019),
  그 사이면 grouped-query(GQA, Ainslie 등 2023). 질의 헤드 h개가 키·값 헤드 h_kv개를 나눠 쓴다.
- flops_*: 이 파일의 행렬곱만 센다(곱셈+덧셈 = 2). LN, softmax, 임베딩 조회, 잔차 덧셈은 세지 않는다.
"""

import numpy as np

# ---------------------------------------------------------------- 9장에서 복사한 부분


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ---------------------------------------------------------------- 모델 파라미터


def init_params(vocab, n_ctx, d, L, h, h_kv=None, ff=4, seed=0):
    """무작위 가중치. 학습 없이도 캐시 유무 비교(같은 결과, 다른 비용)에는 충분하다."""
    h_kv = h if h_kv is None else h_kv
    assert d % h == 0 and h % h_kv == 0
    dk = d // h
    rng = np.random.default_rng(seed)
    n = lambda *s: rng.standard_normal(s)  # noqa: E731
    layers = []
    for _ in range(L):
        layers.append({
            "g1": np.ones(d), "be1": np.zeros(d),
            "Wq": n(d, h * dk) / np.sqrt(d),
            "Wk": n(d, h_kv * dk) / np.sqrt(d),
            "Wv": n(d, h_kv * dk) / np.sqrt(d),
            "Wo": n(h * dk, d) / np.sqrt(d),
            "g2": np.ones(d), "be2": np.zeros(d),
            "W1": n(d, ff * d) * np.sqrt(2.0 / d), "b1": np.zeros(ff * d),
            "W2": n(ff * d, d) / np.sqrt(ff * d), "b2": np.zeros(d),
        })
    return {
        "cfg": {"vocab": vocab, "n_ctx": n_ctx, "d": d, "L": L, "h": h, "h_kv": h_kv, "dk": dk, "ff": ff},
        "tok": n(vocab, d), "pos": n(n_ctx, d), "layers": layers,
        "gf": np.ones(d), "bf": np.zeros(d),
        "Wout": n(d, vocab) / np.sqrt(d), "bout": np.zeros(vocab),
    }


# ---------------------------------------------------------------- 캐시


class KVCache:
    """층마다 (h_kv, max_len, dk) 배열을 미리 잡고, 채운 길이만 기억한다."""

    def __init__(self, L, h_kv, max_len, dk, dtype=np.float64):
        self.k = [np.zeros((h_kv, max_len, dk), dtype) for _ in range(L)]
        self.v = [np.zeros((h_kv, max_len, dk), dtype) for _ in range(L)]
        self.length = [0] * L

    def append(self, layer, k, v):
        """k, v: (h_kv, n_new, dk). 붙인 뒤 지금까지의 K, V 전체(뷰)를 돌려준다."""
        a, b = self.length[layer], self.length[layer] + k.shape[1]
        self.k[layer][:, a:b] = k
        self.v[layer][:, a:b] = v
        self.length[layer] = b
        return self.k[layer][:, :b], self.v[layer][:, :b]

    def nbytes(self):
        """실제로 채운 부분의 바이트 수."""
        return sum(self.k[l][:, : self.length[l]].nbytes + self.v[l][:, : self.length[l]].nbytes for l in range(len(self.k)))


def cache_bytes(L, n, h_kv, dk, itemsize=8):
    """K와 V 두 개 × 층 L × 위치 n × 키·값 헤드 h_kv × 헤드 차원 dk × 원소 크기."""
    return 2 * L * n * h_kv * dk * itemsize


# ---------------------------------------------------------------- 순전파


def _split(t, heads, dk):  # (n, heads*dk) -> (heads, n, dk)
    return t.reshape(t.shape[0], heads, dk).transpose(1, 0, 2)


def attention_block(x, p, cfg, start, cache=None, layer=0, causal=True):
    """Pre-LN 어텐션 부분층. x: (n, d), 위치 start..start+n-1."""
    h, h_kv, dk = cfg["h"], cfg["h_kv"], cfg["dk"]
    a = layer_norm(x, p["g1"], p["be1"])
    q = _split(a @ p["Wq"], h, dk)        # (h, n, dk)
    k = _split(a @ p["Wk"], h_kv, dk)     # (h_kv, n, dk): 새 위치의 것만 계산한다
    v = _split(a @ p["Wv"], h_kv, dk)
    if cache is not None:
        k, v = cache.append(layer, k, v)  # 앞 위치의 k, v는 다시 계산하지 않고 꺼내 쓴다
    rep = h // h_kv                       # 질의 헤드 rep개가 키·값 헤드 하나를 나눠 쓴다
    k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
    m = k.shape[1]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dk)  # (h, n, m)
    if causal:
        qpos = start + np.arange(x.shape[0])[:, None]
        scores = np.where(np.arange(m)[None, :] > qpos, -np.inf, scores)
    out = softmax(scores, -1) @ v                     # (h, n, dk)
    return x + out.transpose(1, 0, 2).reshape(x.shape[0], h * dk) @ p["Wo"]


def mlp_block(x, p):
    a = layer_norm(x, p["g2"], p["be2"])
    return x + np.maximum(a @ p["W1"] + p["b1"], 0.0) @ p["W2"] + p["b2"]


def forward(params, tokens, start=0, cache=None, last_only=False, causal=True, return_kv=False):
    """tokens: 정수 n개. 반환: 로짓 (n, vocab), last_only면 (vocab,)."""
    cfg = params["cfg"]
    tokens = np.asarray(tokens)
    x = params["tok"][tokens] + params["pos"][start: start + len(tokens)]
    kvs = []
    for l, p in enumerate(params["layers"]):
        if return_kv:  # 테스트용: 이 층이 만든 k, v (캐시를 쓰지 않을 때)
            a = layer_norm(x, p["g1"], p["be1"])
            kvs.append((a @ p["Wk"], a @ p["Wv"]))
        x = attention_block(x, p, cfg, start, cache, l, causal)
        x = mlp_block(x, p)
    if last_only:
        x = x[-1:]
    logits = layer_norm(x, params["gf"], params["bf"]) @ params["Wout"] + params["bout"]
    logits = logits[0] if last_only else logits
    return (logits, kvs) if return_kv else logits


def new_cache(params, max_len, dtype=np.float64):
    c = params["cfg"]
    return KVCache(c["L"], c["h_kv"], max_len, c["dk"], dtype)


# ---------------------------------------------------------------- 생성 루프


def _pick(logits, rng):
    if rng is None:
        return int(np.argmax(logits))
    return int(rng.choice(len(logits), p=softmax(logits)))


def generate_no_cache(params, prompt, n_new, rng=None, record_logits=False):
    """매 스텝 지금까지의 토큰 전체를 처음부터 다시 넣는다."""
    seq, logs = list(prompt), []
    for _ in range(n_new):
        logits = forward(params, seq, start=0, last_only=True)
        logs.append(logits)
        seq.append(_pick(logits, rng))
    return (seq, logs) if record_logits else seq


def generate_with_cache(params, prompt, n_new, rng=None, record_logits=False, cache=None):
    """프롬프트를 한 번 넣어 캐시를 채운 뒤, 매 스텝 새 토큰 하나만 넣는다."""
    cache = new_cache(params, len(prompt) + n_new) if cache is None else cache
    seq, logs = list(prompt), []
    logits = forward(params, seq, start=0, cache=cache, last_only=True)
    for i in range(n_new):
        logs.append(logits)
        seq.append(_pick(logits, rng))
        if i < n_new - 1:
            logits = forward(params, seq[-1:], start=len(seq) - 1, cache=cache, last_only=True)
    return (seq, logs) if record_logits else seq


# ---------------------------------------------------------------- 계산량


def layer_flops(n, m, d, h, h_kv, dk, ff=4):
    """한 층이 질의 n개를 키 m개에 대해 처리할 때의 행렬곱 FLOP(곱셈+덧셈=2)."""
    q = 2 * n * d * h * dk
    kv = 2 * 2 * n * d * h_kv * dk
    att = 2 * (2 * n * m * h * dk)  # QK^T, AV
    o = 2 * n * h * dk * d
    mlp = 2 * 2 * n * d * ff * d
    return q + kv + att + o + mlp


def flops_no_cache(prompt_len, n_new, d, L, h, h_kv, dk, vocab, ff=4):
    """스텝 s(접두사 길이 t = prompt_len + s)마다 t개 위치 전체를 다시 계산. 출력층은 마지막 위치만."""
    total = 0
    for s in range(n_new):
        t = prompt_len + s
        total += L * layer_flops(t, t, d, h, h_kv, dk, ff) + 2 * d * vocab
    return total


def flops_with_cache(prompt_len, n_new, d, L, h, h_kv, dk, vocab, ff=4):
    """프리필 한 번(prompt_len개) + 새 토큰 하나씩 n_new-1번. 출력층은 스텝마다 한 위치."""
    total = L * layer_flops(prompt_len, prompt_len, d, h, h_kv, dk, ff) + 2 * d * vocab
    for s in range(1, n_new):
        t = prompt_len + s
        total += L * layer_flops(1, t, d, h, h_kv, dk, ff) + 2 * d * vocab
    return total


def positions_processed(prompt_len, n_new, cached):
    """한 층이 생성 전체에서 순전파한 위치 수(=k, v를 계산한 횟수)."""
    if cached:
        return prompt_len + (n_new - 1)
    return sum(prompt_len + s for s in range(n_new))
