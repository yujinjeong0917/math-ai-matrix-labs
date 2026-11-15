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

입력: 파라미터 dict, 프롬프트 토큰 id 목록, 생성 길이 n_new. 출력: 생성된 토큰 목록(와 필요하면 스텝별 로짓).
기호: n = 이번에 처리하는 새 위치 수, m = 어텐션이 보는 키 위치 수(캐시 길이 포함), d = 모델 차원, h = 질의 헤드 수, h_kv = 키·값 헤드 수,
dk = 헤드당 차원 = d / h, L = 층 수. 장 본문 숫자와의 대응은 experiments.py 머리말 참고.
"""

import numpy as np

# ---------------------------------------------------------------- 9장에서 복사한 부분


def softmax(z, axis=-1):
    # log-sum-exp 안정화: 최댓값을 빼도 softmax는 같다
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    # 편향 분산(1/d). 위치마다 마지막 축(d)에서 따로 구한다
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ---------------------------------------------------------------- 모델 파라미터


def init_params(vocab, n_ctx, d, L, h, h_kv=None, ff=4, seed=0):
    """무작위 가중치. 학습 없이도 캐시 유무 비교(같은 결과, 다른 비용)에는 충분하다."""
    h_kv = h if h_kv is None else h_kv
    # d는 h의 배수, h는 h_kv의 배수여야 한다(GQA 그룹 크기 h // h_kv가 정수)
    assert d % h == 0 and h % h_kv == 0
    dk = d // h
    rng = np.random.default_rng(seed)
    # 표준정규 난수 생성기 단축. 같은 rng에서 순서대로 뽑으므로 호출 순서를 바꾸면 가중치가 달라진다
    n = lambda *s: rng.standard_normal(s)  # noqa: E731
    layers = []
    for _ in range(L):
        layers.append({
            "g1": np.ones(d), "be1": np.zeros(d),
            # Wq: (d, h*dk). 질의는 헤드 h개 전부
            "Wq": n(d, h * dk) / np.sqrt(d),
            # Wk, Wv: (d, h_kv*dk). h_kv가 작을수록 캐시가 작아진다. 분산 1/d로 초기화
            "Wk": n(d, h_kv * dk) / np.sqrt(d),
            "Wv": n(d, h_kv * dk) / np.sqrt(d),
            "Wo": n(h * dk, d) / np.sqrt(d),
            "g2": np.ones(d), "be2": np.zeros(d),
            "W1": n(d, ff * d) * np.sqrt(2.0 / d), "b1": np.zeros(ff * d),
            "W2": n(ff * d, d) / np.sqrt(ff * d), "b2": np.zeros(d),
        })
    return {
        "cfg": {"vocab": vocab, "n_ctx": n_ctx, "d": d, "L": L, "h": h, "h_kv": h_kv, "dk": dk, "ff": ff},
        # 토큰 임베딩, 위치 임베딩 모두 N(0, 1). pos는 n_ctx행뿐
        "tok": n(vocab, d), "pos": n(n_ctx, d), "layers": layers,
        "gf": np.ones(d), "bf": np.zeros(d),
        "Wout": n(d, vocab) / np.sqrt(d), "bout": np.zeros(vocab),
    }


# ---------------------------------------------------------------- 캐시


class KVCache:
    """층마다 (h_kv, max_len, dk) 배열을 미리 잡고, 채운 길이만 기억한다."""

    def __init__(self, L, h_kv, max_len, dk, dtype=np.float64):
        # 층마다 (h_kv, max_len, dk)를 미리 할당한다. 아직 안 채운 칸은 0이지만 length 이전만 읽는다
        self.k = [np.zeros((h_kv, max_len, dk), dtype) for _ in range(L)]
        self.v = [np.zeros((h_kv, max_len, dk), dtype) for _ in range(L)]
        self.length = [0] * L

    def append(self, layer, k, v):
        """k, v: (h_kv, n_new, dk). 붙인 뒤 지금까지의 K, V 전체(뷰)를 돌려준다."""
        # 이번에 채울 구간 [a, b)
        a, b = self.length[layer], self.length[layer] + k.shape[1]
        self.k[layer][:, a:b] = k
        self.v[layer][:, a:b] = v
        self.length[layer] = b
        # 앞 b칸의 뷰(복사 아님). 이후 연산이 이 배열을 바꾸면 캐시도 바뀌므로 읽기만 한다
        return self.k[layer][:, :b], self.v[layer][:, :b]

    def nbytes(self):
        """실제로 채운 부분의 바이트 수."""
        return sum(self.k[l][:, : self.length[l]].nbytes + self.v[l][:, : self.length[l]].nbytes for l in range(len(self.k)))


def cache_bytes(L, n, h_kv, dk, itemsize=8):
    """K와 V 두 개 × 층 L × 위치 n × 키·값 헤드 h_kv × 헤드 차원 dk × 원소 크기."""
    # 2 = K와 V. itemsize 8 = float64
    return 2 * L * n * h_kv * dk * itemsize


# ---------------------------------------------------------------- 순전파


def _split(t, heads, dk):  # (n, heads*dk) -> (heads, n, dk)
    # (n, heads*dk) -> (n, heads, dk) -> (heads, n, dk)
    return t.reshape(t.shape[0], heads, dk).transpose(1, 0, 2)


def attention_block(x, p, cfg, start, cache=None, layer=0, causal=True):
    """Pre-LN 어텐션 부분층. x: (n, d), 위치 start..start+n-1."""
    h, h_kv, dk = cfg["h"], cfg["h_kv"], cfg["dk"]
    # Pre-LN: 부분층 입력만 정규화하고 잔차 x는 그대로 둔다
    a = layer_norm(x, p["g1"], p["be1"])
    # 식: Q = LN(x) W_Q, 헤드로 분해 (h, n, dk)
    q = _split(a @ p["Wq"], h, dk)        # (h, n, dk)
    k = _split(a @ p["Wk"], h_kv, dk)     # (h_kv, n, dk): 새 위치의 것만 계산한다
    v = _split(a @ p["Wv"], h_kv, dk)
    # 캐시가 있으면 k, v는 (h_kv, m, dk)로 바뀐다(m = 이전 길이 + n)
    if cache is not None:
        k, v = cache.append(layer, k, v)  # 앞 위치의 k, v는 다시 계산하지 않고 꺼내 쓴다
    rep = h // h_kv                       # 질의 헤드 rep개가 키·값 헤드 하나를 나눠 쓴다
    # GQA/MQA: 키·값 헤드를 질의 헤드 수에 맞게 복제 (h_kv, m, dk) -> (h, m, dk). 계산량은 그대로지만 저장은 h_kv개만
    k, v = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
    m = k.shape[1]
    # 식: Q K^T / sqrt(d_k). (h, n, dk) @ (h, dk, m) -> (h, n, m)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dk)  # (h, n, m)
    if causal:
        # 질의의 절대 위치(start + 행 번호). 키 위치 j > 질의 위치면 가린다. 주의: cache=None이고 start > 0이면 키 위치를 0부터라고 가정해 마스크가 틀려진다(이 파일은 그 경우를 쓰지 않는다)
        qpos = start + np.arange(x.shape[0])[:, None]
        scores = np.where(np.arange(m)[None, :] > qpos, -np.inf, scores)
    # 식: softmax(scores) V -> (h, n, dk). 이어 붙여 (n, h*dk) 뒤 W_O를 곱하고 잔차를 더한다
    out = softmax(scores, -1) @ v                     # (h, n, dk)
    return x + out.transpose(1, 0, 2).reshape(x.shape[0], h * dk) @ p["Wo"]


def mlp_block(x, p):
    a = layer_norm(x, p["g2"], p["be2"])
    # Pre-LN MLP: x + ReLU(LN(x) W1 + b1) W2 + b2
    return x + np.maximum(a @ p["W1"] + p["b1"], 0.0) @ p["W2"] + p["b2"]


def forward(params, tokens, start=0, cache=None, last_only=False, causal=True, return_kv=False):
    """tokens: 정수 n개. 반환: 로짓 (n, vocab), last_only면 (vocab,)."""
    cfg = params["cfg"]
    tokens = np.asarray(tokens)
    # 토큰 임베딩 + 절대 위치 임베딩. 위치는 start부터: 캐시로 새 토큰만 넣을 때도 올바른 위치를 써야 한다(함정)
    x = params["tok"][tokens] + params["pos"][start: start + len(tokens)]
    kvs = []
    for l, p in enumerate(params["layers"]):
        if return_kv:  # 테스트용: 이 층이 만든 k, v (캐시를 쓰지 않을 때)
            # 테스트용 k, v 기록(분해 전 (n, h_kv*dk))
            a = layer_norm(x, p["g1"], p["be1"])
            kvs.append((a @ p["Wk"], a @ p["Wv"]))
        x = attention_block(x, p, cfg, start, cache, l, causal)
        x = mlp_block(x, p)
    if last_only:
        # 마지막 위치만 남겨 출력층 비용을 줄인다(생성에서는 마지막 로짓만 필요)
        x = x[-1:]
    logits = layer_norm(x, params["gf"], params["bf"]) @ params["Wout"] + params["bout"]
    logits = logits[0] if last_only else logits
    return (logits, kvs) if return_kv else logits


# max_len: 프롬프트 + 생성 길이
def new_cache(params, max_len, dtype=np.float64):
    c = params["cfg"]
    return KVCache(c["L"], c["h_kv"], max_len, c["dk"], dtype)


# ---------------------------------------------------------------- 생성 루프


def _pick(logits, rng):
    if rng is None:
        # rng가 없으면 탐욕(argmax), 있으면 softmax 확률로 표집
        return int(np.argmax(logits))
    return int(rng.choice(len(logits), p=softmax(logits)))


def generate_no_cache(params, prompt, n_new, rng=None, record_logits=False):
    """매 스텝 지금까지의 토큰 전체를 처음부터 다시 넣는다."""
    seq, logs = list(prompt), []
    for _ in range(n_new):
        # 캐시 없음: 스텝마다 지금까지의 접두사 전체를 다시 계산한다 (비용이 길이에 따라 제곱으로 증가)
        logits = forward(params, seq, start=0, last_only=True)
        logs.append(logits)
        seq.append(_pick(logits, rng))
    return (seq, logs) if record_logits else seq


def generate_with_cache(params, prompt, n_new, rng=None, record_logits=False, cache=None):
    """프롬프트를 한 번 넣어 캐시를 채운 뒤, 매 스텝 새 토큰 하나만 넣는다."""
    cache = new_cache(params, len(prompt) + n_new) if cache is None else cache
    seq, logs = list(prompt), []
    # 프리필(prefill): 프롬프트 전체를 한 번에 넣어 캐시를 채우고 마지막 로짓을 얻는다
    logits = forward(params, seq, start=0, cache=cache, last_only=True)
    for i in range(n_new):
        logs.append(logits)
        seq.append(_pick(logits, rng))
        # 마지막 스텝은 다음 로짓이 필요 없어 순전파를 생략한다(off-by-one 주의: 캐시 길이가 prompt + n_new - 1)
        if i < n_new - 1:
            # 디코드(decode): 새 토큰 하나만, 위치 len(seq)-1로 넣는다
            logits = forward(params, seq[-1:], start=len(seq) - 1, cache=cache, last_only=True)
    return (seq, logs) if record_logits else seq


# ---------------------------------------------------------------- 계산량


def layer_flops(n, m, d, h, h_kv, dk, ff=4):
    """한 층이 질의 n개를 키 m개에 대해 처리할 때의 행렬곱 FLOP(곱셈+덧셈=2)."""
    # 행렬곱 (n, d) @ (d, h*dk)는 2*n*d*h*dk (곱셈+덧셈)
    q = 2 * n * d * h * dk
    # K, V 두 개: 2 * (2*n*d*h_kv*dk)
    kv = 2 * 2 * n * d * h_kv * dk
    # QK^T와 AV가 각각 2*n*m*h*dk
    att = 2 * (2 * n * m * h * dk)  # QK^T, AV
    # 출력 투영 W_O
    o = 2 * n * h * dk * d
    # MLP 두 행렬곱: 각각 2*n*d*(ff*d)
    mlp = 2 * 2 * n * d * ff * d
    return q + kv + att + o + mlp


def flops_no_cache(prompt_len, n_new, d, L, h, h_kv, dk, vocab, ff=4):
    """스텝 s(접두사 길이 t = prompt_len + s)마다 t개 위치 전체를 다시 계산. 출력층은 마지막 위치만."""
    total = 0
    for s in range(n_new):
        t = prompt_len + s
        # 접두사 t개 전체를 질의 t개 * 키 t개로 처리. 출력층 2*d*vocab은 마지막 위치 하나
        total += L * layer_flops(t, t, d, h, h_kv, dk, ff) + 2 * d * vocab
    return total


def flops_with_cache(prompt_len, n_new, d, L, h, h_kv, dk, vocab, ff=4):
    """프리필 한 번(prompt_len개) + 새 토큰 하나씩 n_new-1번. 출력층은 스텝마다 한 위치."""
    # 프리필: 질의 prompt_len개, 키 prompt_len개
    total = L * layer_flops(prompt_len, prompt_len, d, h, h_kv, dk, ff) + 2 * d * vocab
    for s in range(1, n_new):
        t = prompt_len + s
        # 스텝 s: 새 질의 1개가 키 t개(캐시 + 자기 자신)를 본다. 여기서 t = prompt_len + s는 새 토큰까지 포함한 길이
        total += L * layer_flops(1, t, d, h, h_kv, dk, ff) + 2 * d * vocab
    return total


def positions_processed(prompt_len, n_new, cached):
    """한 층이 생성 전체에서 순전파한 위치 수(=k, v를 계산한 횟수)."""
    if cached:
        # 프리필 prompt_len개 + 디코드 n_new - 1개(마지막 토큰은 순전파 안 함)
        return prompt_len + (n_new - 1)
    return sum(prompt_len + s for s in range(n_new))
