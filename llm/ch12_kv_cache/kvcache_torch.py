"""12장 PyTorch 대응 구현.

kvcache_np.py와 같은 수식을 텐서로 옮긴다(배치 차원 B만 붙는다). 블록 구조는 9장 Pre-LN에서 복사했다.
출처: llm/ch09_residual_norm/resnorm_torch.py

캐시 두 가지를 같은 인터페이스로 둔다.
- CatCache: 스텝마다 torch.cat으로 K, V를 한 칸씩 늘린다. 매번 새 텐서를 만들고 앞부분을 복사한다.
- PreallocCache: 최대 길이만큼 미리 잡아 두고 길이 포인터만 옮긴다. Shazeer(2019) 각주처럼
  고정 모양이 필요한 시스템이 쓰는 방식이고, 대신 아직 안 쓴 칸까지 메모리를 차지한다.

텐서 모양: 토큰 id (B, n), 로짓 (B, n, vocab) 또는 last_only이면 (B, vocab), 캐시 K·V (B, h_kv, m, dk).
B = 배치, n = 새 위치 수, m = 캐시 포함 키 길이, h_kv = 키·값 헤드 수, dk = 헤드 차원.
"""

import math

import torch
import torch.nn as nn


class CatCache:
    def __init__(self, L):
        # 층마다 하나씩. 처음엔 비어 있다
        self.k, self.v = [None] * L, [None] * L

    def append(self, layer, k, v):  # k, v: (B, h_kv, n_new, dk)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            # 시간 축(dim=2)으로 이어 붙인다. 매번 새 텐서를 만들어 앞부분을 복사하므로 비용이 늘어난다
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]


class PreallocCache:
    def __init__(self, L, B, h_kv, max_len, dk, dtype=torch.float32):
        # (L, B, h_kv, max_len, dk): 전 층을 한 텐서로 미리 할당. 안 쓴 칸도 메모리를 차지한다
        self.k = torch.zeros(L, B, h_kv, max_len, dk, dtype=dtype)
        self.v = torch.zeros(L, B, h_kv, max_len, dk, dtype=dtype)
        self.length = [0] * L

    def append(self, layer, k, v):
        a, b = self.length[layer], self.length[layer] + k.shape[2]
        self.k[layer, :, :, a:b] = k
        self.v[layer, :, :, a:b] = v
        self.length[layer] = b
        # 앞 b칸의 뷰 (B, h_kv, b, dk)
        return self.k[layer, :, :, :b], self.v[layer, :, :, :b]


class Attention(nn.Module):
    def __init__(self, d, h, h_kv):
        super().__init__()
        self.h, self.h_kv, self.dk = h, h_kv, d // h
        self.Wq = nn.Parameter(torch.randn(d, h * self.dk) / math.sqrt(d))
        self.Wk = nn.Parameter(torch.randn(d, h_kv * self.dk) / math.sqrt(d))
        self.Wv = nn.Parameter(torch.randn(d, h_kv * self.dk) / math.sqrt(d))
        self.Wo = nn.Parameter(torch.randn(h * self.dk, d) / math.sqrt(d))

    def forward(self, a, start=0, cache=None, layer=0):
        B, n, _ = a.shape
        # (B, n, H*dk) -> (B, n, H, dk) -> (B, H, n, dk)
        split = lambda t, H: t.view(B, n, H, self.dk).transpose(1, 2)  # noqa: E731
        # Q: (B, h, n, dk)
        q = split(a @ self.Wq, self.h)
        k, v = split(a @ self.Wk, self.h_kv), split(a @ self.Wv, self.h_kv)
        if cache is not None:
            k, v = cache.append(layer, k, v)
        rep = self.h // self.h_kv
        if rep > 1:
            # GQA: 헤드 축(dim=1)에서 각 키·값 헤드를 rep번 복제해 (B, h, m, dk). repeat_interleave는 이웃 헤드끼리 묶어 NumPy np.repeat와 같은 대응
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        m = k.shape[2]
        # 식: Q K^T / sqrt(d_k) -> (B, h, n, m)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        # 질의의 절대 위치 = start + 행 번호. 키 위치가 이보다 크면 가린다(NumPy 구현과 같은 규칙)
        qpos = start + torch.arange(n)[:, None]
        scores = scores.masked_fill(torch.arange(m)[None, :] > qpos, float("-inf"))
        out = torch.softmax(scores, -1) @ v
        # (B, h, n, dk) -> (B, n, h*dk) 뒤 W_O. 잔차는 Block에서 더한다
        return out.transpose(1, 2).reshape(B, n, self.h * self.dk) @ self.Wo


class Block(nn.Module):
    def __init__(self, d, h, h_kv, ff=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, h_kv)
        self.W1 = nn.Parameter(torch.randn(d, ff * d) * math.sqrt(2.0 / d))
        self.b1 = nn.Parameter(torch.zeros(ff * d))
        self.W2 = nn.Parameter(torch.randn(ff * d, d) / math.sqrt(ff * d))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x, start=0, cache=None, layer=0):
        # Pre-LN: x + Attn(LN(x))
        x = x + self.attn(self.ln1(x), start, cache, layer)
        return x + torch.relu(self.ln2(x) @ self.W1 + self.b1) @ self.W2 + self.b2


class TinyGPT(nn.Module):
    def __init__(self, vocab, n_ctx, d, L, h, h_kv=None, ff=4):
        super().__init__()
        h_kv = h if h_kv is None else h_kv
        self.cfg = {"vocab": vocab, "n_ctx": n_ctx, "d": d, "L": L, "h": h, "h_kv": h_kv, "dk": d // h, "ff": ff}
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(n_ctx, d))
        self.blocks = nn.ModuleList(Block(d, h, h_kv, ff) for _ in range(L))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1.0 / math.sqrt(d))
            self.head.bias.zero_()

    def forward(self, tokens, start=0, cache=None, last_only=False):
        """tokens: (B, n). 반환: (B, n, vocab) 또는 last_only면 (B, vocab)."""
        # 토큰 임베딩 (B, n, d) + 위치 임베딩 (n, d). 위치는 start부터 시작
        x = self.tok(tokens) + self.pos[start: start + tokens.shape[1]]
        for l, b in enumerate(self.blocks):
            x = b(x, start, cache, l)
        if last_only:
            # 마지막 위치만: (B, n, d) -> (B, d)
            x = x[:, -1]
        return self.head(self.lnf(x))


def from_numpy(params):
    """kvcache_np.init_params와 같은 가중치를 가진 float64 TinyGPT."""
    c = params["cfg"]
    m = TinyGPT(c["vocab"], c["n_ctx"], c["d"], c["L"], c["h"], c["h_kv"], c["ff"]).double()
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    with torch.no_grad():
        # NumPy 파라미터를 같은 값으로 복사해 수치 대조가 가능하게 한다
        m.tok.weight.copy_(t(params["tok"]))
        m.pos.copy_(t(params["pos"]))
        for b, p in zip(m.blocks, params["layers"]):
            for k in ("Wq", "Wk", "Wv", "Wo"):
                getattr(b.attn, k).copy_(t(p[k]))
            for k in ("W1", "b1", "W2", "b2"):
                getattr(b, k).copy_(t(p[k]))
            b.ln1.weight.copy_(t(p["g1"])), b.ln1.bias.copy_(t(p["be1"]))
            b.ln2.weight.copy_(t(p["g2"])), b.ln2.bias.copy_(t(p["be2"]))
        m.lnf.weight.copy_(t(params["gf"])), m.lnf.bias.copy_(t(params["bf"]))
        # nn.Linear의 weight는 (out, in) 모양이라 NumPy의 Wout (d, vocab)을 전치해 넣는다(함정)
        m.head.weight.copy_(t(params["Wout"]).T), m.head.bias.copy_(t(params["bout"]))
    return m


@torch.no_grad()
def generate(model, prompt, n_new, mode="prealloc"):
    """탐욕 생성. prompt: (B, p) 정수 텐서. mode: "none" | "cat" | "prealloc"."""
    c = model.cfg
    seq = prompt
    B = prompt.shape[0]
    if mode == "none":
        for _ in range(n_new):
            # 캐시 없음: 매 스텝 접두사 전체를 처음부터 다시 계산. 탐욕(argmax)이라 결정적
            nxt = model(seq, last_only=True).argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        return seq
    dtype = next(model.parameters()).dtype
    # mode에 따라 cat 캐시 또는 미리 할당 캐시(길이 prompt + n_new)
    cache = CatCache(c["L"]) if mode == "cat" else PreallocCache(c["L"], B, c["h_kv"], prompt.shape[1] + n_new, c["dk"], dtype)
    # 프리필: 프롬프트 전체를 넣어 캐시를 채운다
    logits = model(seq, 0, cache, last_only=True)
    for i in range(n_new):
        nxt = logits.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        if i < n_new - 1:
            # 디코드: 새 토큰 하나만, 위치 = 새 길이 - 1
            logits = model(nxt, seq.shape[1] - 1, cache, last_only=True)
    return seq
