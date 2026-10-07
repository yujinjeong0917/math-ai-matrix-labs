"""12장 PyTorch 대응 구현.

kvcache_np.py와 같은 수식을 텐서로 옮긴다(배치 차원 B만 붙는다). 블록 구조는 9장 Pre-LN에서 복사했다.
출처: llm/ch09_residual_norm/resnorm_torch.py

캐시 두 가지를 같은 인터페이스로 둔다.
- CatCache: 스텝마다 torch.cat으로 K, V를 한 칸씩 늘린다. 매번 새 텐서를 만들고 앞부분을 복사한다.
- PreallocCache: 최대 길이만큼 미리 잡아 두고 길이 포인터만 옮긴다. Shazeer(2019) 각주처럼
  고정 모양이 필요한 시스템이 쓰는 방식이고, 대신 아직 안 쓴 칸까지 메모리를 차지한다.
"""

import math

import torch
import torch.nn as nn


class CatCache:
    def __init__(self, L):
        self.k, self.v = [None] * L, [None] * L

    def append(self, layer, k, v):  # k, v: (B, h_kv, n_new, dk)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]


class PreallocCache:
    def __init__(self, L, B, h_kv, max_len, dk, dtype=torch.float32):
        self.k = torch.zeros(L, B, h_kv, max_len, dk, dtype=dtype)
        self.v = torch.zeros(L, B, h_kv, max_len, dk, dtype=dtype)
        self.length = [0] * L

    def append(self, layer, k, v):
        a, b = self.length[layer], self.length[layer] + k.shape[2]
        self.k[layer, :, :, a:b] = k
        self.v[layer, :, :, a:b] = v
        self.length[layer] = b
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
        split = lambda t, H: t.view(B, n, H, self.dk).transpose(1, 2)  # noqa: E731
        q = split(a @ self.Wq, self.h)
        k, v = split(a @ self.Wk, self.h_kv), split(a @ self.Wv, self.h_kv)
        if cache is not None:
            k, v = cache.append(layer, k, v)
        rep = self.h // self.h_kv
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        m = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        qpos = start + torch.arange(n)[:, None]
        scores = scores.masked_fill(torch.arange(m)[None, :] > qpos, float("-inf"))
        out = torch.softmax(scores, -1) @ v
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
        x = self.tok(tokens) + self.pos[start: start + tokens.shape[1]]
        for l, b in enumerate(self.blocks):
            x = b(x, start, cache, l)
        if last_only:
            x = x[:, -1]
        return self.head(self.lnf(x))


def from_numpy(params):
    """kvcache_np.init_params와 같은 가중치를 가진 float64 TinyGPT."""
    c = params["cfg"]
    m = TinyGPT(c["vocab"], c["n_ctx"], c["d"], c["L"], c["h"], c["h_kv"], c["ff"]).double()
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    with torch.no_grad():
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
            nxt = model(seq, last_only=True).argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
        return seq
    dtype = next(model.parameters()).dtype
    cache = CatCache(c["L"]) if mode == "cat" else PreallocCache(c["L"], B, c["h_kv"], prompt.shape[1] + n_new, c["dk"], dtype)
    logits = model(seq, 0, cache, last_only=True)
    for i in range(n_new):
        nxt = logits.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        if i < n_new - 1:
            logits = model(nxt, seq.shape[1] - 1, cache, last_only=True)
    return seq
