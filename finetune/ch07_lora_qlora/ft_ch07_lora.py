"""7장 LoRA와 QLoRA 계산기. 표준 라이브러리만 쓴다.

- LoRA 학습 파라미터 수: 선형층 하나(d_in → d_out)에 랭크 r을 붙이면 r(d_in + d_out)개.
- 어느 층에 붙이나: q,v(PEFT 기본) / q,k,v,o / MLP / 모든 선형층(all-linear).
- 배율 α/r(rsLoRA는 α/√r).
- QLoRA 메모리 사다리: 5장 미리 보기(모든 숫자 0.5바이트)에서 출발해, 4비트가 되는 선형층과
  원래 형식으로 남는 층을 나누고, 6장의 블록 상수, LoRA 몫, 활성값을 더한다.
- 작은 행렬 분해 장난감: 특이값을 아는 4×4 행렬로 낮은 랭크 근사, 거듭제곱법, 기울기 하강으로 B·A 학습.
- 1GB = 10^9바이트.
"""

import json
import math
import random
import struct
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"
GB = 10**9

TARGETS = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


# ---------- 사양표와 config ----------

def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def load_config(specs=None):
    specs = specs or load_specs()
    return json.loads((HERE / "data" / specs["model"]["config_file"]).read_text(encoding="utf-8"))


# ---------- 숫자 하나의 바이트 (4장 리캡) ----------

def to_bf16_bytes(x):
    """fp32로 담은 뒤 위쪽 16비트만 남긴다. 반올림 규칙: 아래 16비트를 버리는 자르기(0 쪽으로 버림).
    실제 라이브러리는 보통 가장 가까운 값으로 반올림(round-to-nearest-even)한다. 이 장은 바이트 수만 쓴다."""
    return struct.pack(">f", x)[:2]


def bytes_per_number(fmt):
    if fmt == "fp32":
        return len(struct.pack("<f", 1.0))
    if fmt == "fp16":
        return len(struct.pack("<e", 1.0))
    if fmt == "bf16":
        return len(to_bf16_bytes(1.0))
    raise ValueError(fmt)


# ---------- 선형층과 LoRA 파라미터 수 ----------

def linear_shapes(cfg):
    """Qwen2 모양 한 층의 선형층: 이름 -> (d_in, d_out)."""
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    d = h // cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"] * d
    return {
        "q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h),
        "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h),
    }


def lora_params_module(d_in, d_out, r):
    """A는 r × d_in, B는 d_out × r. 합 r(d_in + d_out)."""
    return r * d_in + d_out * r


def lora_params(cfg, targets, r):
    shapes = linear_shapes(cfg)
    per_layer = sum(lora_params_module(*shapes[m], r) for m in targets)
    return per_layer * cfg["num_hidden_layers"]


def lora_per_layer_coeff(cfg, targets):
    """r 하나당 한 층의 LoRA 파라미터 수(r의 계수)."""
    shapes = linear_shapes(cfg)
    return sum(a + b for a, b in (shapes[m] for m in targets))


def full_params_module(d_in, d_out):
    return d_in * d_out


def total_params(cfg):
    """5장과 같은 닫힌 식."""
    h, i, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    a, kv, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    d = h // a
    layer = 2 * h * h + 2 * h * kv * d + h + 2 * kv * d + 3 * h * i + 2 * h
    heads = 1 if cfg.get("tie_word_embeddings", False) else 2
    return heads * V * h + L * layer + h


def base_split(cfg):
    """4비트가 되는 것(층 안의 nn.Linear 가중치)과 원래 형식으로 남는 것(임베딩, lm_head, 편향, 정규화)."""
    L, h, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"]
    kv = cfg["num_key_value_heads"] * (h // cfg["num_attention_heads"])
    linear = L * sum(full_params_module(*s) for s in linear_shapes(cfg).values())
    embed = V * h
    head = 0 if cfg.get("tie_word_embeddings", False) else V * h
    bias = L * (h + 2 * kv)
    norms = L * 2 * h + h
    kept = embed + head + bias + norms
    assert linear + kept == total_params(cfg)
    return {"linear_4bit": linear, "embed": embed, "lm_head": head, "qkv_bias": bias, "norms": norms, "kept": kept,
            "total": linear + kept}


# ---------- 배율 ----------

def scaling(alpha, r, rslora=False):
    return F(alpha) / F(r) if not rslora else alpha / math.sqrt(r)


# ---------- 메모리 ----------

def nf4_bits(blocksize=64, const_bits=32, double_quant=False, second_block=256):
    """6장: 숫자 하나당 비트 = 4 + 상수 비트 ÷ 블록. 두 번 양자화면 8/64 + 32/(64·256)."""
    if double_quant:
        return 4 + F(8, blocksize) + F(32, blocksize * second_block)
    return 4 + F(const_bits, blocksize)


def lora_state_bytes_per_param(path):
    """학습하는 LoRA 숫자 하나당 바이트.

    peft_fp32: PEFT 기본(autocast_adapter_dtype=True)으로 어댑터가 fp32. 가중치 4 + 기울기 4 + Adam 4 + 4 = 16.
    paper_bf16: QLoRA 논문처럼 어댑터 bf16. 가중치 2 + 기울기 2 + Adam(fp32) 4 + 4 = 12.
    """
    return {"peft_fp32": 16, "paper_bf16": 12}[path]


def activation_ckpt_bytes(s, b, h, a, L):
    """5장과 같은 추정: 체크포인팅을 켜면 2sbhL + 한 층 몫 sbh(34 + 5as/h)."""
    return 2 * s * b * h * L + 34 * s * b * h + 5 * a * s * s * b


def qlora_ladder(cfg, r=16, targets="all_linear", double_quant=True, kept_bytes=2, lora_path="peft_fp32",
                 s=1024, b=1):
    """5장 QLoRA 미리 보기(모든 숫자 0.5바이트)에서 실제에 가까운 합계로 가는 사다리(바이트, 분수)."""
    sp = base_split(cfg)
    n = sp["total"]
    preview = F(n, 2)
    remove_kept = -F(sp["kept"], 2)
    kept = sp["kept"] * kept_bytes
    consts = sp["linear_4bit"] * (nf4_bits(double_quant=double_quant) - 4) / 8
    lp = lora_params(cfg, TARGETS[targets], r)
    lora = lp * lora_state_bytes_per_param(lora_path)
    act = activation_ckpt_bytes(s, b, cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"])
    steps = [
        ("5장 미리 보기: 모든 숫자 0.5바이트", preview),
        ("4비트가 안 되는 층을 빼기", remove_kept),
        ("그 층을 원래 형식으로 다시 더하기", F(kept)),
        ("6장 블록 상수", consts),
        ("LoRA 가중치·기울기·Adam 상태", F(lora)),
        ("5장 활성값 추정(체크포인팅)", F(act)),
    ]
    total = sum(v for _, v in steps)
    return {"steps": steps, "total": total, "lora_params": lp, "alpha_part": total - preview - act}


def lora_bf16_total(cfg, r, targets, lora_path="peft_fp32", s=1024, b=1):
    """LoRA(기본 모델 bf16로 얼림): 2Ψ + LoRA 몫 + 활성값."""
    n = total_params(cfg)
    lp = lora_params(cfg, TARGETS[targets], r)
    act = activation_ckpt_bytes(s, b, cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"])
    return {"base": 2 * n, "lora": lp * lora_state_bytes_per_param(lora_path), "act": act,
            "total": 2 * n + lp * lora_state_bytes_per_param(lora_path) + act, "lora_params": lp}


# ---------- 페이징 옵티마이저 장난감 ----------

def paged_window(fixed_bytes, optim_bytes, cap_bytes, h, a, L, b=1, lengths=range(256, 16385, 256)):
    """길이를 늘려 가며 합계(고정 몫 + 활성값)가 GPU에 들어가는지 본다.

    no_paging: 합계 ≤ GPU면 통과.
    paged: 옵티마이저 상태만큼은 CPU로 내보낼 수 있다고 보면 합계 ≤ GPU + 옵티마이저 상태까지 통과.
    단순 모형이다(내보낸 상태는 갱신할 때 다시 들여와야 하고, 다른 텐서는 내보내지 않는다고 가정).
    """
    last_fit = last_paged = None
    for s in lengths:
        tot = fixed_bytes + activation_ckpt_bytes(s, b, h, a, L)
        if tot <= cap_bytes:
            last_fit = s
        if tot <= cap_bytes + optim_bytes:
            last_paged = s
    return {"max_len_no_paging": last_fit, "max_len_paged": last_paged}


def paging_overhead_ms(evict_gb, link_gb_s=16, share=0.5):
    """bitsandbytes 문서의 어림: 내보내는 양 ÷ (대역폭 × 0.5)."""
    return 1000 * evict_gb / (link_gb_s * share)


# ---------- 작은 행렬 장난감 ----------

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def add(A, B, s=1):
    return [[a + s * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def scale(A, s):
    return [[s * a for a in row] for row in A]


def frob2(A):
    return sum(a * a for row in A for a in row)


def transpose(A):
    return [list(c) for c in zip(*A)]


HADAMARD = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]


def hadamard_col(j):
    """4×4 하다마르 행렬의 j번째 열을 길이 1로(÷2). 열끼리 서로 수직이다."""
    return [F(HADAMARD[i][j], 2) for i in range(4)]


def toy_matrix(sigmas=(F(3), F(1), F(1, 10)), u_cols=(0, 1, 2), v_cols=(1, 2, 3)):
    """특이값을 미리 아는 4×4 행렬 M = Σ σ_i u_i v_iᵀ (u, v는 하다마르 열)."""
    M = [[F(0)] * 4 for _ in range(4)]
    for s, ui, vi in zip(sigmas, u_cols, v_cols):
        M = add(M, scale(outer(hadamard_col(ui), hadamard_col(vi)), s))
    return M


def truncated(sigmas, r, u_cols=(0, 1, 2), v_cols=(1, 2, 3)):
    """큰 특이값 r개만 남긴 근사 M_r."""
    M = [[F(0)] * 4 for _ in range(4)]
    for s, ui, vi in list(zip(sigmas, u_cols, v_cols))[:r]:
        M = add(M, scale(outer(hadamard_col(ui), hadamard_col(vi)), s))
    return M


def power_iteration(M, steps=60, x0=None):
    """MᵀM의 가장 큰 고유값의 제곱근 = 가장 큰 특이값. 결정적인 시작 벡터를 쓴다."""
    Mf = [[float(a) for a in row] for row in M]
    MtM = matmul(transpose(Mf), Mf)
    x = list(x0 or [1.0, 0.5, 0.25, 0.125])
    for _ in range(steps):
        y = matvec(MtM, x)
        n = math.sqrt(sum(t * t for t in y))
        x = [t / n for t in y]
    lam = sum(a * b for a, b in zip(x, matvec(MtM, x)))
    return math.sqrt(lam), x


def train_lora_toy(target, r, steps=4000, lr=0.05, seed=7, init_std=0.1):
    """target(4×4)을 B(4×r)·A(r×4)로 맞추는 기울기 하강. LoRA처럼 A는 작은 무작위 값, B는 0에서 출발.

    손실 = ‖BA − target‖² 의 절반. 반환: 시작 손실, 끝 손실(‖·‖²), 처음 BA가 0이었는지.
    """
    rng = random.Random(seed)
    T = [[float(a) for a in row] for row in target]
    n, m = len(T), len(T[0])
    A = [[rng.gauss(0, init_std) for _ in range(m)] for _ in range(r)]
    B = [[0.0] * r for _ in range(n)]
    start_zero = all(v == 0 for row in matmul(B, A) for v in row)
    start = frob2(add(matmul(B, A), T, -1))
    for _ in range(steps):
        E = add(matmul(B, A), T, -1)          # n × m
        gB = matmul(E, transpose(A))           # n × r
        gA = matmul(transpose(B), E)           # r × m
        B = add(B, gB, -lr)
        A = add(A, gA, -lr)
    end = frob2(add(matmul(B, A), T, -1))
    return {"start_err2": start, "end_err2": end, "start_is_zero": start_zero}


def merged_equals_separate(W0, B, A, x, s):
    """(W0 + s·BA)x 와 W0x + s·B(Ax)가 같은지 분수로 확인."""
    merged = matvec(add(W0, scale(matmul(B, A), s)), x)
    sep = [a + s * b for a, b in zip(matvec(W0, x), matvec(B, matvec(A, x)))]
    return merged, sep


def random_sign_growth(rs=(4, 16, 64), d=256, trials=200, seed=11):
    """B, A의 칸이 ±1 무작위일 때 (BA)x 한 칸의 크기(제곱평균제곱근)가 r에 따라 어떻게 자라는지.

    x도 ±1. 같은 부호로 정렬되면 r·d에 비례하지만, 무작위면 √r·√d 쯤으로 자란다.
    """
    rng = random.Random(seed)
    out = {}
    for r in rs:
        acc = 0.0
        for _ in range(trials):
            x = [rng.choice((-1, 1)) for _ in range(d)]
            a_rows = [[rng.choice((-1, 1)) for _ in range(d)] for _ in range(r)]
            brow = [rng.choice((-1, 1)) for _ in range(r)]
            ax = [sum(p * q for p, q in zip(row, x)) for row in a_rows]
            y = sum(p * q for p, q in zip(brow, ax))
            acc += y * y
        out[r] = math.sqrt(acc / trials)
    return out
