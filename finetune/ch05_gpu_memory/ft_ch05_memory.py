"""5장 GPU 메모리 계산기. 표준 라이브러리만 쓴다.

- 숫자 하나의 바이트 수: fp32는 struct 'f', fp16은 struct 'e', bf16은 fp32 비트의 위쪽 16비트.
- config.json으로 파라미터 수 세기: 닫힌 식, 텐서 하나씩 더하기.
- 학습 방식별 파라미터 하나당 바이트(사양표의 recipes).
- 활성값: Korthikanti 외(2022) 식 (1)과 그 항목별 내역. GPT 모양 블록을 가정한 근사다.
- 1GB = 10^9바이트.
"""

import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"
SCENARIO_FILE = HERE / "data" / "scenario.json"
GB = 10**9


# ---------- 숫자 하나의 바이트 수 (4장 리캡) ----------

def to_bf16_bytes(x):
    """fp32로 담은 뒤 위쪽 16비트(부호 1, 지수 8, 가수 위 7비트)만 남긴다.

    반올림 규칙: 아래 16비트를 그냥 버리는 '자르기'(0 쪽으로 버림)다. 실제 라이브러리는 보통
    가장 가까운 값으로 반올림(round-to-nearest-even)하지만, 이 장은 바이트 수만 재므로 자르기로 충분하다.
    """
    return struct.pack(">f", x)[:2]


def pack_numbers(values, fmt):
    """숫자 목록을 형식대로 바이트로 담는다. fmt: 'fp32' | 'fp16' | 'bf16'."""
    if fmt == "fp32":
        return struct.pack(f"<{len(values)}f", *values)
    if fmt == "fp16":
        return struct.pack(f"<{len(values)}e", *values)
    if fmt == "bf16":
        return b"".join(to_bf16_bytes(v) for v in values)
    raise ValueError(fmt)


def bytes_per_number(fmt):
    return len(pack_numbers([1.0], fmt))


# ---------- 사양표와 config ----------

def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def load_scenario():
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


def load_config(specs=None):
    specs = specs or load_specs()
    return json.loads((HERE / "data" / specs["model"]["config_file"]).read_text(encoding="utf-8"))


def named_tensors(cfg):
    """Qwen2 모양 모델의 가중치 텐서를 이름과 모양으로 하나씩 늘어놓는다."""
    h = cfg["hidden_size"]
    i = cfg["intermediate_size"]
    a = cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    d = h // a
    out = [("model.embed_tokens.weight", (V, h))]
    for n in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{n}."
        out += [
            (p + "self_attn.q_proj.weight", (a * d, h)), (p + "self_attn.q_proj.bias", (a * d,)),
            (p + "self_attn.k_proj.weight", (kv * d, h)), (p + "self_attn.k_proj.bias", (kv * d,)),
            (p + "self_attn.v_proj.weight", (kv * d, h)), (p + "self_attn.v_proj.bias", (kv * d,)),
            (p + "self_attn.o_proj.weight", (h, a * d)),
            (p + "mlp.gate_proj.weight", (i, h)), (p + "mlp.up_proj.weight", (i, h)),
            (p + "mlp.down_proj.weight", (h, i)),
            (p + "input_layernorm.weight", (h,)), (p + "post_attention_layernorm.weight", (h,)),
        ]
    out.append(("model.norm.weight", (h,)))
    if not cfg.get("tie_word_embeddings", False):
        out.append(("lm_head.weight", (V, h)))
    return out


def count_by_tensors(cfg):
    return sum(math.prod(shape) for _, shape in named_tensors(cfg))


def count_closed_form(cfg):
    """한 층 = 어텐션(4개 행렬 + q·k·v 편향) + MLP(행렬 3개) + 정규화 2개."""
    h, i, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    a, kv, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    d = h // a
    attn = 2 * h * h + 2 * h * kv * d + h + 2 * kv * d
    mlp = 3 * h * i
    layer = attn + mlp + 2 * h
    heads = 1 if cfg.get("tie_word_embeddings", False) else 2
    return {"per_layer": layer, "attn": attn, "mlp": mlp, "embedding": V * h,
            "total": heads * V * h + L * layer + h, "non_embedding": L * layer + h}


# ---------- 학습 방식별 바이트 ----------

def recipe_bytes(recipe):
    """파라미터 하나당 바이트(분수)."""
    return sum(F(str(v)) for v in recipe["parts"].values())


def model_state_gb(n_params, recipe):
    return {k: F(str(v)) * n_params / GB for k, v in recipe["parts"].items()}


# ---------- 활성값 (Korthikanti 외 2022, 식 (1)) ----------

def activation_parts(s, b, h, a):
    """논문 4.1절의 항목별 내역(바이트). 2바이트 활성값, 1바이트 드롭아웃 마스크."""
    sbh, as2b = s * b * h, a * s * s * b
    return {
        "attn.qkv_input": 2 * sbh,
        "attn.QK^T(Q,K)": 4 * sbh,
        "attn.softmax_out": 2 * as2b,
        "attn.softmax_dropout_mask": as2b,
        "attn.over_V(dropout_out)": 2 * as2b,
        "attn.over_V(V)": 2 * sbh,
        "attn.proj_input": 2 * sbh,
        "attn.dropout_mask": sbh,
        "mlp.linear1_input": 2 * sbh,
        "mlp.linear2_input": 8 * sbh,
        "mlp.gelu_input": 8 * sbh,
        "mlp.dropout_mask": sbh,
        "layernorm.inputs(2개)": 4 * sbh,
    }


def activation_per_layer(s, b, h, a, keep_scores=True):
    """sbh(34 + 5as/h). keep_scores=False면 s²에 비례하는 5as²b 항을 뺀다(점수 표를 저장하지 않는 어텐션 가정)."""
    return 34 * s * b * h + (5 * a * s * s * b if keep_scores else 0)


def activation_total(s, b, h, a, L, keep_scores=True, checkpoint=False):
    per = activation_per_layer(s, b, h, a, keep_scores)
    if not checkpoint:
        return L * per
    # 논문: 층마다 입력만 저장하면 2sbhL. 거꾸로 갈 때 한 층을 다시 계산하는 동안 그 층 몫이 잠깐 더 든다.
    return 2 * s * b * h * L + per


# ---------- 체크포인팅 흉내 ----------

def simulate_checkpointing(L, k, input_bytes, layer_bytes):
    """층 L개 사슬을 앞으로 한 번, 거꾸로 한 번 지나가며 살아 있는 메모리를 센다.

    k=None: 모든 층의 활성값을 저장한다.
    k=정수: 0, k, 2k, ...번째 층의 입력만 저장하고, 거꾸로 갈 때 구간(k층)을 다시 계산해 그 구간의 활성값을 잠깐 든다.
    반환: 최대 메모리(바이트), 다시 계산한 층 수.
    """
    live = peak = 0
    recomputed = 0
    if k is None:
        for _ in range(L):
            live += layer_bytes
            peak = max(peak, live)
        for _ in range(L):
            live -= layer_bytes
        return {"peak": peak, "recomputed_layers": 0}
    starts = list(range(0, L, k))
    for _ in starts:  # 앞으로: 체크포인트 입력만 남긴다
        live += input_bytes
        peak = max(peak, live)
    for st in reversed(starts):  # 거꾸로: 구간을 다시 계산해 활성값을 들고, 기울기를 구한 뒤 버린다
        seg = range(st, min(st + k, L))
        for _ in seg:
            live += layer_bytes
            recomputed += 1
            peak = max(peak, live)
        for _ in seg:
            live -= layer_bytes
        live -= input_bytes
    assert live == 0
    return {"peak": peak, "recomputed_layers": recomputed}


# ---------- 기울기 누적 장난감 ----------

def grad_mse(w, batch):
    """y ≈ w·x 의 평균 제곱 오차를 w로 미분한 값: 평균(2(wx − y)x)."""
    return sum(2 * (w * x - y) * x for x, y in batch) / len(batch)


def grad_accumulated(w, batch, steps, divide=True):
    size = len(batch) // steps
    g = F(0)
    for j in range(steps):
        part = grad_mse(w, batch[j * size:(j + 1) * size])
        g += part / steps if divide else part
    return g


def fits(total_gb, gpu_gb):
    return total_gb <= gpu_gb
