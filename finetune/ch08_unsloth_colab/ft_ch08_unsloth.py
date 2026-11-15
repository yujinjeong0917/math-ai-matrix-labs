"""8장 Unsloth·Colab 계산 모형. 표준 라이브러리만 쓴다.

- config.json으로 파라미터 수 세기(5장과 같은 방식, 입력·출력 임베딩 공유 여부까지).
- bitsandbytes 4bit(NF4 + 두 번 양자화) 파일 크기: 양자화되는 선형층과 bf16으로 남는 나머지를 나눠 센다.
- LoRA 학습 파라미터 수: 닫힌 식과 텐서 하나씩 더하기.
- 학습 메모리 어림: 4bit 가중치 + LoRA 몫 + 활성값(5장 식, 체크포인팅) + 출력 칸(logits).
- 스텝 수: Trainer 방식 ceil(ceil(N / 묶음) / 누적).
- 채우기(padding)와 묶어 담기(packing)의 토큰 수, 에폭 시간 어림(FLOPs ÷ (최고 성능 × 가정한 활용률)).
- 1GB = 10^9바이트. 무작위 값을 쓰지 않는다.
"""

import json
import math
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"
SCENARIO_FILE = HERE / "data" / "scenario.json"
GB = 10**9
# GiB(2^30). experiments.py가 메모리 합계를 GiB로도 적을 때 쓰고, 이 모듈의 계산은 GB = 10^9 기준이다
GIB = 2**30

# LoRA를 붙일 층. 기본은 층 안의 모든 선형층 일곱 개
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------- 파일 읽기 ----------

def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def load_scenario():
    return json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))


def load_config(key, specs=None):
    specs = specs or load_specs()
    name = specs["models"][key]["config_file"]
    return json.loads((HERE / "data" / name).read_text(encoding="utf-8"))


# ---------- 파라미터 세기 ----------

def named_tensors(cfg):
    """Qwen2 모양 모델의 가중치 텐서를 (이름, 모양, 양자화되는 선형층인가)로 늘어놓는다."""
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    a, kv, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    d = h // a
    # 튜플 마지막 값은 '4비트로 양자화되는 선형층인가'. 임베딩, 편향, 정규화, lm_head는 양자화하지 않고 bf16으로 남는다
    out = [("model.embed_tokens.weight", (V, h), False)]
    for n in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{n}."
        out += [
            (p + "self_attn.q_proj.weight", (a * d, h), True), (p + "self_attn.q_proj.bias", (a * d,), False),
            (p + "self_attn.k_proj.weight", (kv * d, h), True), (p + "self_attn.k_proj.bias", (kv * d,), False),
            (p + "self_attn.v_proj.weight", (kv * d, h), True), (p + "self_attn.v_proj.bias", (kv * d,), False),
            (p + "self_attn.o_proj.weight", (h, a * d), True),
            (p + "mlp.gate_proj.weight", (i, h), True), (p + "mlp.up_proj.weight", (i, h), True),
            (p + "mlp.down_proj.weight", (h, i), True),
            (p + "input_layernorm.weight", (h,), False), (p + "post_attention_layernorm.weight", (h,), False),
        ]
    out.append(("model.norm.weight", (h,), False))
    if not cfg.get("tie_word_embeddings", False):
        # tie_word_embeddings가 거짓이면 lm_head를 따로 센다(양자화에서 제외, 코드 주석 참고)
        out.append(("lm_head.weight", (V, h), False))  # llm_int8_skip_modules에 lm_head가 있어 양자화하지 않는다
    return out


def count_params(cfg):
    return sum(math.prod(s) for _, s, _ in named_tensors(cfg))


def count_quantized_linear(cfg):
    return sum(math.prod(s) for _, s, q in named_tensors(cfg) if q)


def linear_shapes(cfg):
    """층 하나의 LoRA 대상 선형층 (이름, 입력 칸, 출력 칸)."""
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    a, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = h // a
    return [("q_proj", h, a * d), ("k_proj", h, kv * d), ("v_proj", h, kv * d), ("o_proj", a * d, h),
            ("gate_proj", h, i), ("up_proj", h, i), ("down_proj", i, h)]


# ---------- 4bit 파일 크기 ----------

def bits_per_param_double_quant(block=64, block2=256):
    """6장: 4비트 + 블록 상수 8비트/64 + 두 번째 상수 32비트/(64×256)."""
    # 두 번 양자화: 4 + 8/64 + 32/(64 x 256) = 4.126953...비트
    return F(4) + F(8, block) + F(32, block * block2)


def bnb4_bytes(cfg, block=64, block2=256):
    """NF4 + 두 번 양자화로 저장한 파일의 텐서 바이트(헤더·작은 상태값 제외).

    선형층 텐서마다: 4비트 칸 n/2바이트 + 블록 상수 ceil(n/64)바이트(8비트) + 두 번째 상수 ceil(블록 수/256)×4바이트.
    나머지(임베딩, 출력층, 정규화, 편향)는 bf16 2바이트.
    """
    packed = absmax = nested = rest = 0
    for _, shape, q in named_tensors(cfg):
        n = math.prod(shape)
        if q:
            # 텐서마다 64개씩 묶은 블록 수(나머지가 있으면 올림). 블록마다 8비트 상수 1개
            blocks = math.ceil(n / block)
            # 4비트 값 두 개를 한 바이트에 담으므로 n/2바이트(n이 짝수라는 가정)
            packed += n // 2
            absmax += blocks
            # 블록 상수 256개마다 fp32 상수(4바이트) 하나
            nested += math.ceil(blocks / block2) * 4
        else:
            # 양자화되지 않는 텐서는 bf16(2바이트)로 그대로
            rest += 2 * n
    return {"packed_4bit": packed, "absmax_8bit": absmax, "nested_fp32": nested, "bf16_rest": rest,
            "total": packed + absmax + nested + rest}


# ---------- LoRA ----------

def lora_params_closed(cfg, r, targets=TARGETS):
    per_layer = sum(din + dout for name, din, dout in linear_shapes(cfg) if name in targets)
    return r * per_layer * cfg["num_hidden_layers"]


def lora_tensors(cfg, r, targets=TARGETS):
    out = []
    for n in range(cfg["num_hidden_layers"]):
        for name, din, dout in linear_shapes(cfg):
            if name in targets:
                out.append((f"layers.{n}.{name}.lora_A", (r, din)))
                out.append((f"layers.{n}.{name}.lora_B", (dout, r)))
    return out


LORA_BYTES = {
    # LoRA 가중치를 fp32로 둔다고 가정: 가중치 4 + 기울기 4 + 옵티마이저 상태
    # 숫자 하나당 바이트: fp32 가중치 4 + 기울기 4 + 8비트 Adam 상태 2개 x 1 = 2. adamw_torch는 상태가 fp32 두 개(8)
    "adamw_8bit": 4 + 4 + 2,
    "adamw_torch": 4 + 4 + 8,
}


# ---------- 활성값과 출력 칸 (5장 식) ----------

def activation_ckpt_bytes(s, b, cfg):
    """5장: 층마다 입력만 남기는 체크포인팅 2sbhL + 다시 계산하는 한 층 sbh(34 + 5as/h). GPT 모양 가정의 추정."""
    h, a, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    # 5장 체크포인팅 추정식(층 입력 + 다시 계산하는 한 층). GPT 모양 가정의 어림
    return 2 * s * b * h * L + (34 * s * b * h + 5 * a * s * s * b)


def logits_bytes(s, b, cfg):
    """출력 칸: 토큰마다 어휘 수만큼 fp32 점수. 한 번에 다 만든다고 본 상한."""
    # 토큰마다 어휘 수만큼 fp32(4바이트) 점수를 한꺼번에 만든다고 본 상한. 실제로는 나눠 계산하거나 줄일 수 있다
    return s * b * cfg["vocab_size"] * 4


# ---------- 메모리 합계 ----------

def memory_parts(cfg, s, b, r, method="qlora", optim="adamw_8bit", gc_claim=None):
    """학습 메모리 어림(바이트). method: 'qlora'(4bit 얼린 가중치) | 'lora16'(16비트 얼린 가중치).

    gc_claim: Unsloth 체크포인팅 회사 주장(예: 0.30)을 활성값 몫에만 적용해 보는 값. None이면 적용하지 않는다.
    """
    if method == "qlora":
        weights = bnb4_bytes(cfg)["total"]
    elif method == "lora16":
        weights = 2 * count_params(cfg)
    else:
        raise ValueError(method)
    lora = lora_params_closed(cfg, r) * LORA_BYTES[optim]
    act = activation_ckpt_bytes(s, b, cfg)
    if gc_claim is not None:
        # 회사가 주장한 체크포인팅 절감 비율을 활성값 몫에만 곱해 본다. 주장이 맞다는 보장은 없다
        act = act * (1 - F(str(gc_claim)))
    logit = logits_bytes(s, b, cfg)
    return {"weights": weights, "lora": lora, "activations": act, "logits": logit,
            "total": weights + lora + act + logit}


# ---------- 스텝 수 ----------

def steps_per_epoch(n, b, g):
    """Trainer 방식: 미니배치 수 ceil(n/b)를 누적 g번마다 한 번 갱신, 끝의 모자란 묶음도 한 스텝."""
    # 미니배치 수 ceil(n/b)를 g개씩 모아 한 스텝. 끝에 모자라는 묶음도 한 스텝으로 센다
    return math.ceil(math.ceil(n / b) / g)


def examples_seen(max_steps, b, g):
    return max_steps * b * g


# ---------- 채우기와 묶어 담기 ----------

def expand_hist(hist):
    """{"길이": 개수} → 길이 목록(긴 것부터)."""
    out = []
    for k in sorted(hist, key=lambda x: -int(x)):
        # 히스토그램 {길이: 개수}를 길이 목록으로 펼친다. 키가 문자열이라 int로 바꾼다
        out += [int(k)] * hist[k]
    return out


def tokens_real(hist):
    return sum(int(k) * v for k, v in hist.items())


def tokens_pad_to_max(hist, s):
    return sum(hist.values()) * s


# b개를 독립으로 뽑을 때 최댓값의 기댓값: 각 길이 v에서 P(최댓값 = v) = F(v)^b - F(v 미만)^b (F는 누적분포). 분수로 정확히 계산
def expected_batch_max(hist, b):
    """같은 분포에서 b개를 독립으로 뽑았을 때 가장 긴 길이의 기댓값(분수, 정확). 묶음 안 최장 길이로 채우는 경우."""
    n = sum(hist.values())
    keys = sorted(int(k) for k in hist)
    e = F(0)
    cum_prev = F(0)
    for v in keys:
        cum = cum_prev + F(hist[str(v)], n)
        e += v * (cum**b - cum_prev**b)
        cum_prev = cum
    return e


def tokens_pad_to_batch_max(hist, b):
    return sum(hist.values()) * expected_batch_max(hist, b)


def pack_bfd(lengths, cap):
    """best-fit decreasing: 긴 것부터, 남는 자리가 가장 적게 남는 칸에 넣는다. 칸 개수와 칸별 채운 양."""
    bins = []
    for x in sorted(lengths, reverse=True):
        best = None
        for j, used in enumerate(bins):
            room = cap - used
            # 들어갈 수 있는 칸 중 남는 자리가 가장 작은 칸(best-fit)을 고른다. 동점이면 앞 칸
            if x <= room and (best is None or room < cap - bins[best]):
                best = j
        if best is None:
            bins.append(x)
        else:
            bins[best] += x
    return bins


def unsloth_packing_speedup(L, S):
    """Unsloth 글의 식: 긴 것 L과 짧은 것 S가 반씩일 때 2L/(L+S)."""
    # 긴 것 L과 짧은 것 S가 반반이면 채우기는 2L칸이 필요하고 묶어 담으면 (L+S)칸이면 되어 속도 비 2L/(L+S)
    return F(2 * L, L + S)


# ---------- 시간 어림 ----------

def flops_per_token(n_params, checkpoint=True):
    """얼린 기본 모델 + LoRA: 앞으로 2N, 거꾸로는 입력 쪽 기울기만 2N(얼린 가중치 기울기는 안 구함),
    체크포인팅이면 앞으로를 한 번 더 2N. 어텐션 점수 계산과 LoRA 몫은 뺀 어림."""
    # 토큰당 FLOPs 어림: 앞으로 2N + 입력 기울기 2N = 4N, 체크포인팅은 앞으로를 한 번 더 해 6N. 어텐션 점수와 LoRA 몫은 뺀 어림
    return (6 if checkpoint else 4) * n_params


def epoch_seconds(tokens, n_params, tflops, mfu, checkpoint=True):
    # 시간 = 전체 FLOPs / (최고 성능 x 활용률 mfu). mfu는 가정값이라 실제 측정이 아니다
    return F(tokens * flops_per_token(n_params, checkpoint)) / (F(tflops) * 10**12 * F(str(mfu)))


def mfu_from_runtime(tokens, n_params, tflops, seconds, checkpoint=True):
    """노트북에서 잰 train_runtime으로 활용률을 거꾸로 구한다."""
    return F(tokens * flops_per_token(n_params, checkpoint)) / (F(tflops) * 10**12 * F(str(seconds)))
