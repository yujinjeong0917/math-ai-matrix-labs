"""10장 내보내고 띄우기 계산기. 표준 라이브러리만 쓴다.

- 어댑터 합치기: W' = W + (α/r)BA 를 분수로 손계산하고, 합친 표와 옆길 계산이 같은지 본다.
- 저장 크기: 어댑터를 따로 두면 기본 모델 1벌 + 어댑터 N개, 합치면 모델 N벌.
- 4비트로 학습한 어댑터 합치기 장난감: NF4(6장) 블록 64로 얼린 W 위에서 학습한 변화 sBA를
  (A) 따로 두기, (B) 4비트를 되돌린 W에 bf16으로 합치기, (C) 원래 bf16 W에 합치기,
  (D) 합친 뒤 다시 NF4로 담기, 네 가지로 비교한다.
- GGUF 파일 크기: ggml 블록 크기와 llama.cpp의 텐서별 형식 규칙(Q4_K_M 등)으로 Qwen2.5-7B의 크기를 계산한다.
- 띄우기: KV 캐시 토큰당 바이트, vLLM 메모리 예산, 메모리 대역폭으로 본 토큰/초 상한, 한 달 비용.
- 1GB = 10^9바이트.

bf16 반올림 규칙: 이 장의 합치기 장난감은 실제 라이브러리처럼 "가장 가까운 값, 딱 가운데면 짝수 쪽"
(round-to-nearest-even)으로 fp32 비트의 위쪽 16비트를 남긴다. 트랙 앞 장의 자르기(버림)도 같이 둬서 비교한다.
NF4 규칙은 6장과 같다: 블록 absmax는 fp32로 한 번 거치고, 표에서 가장 가까운 값을 고르며 거리가 같으면 앞(작은 값)을 고른다.
"""

import json
import math
import random
import struct
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"
SIZES_FILE = HERE / "data" / "gguf_sizes_2026-10-11.json"
SCENARIO_FILE = HERE / "data" / "scenario.json"
GB = 10**9

# NF4 표 16칸([-1, 1] 범위, 값은 아래 줄 끝 설명대로 6장 사양표의 nf4 표)
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]  # 6장 data/specs_2026-10-11.json의 bitsandbytes nf4 표


# ---------- 파일 읽기 ----------

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_specs():
    return load_json(SPEC_FILE)


def load_scenario():
    return load_json(SCENARIO_FILE)


def load_sizes():
    return load_json(SIZES_FILE)


def load_config(key="model", specs=None):
    specs = specs or load_specs()
    return load_json(HERE / "data" / specs[key]["config_file"])


# ---------- 숫자 형식 (4장 리캡) ----------

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_trunc(x):
    """fp32 비트의 위쪽 16비트만 남긴다(아래 16비트를 버림)."""
    # 빅엔디언 fp32 바이트에서 앞 2바이트만 남기고 뒤를 0으로 채운 값을 다시 fp32로 읽는다(위 16비트만 쓰는 bf16 자르기)
    b = struct.pack(">f", x)
    return struct.unpack(">f", b[:2] + b"\x00\x00")[0]


def bf16_rne(x):
    """fp32 비트의 위쪽 16비트로 반올림한다. 버리는 아래 16비트가 0x8000보다 크면 올리고,
    딱 0x8000이면 남는 마지막 비트가 짝수가 되게 한다(round-to-nearest-even). inf·nan은 다루지 않는다."""
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    # 버려질 아래 16비트. 0x8000이 정확히 가운데
    lower = u & 0xFFFF
    upper = u >> 16
    # 가운데보다 크면 올리고, 정확히 가운데면 위 16비트의 마지막 비트가 홀수일 때만 올려 짝수로 맞춘다(round-to-nearest-even)
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper += 1
    return struct.unpack(">f", struct.pack(">I", (upper & 0xFFFF) << 16))[0]


# ---------- 작은 행렬 도구 ----------

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def matvec(M, x):
    return [sum(a * b for a, b in zip(row, x)) for row in M]


def add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def scale(M, s):
    return [[s * v for v in row] for row in M]


def rms(vals):
    vals = list(vals)
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def flat(M):
    return [v for row in M for v in row]


# ---------- 1. 손으로 합치기 ----------

def merge_hand(W0, B, A, alpha, r, x):
    """W' = W0 + (α/r)BA. 분수로 계산해 합친 표로 곱한 값과 옆길로 따로 계산한 값을 함께 돌려준다."""
    # LoRA 배율 alpha/r. 분수로 정확히 계산
    s = F(alpha, r)
    W0 = [[F(v) for v in row] for row in W0]
    BA = matmul([[F(v) for v in row] for row in B], [[F(v) for v in row] for row in A])
    merged = add(W0, scale(BA, s))
    xs = [F(v) for v in x]
    via_merged = matvec(merged, xs)
    Ax = matvec([[F(v) for v in row] for row in A], xs)
    # 옆길 계산: W0 x + s B (A x). A x를 먼저 구해 작은 행렬만 곱한다. 합친 행렬 W'로 곱한 값과 정확히 같아야 한다(분수)
    side = [w + s * sum(b * a for b, a in zip(brow, Ax)) for w, brow in zip(matvec(W0, xs), B)]
    return {"s": s, "BA": BA, "merged": merged, "via_merged": via_merged, "via_side": side,
            "mults_merged": len(W0) * len(W0[0]),
            # 곱셈 횟수 비교: 합친 표는 W 크기만큼, 옆길은 W와 A, B 크기만큼 곱셈이 더 든다(이 손 예시의 셈)
            "mults_side": len(W0) * len(W0[0]) + len(A) * len(A[0]) + len(B) * len(B[0])}


def storage(n_adapters, base_gb, adapter_mb, merged_gb=None):
    """따로 두기: 기본 모델 1벌 + 어댑터 N개. 합치기: 모델 N벌(merged_gb는 합친 뒤 형식의 크기)."""
    merged_gb = base_gb if merged_gb is None else merged_gb
    return {"separate_gb": base_gb + n_adapters * adapter_mb / 1000, "merged_gb": n_adapters * merged_gb}


# ---------- 2. NF4 블록과 4비트 어댑터 합치기 장난감 ----------

def nf4_quant(row):
    # 한 행(여기서는 블록 하나)의 절댓값 최대를 fp32로 저장
    am = f32(max(abs(v) for v in row))
    # x / absmax에 가장 가까운 NF4 칸 번호. 거리가 같으면 번호가 작은 쪽(키의 둘째 원소)
    codes = [min(range(16), key=lambda k: (abs(v / am - NF4[k]), k)) for v in row]
    return am, codes


def nf4_dequant(am, codes):
    return [am * NF4[c] for c in codes]


def toy_setup(sc=None):
    sc = sc or load_scenario()["toy"]
    rng = random.Random(sc["seed"])
    R, C, r = sc["rows"], sc["cols"], sc["rank"]
    # 합성 가중치 W, LoRA 곱 B(R x r), A(r x C), 입력 x. 모두 시드 고정 정규분포. 실제 모델이 아니다
    W = [[rng.gauss(0, sc["w_std"]) for _ in range(C)] for _ in range(R)]
    B = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(R)]
    A = [[rng.gauss(0, 1) for _ in range(C)] for _ in range(r)]
    xs = [[rng.gauss(0, 1) for _ in range(C)] for _ in range(sc["n_inputs"])]
    return W, B, A, xs


def merge_toy(rho, sc=None):
    """rho = rms(sBA) ÷ rms(W). 반환값의 *_vs_effect는 '어댑터가 출력에 더하는 몫' 대비 어긋남,
    *_vs_output은 학습 때 출력 전체 대비 어긋남(둘 다 제곱평균제곱근 비율)."""
    sc = sc or load_scenario()["toy"]
    W, B, A, xs = toy_setup(sc)
    BA = matmul(B, A)
    # 배율 s를 정해 sBA의 크기가 W 크기의 rho배가 되게 한다(어댑터 변화량의 상대 크기 rho를 직접 조절)
    s = rho * rms(flat(W)) / rms(flat(BA))  # α/r에 학습된 크기까지 묶은 배율. rho를 맞추려고 정한다.
    D = scale(BA, s)
    Q = [nf4_quant(row) for row in W]
    Wd = [nf4_dequant(*q) for q in Q]
    # 기준(A): 학습 때와 같은 계산. 4비트를 되돌린 W와 어댑터를 따로 계산해 더한 출력
    ref = [[a + b for a, b in zip(matvec(Wd, x), matvec(D, x))] for x in xs]  # (A) 학습 때와 같은 계산
    effect = rms(v for x in xs for v in matvec(D, x))
    out = rms(v for row in ref for v in row)

    # B: 되돌린 W + sBA를 bf16으로 합침(반올림 두 가지), C: 원래 bf16 W에 합침, D: 합친 뒤 NF4로 다시 4비트화
    variants = {
        "B_dequant_bf16": [[bf16_rne(a + b) for a, b in zip(r1, r2)] for r1, r2 in zip(Wd, D)],
        "B_dequant_bf16_trunc": [[bf16_trunc(a + b) for a, b in zip(r1, r2)] for r1, r2 in zip(Wd, D)],
        "C_original_bf16": [[bf16_rne(a + b) for a, b in zip(r1, r2)] for r1, r2 in zip(W, D)],
    }
    # D: 합친 값을 다시 양자화하면 작은 변화는 같은 칸에 들어가 사라질 수 있다
    Qd = [nf4_quant([a + b for a, b in zip(r1, r2)]) for r1, r2 in zip(Wd, D)]
    variants["D_requant_nf4"] = [nf4_dequant(*q) for q in Qd]

    res = {"rho": rho, "s": s, "effect_rms": effect, "output_rms": out}
    for name, M in variants.items():
        # 각 방식의 출력과 기준 출력의 차이를 제곱평균제곱근으로 잰다. vs_effect는 어댑터 몫 대비, vs_output은 출력 전체 대비
        diff = rms(u - v for x, rf in zip(xs, ref) for u, v in zip(matvec(M, x), rf))
        res[name] = {"vs_effect": diff / effect, "vs_output": diff / out}
    n = sc["rows"] * sc["cols"]
    # 다시 양자화했을 때 NF4 코드가 그대로인 칸 수. 많을수록 어댑터 변화가 사라졌다는 뜻
    res["codes_unchanged"] = sum(c1 == c2 for (_, a), (_, b) in zip(Q, Qd) for c1, c2 in zip(a, b))
    res["codes_total"] = n
    res["quant_err_vs_w"] = rms(a - b for r1, r2 in zip(W, Wd) for a, b in zip(r1, r2)) / rms(flat(W))
    # 0 근처 칸 간격의 절반(블록 absmax 평균 × NF4 0과 바로 옆 양수 칸 사이의 절반)
    # 0 근처 칸 간격의 절반 근사: 평균 absmax x NF4[8] / 2. 변화가 이보다 작으면 같은 칸에 남기 쉽다
    res["half_gap_near0"] = sum(q[0] for q in Q) / len(Q) * NF4[8] / 2
    res["delta_abs_mean"] = sum(abs(v) for v in flat(D)) / n
    return res


# ---------- 3. GGUF 파일 크기 ----------

# 파일 형식 이름 -> 기본 텐서 형식. 텐서마다 달라지는 규칙은 tensor_type()에 있다
BASE_TYPE = {"Q2_K": "Q2_K", "Q3_K_M": "Q3_K", "Q4_0": "Q4_0", "Q4_K_M": "Q4_K", "Q5_0": "Q5_0",
             "Q5_K_M": "Q5_K", "Q6_K": "Q6_K", "Q8_0": "Q8_0", "F16": "F16"}


def use_more_bits(i, n):
    """llama.cpp llama-quant.cpp의 람다를 그대로 옮겼다(C++ 정수 나눗셈과 %)."""
    # llama.cpp 람다를 옮긴 것. 앞쪽 1/8, 뒤쪽 1/8 층과 그 사이 세 층에 한 번꼴로 더 많은 비트(Q6_K)를 쓴다. C++ 정수 나눗셈과 같게 //를 쓴다
    q = n // 8
    return i < q or i >= 7 * n // 8 or (i - q) % 3 == 2


# 텐서 분류(cat)와 층 번호로 형식을 고르는 규칙. 이 규칙은 llama.cpp의 소스를 읽고 옮긴 것이므로 버전이 바뀌면 달라질 수 있다(확인 필요)
def tensor_type(ftype, cat, i_layer, n_layer, n_gqa):
    """Qwen2 모양(전문가 없음, 70B 아님, Falcon 아님, imatrix 없음)에서 llama.cpp가 고르는 형식."""
    base = BASE_TYPE[ftype]
    if ftype == "F16":
        return "F16"
    if cat == "output":
        return base if base == "Q8_0" else "Q6_K"
    if cat == "token_embd":
        return base
    if cat == "attn_v":
        if ftype == "Q2_K":
            return "Q4_K" if n_gqa >= 4 else "Q3_K"
        if ftype == "Q3_K_M":
            return "Q5_K" if i_layer < 2 else "Q4_K"
        if ftype in ("Q4_K_M", "Q5_K_M") and use_more_bits(i_layer, n_layer):
            return "Q6_K"
        return base
    if cat == "ffn_down":
        if ftype == "Q2_K":
            return "Q3_K"
        if ftype == "Q3_K_M":
            return "Q5_K" if i_layer < n_layer // 16 else "Q4_K"
        if ftype in ("Q4_K_M", "Q5_K_M") and use_more_bits(i_layer, n_layer):
            return "Q6_K"
        return base
    if cat == "attn_output":
        if ftype == "Q2_K":
            return "Q3_K"
        if ftype == "Q3_K_M":
            return "Q4_K"
        return base
    return base


def tensors(cfg):
    """(이름 분류, 층 번호, 숫자 개수, 첫 차원). 1차원 텐서(편향, 정규화)는 'f32'로 따로 센다."""
    h, i, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    kv = cfg["num_key_value_heads"] * (h // cfg["num_attention_heads"])
    out = [("token_embd", -1, V * h, h)]
    if not cfg.get("tie_word_embeddings", False):
        out.append(("output", -1, V * h, h))
    for il in range(L):
        out += [("attn_q", il, h * h, h), ("attn_k", il, h * kv, h), ("attn_v", il, h * kv, h),
                ("attn_output", il, h * h, h), ("ffn_gate", il, h * i, h), ("ffn_up", il, h * i, h),
                ("ffn_down", il, i * h, i)]
        # q, k, v 편향과 정규화 두 개는 1차원이라 양자화하지 않고 fp32로 센다
        out += [("f32", il, h + 2 * kv, 1), ("f32", il, 2 * h, 1)]  # q·k·v 편향, 정규화 두 개
    out.append(("f32", -1, h, 1))  # 마지막 정규화
    return out


def type_bytes(t, n, blocks):
    per, nbytes = blocks[t]
    # 형식의 블록 크기로 숫자 개수가 나누어떨어져야 한다(안 나누어지면 계산이 틀린다)
    assert n % per == 0, (t, n)
    return n // per * nbytes


def gguf_tensor_bytes(cfg, ftype, blocks):
    L = cfg["num_hidden_layers"]
    n_gqa = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
    total, by_type = 0, {}
    for cat, il, n, ne0 in tensors(cfg):
        t = "F32" if cat == "f32" else tensor_type(ftype, cat, il, L, n_gqa)
        # 블록은 텐서의 첫 차원 방향으로 묶이므로 첫 차원이 블록 크기의 배수여야 그 형식을 쓸 수 있다
        assert ne0 % blocks[t][0] == 0, (cat, t)
        b = type_bytes(t, n, blocks)
        total += b
        by_type[t] = by_type.get(t, 0) + n
    return total, by_type


def bits_per_weight(nbytes, n_params):
    return nbytes * 8 / n_params


def q6k_layers(ftype, cat, n_layer=28, n_gqa=7):
    return [il for il in range(n_layer) if tensor_type(ftype, cat, il, n_layer, n_gqa) == "Q6_K"]


def gguf_table(cfg, blocks, actual):
    """형식별 계산 크기와 실제 파일 크기. 메타데이터 몫은 F16 파일에서 텐서 몫을 뺀 값으로 잰다."""
    n = sum(t[2] for t in tensors(cfg))
    f16_tensors, _ = gguf_tensor_bytes(cfg, "F16", blocks)
    # 메타데이터(어휘 등) 몫 = 실제 F16 파일 크기 - 계산한 텐서 바이트. 다른 형식에도 같은 몫이 든다고 가정한다
    meta = sum(actual["F16"]) - f16_tensors
    rows = {}
    for ft in ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]:
        tb, by_type = gguf_tensor_bytes(cfg, ft, blocks)
        a = sum(actual[ft])
        rows[ft] = {"tensor_bytes": tb, "pred_bytes": tb + meta, "actual_bytes": a, "diff_bytes": a - (tb + meta),
                    "bpw": bits_per_weight(tb, n), "pred_gb": (tb + meta) / GB, "by_type": by_type,
                    "shards": len(actual[ft])}
    return {"n_params": n, "meta_bytes": meta, "rows": rows}


# ---------- 4. 띄우기: KV 캐시, 메모리, 속도, 비용 ----------

def kv_bytes_per_token(cfg, bytes_per=2):
    d = cfg["hidden_size"] // cfg["num_attention_heads"]
    # 토큰 하나의 KV 캐시 = 2(K, V) x 층 x 키·값 헤드 x 헤드 차원 x 바이트(기본 2바이트)
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * d * bytes_per


def paged_tokens(n_tokens, block):
    # 블록 단위로 할당하므로 토큰 수를 블록 크기의 배수로 올린다
    return block * math.ceil(n_tokens / block)


def vllm_budget(gpu_gb, util, weight_bytes, kv_per_token, req_tokens, block, adapters_bytes=0):
    """가중치와 어댑터를 빼고 남는 자리를 모두 KV 캐시로 본 상한. 활성값·CUDA 그래프 몫은 빼지 않았다."""
    usable = gpu_gb * GB * util
    # 사용 가능 메모리 = GPU 용량 x 활용 비율. 가중치와 어댑터를 빼고 남은 자리를 모두 KV 캐시로 본 상한(활성값 등은 뺀 어림)
    left = usable - weight_bytes - adapters_bytes
    if left <= 0:
        return {"usable_gb": usable / GB, "left_gb": left / GB, "fits": False, "kv_tokens": 0, "requests": 0}
    kv_tokens = int(left // kv_per_token)
    per_req = paged_tokens(req_tokens, block)
    return {"usable_gb": usable / GB, "left_gb": left / GB, "fits": True, "kv_tokens": kv_tokens,
            "per_request_tokens": per_req, "requests": kv_tokens // per_req}


def tokens_per_s_upper(bandwidth_gb_s, weight_bytes, batch=1):
    """토큰 하나를 만들 때마다 가중치를 한 번 다 읽는다고 본 상한. 묶음 B면 한 번 읽어 B개를 만든다."""
    # 토큰 하나마다 가중치를 한 번 읽는다고 본 상한(초당 토큰). 실제 속도는 이보다 낮다
    return bandwidth_gb_s * GB / weight_bytes * batch


def monthly_cost(calls, answer_tokens, tok_s, usd_per_hour, hours_per_month):
    tokens = calls * answer_tokens
    # 실제로 생성만 하는 시간(시간 단위). 상시 켜 두면 hours_per_month 전부 청구된다
    busy_h = tokens / tok_s / 3600
    return {"tokens": tokens, "busy_hours": busy_h, "always_on_usd": usd_per_hour * hours_per_month,
            "busy_only_usd": usd_per_hour * busy_h, "busy_share": busy_h / hours_per_month,
            "always_on_per_call_usd": usd_per_hour * hours_per_month / calls}


def usd_per_million_tokens(usd_per_hour, tok_s):
    return usd_per_hour / (tok_s * 3600) * 1e6


# ---------- 예시 파일 읽기 ----------

def parse_modelfile(text):
    """줄 맨 앞 낱말(대소문자 무시)을 명령으로 본다. 세 따옴표 안의 줄은 앞 명령에 붙인다."""
    # Modelfile을 줄 단위로 읽는다. 세 따옴표(""") 안은 앞 명령의 값에 줄바꿈으로 이어 붙인다. 이 장 예시 파일에 필요한 만큼만 읽는 단순 파서
    cmds, inside = [], False
    for line in text.splitlines():
        if inside:
            cmds[-1][1] += "\n" + line
            if line.rstrip().endswith('"""'):
                inside = False
            continue
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        word, _, rest = s.partition(" ")
        cmds.append([word.upper(), rest])
        # 한 줄에 세 따옴표가 한 번만 있으면 여러 줄 값이 시작된 것
        if rest.count('"""') == 1:
            inside = True
    return cmds


def card_front_matter(text):
    """--- 사이의 간단한 YAML(키: 값, - 목록)만 읽는다."""
    lines = text.splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    meta, key = {}, None
    for line in lines[1:end]:
        if line.startswith("- ") and key:
            meta.setdefault(key, [])
            if not isinstance(meta[key], list):
                meta[key] = []
            meta[key].append(line[2:].strip())
        else:
            key, _, val = line.partition(":")
            key = key.strip()
            meta[key] = val.strip() if val.strip() else []
    return meta


def nf4_cell_hand(absmax, w, delta):
    """블록 기준값 absmax인 숫자 w 하나를 NF4로 담고, 되돌린 값에 delta를 더해 다시 담는 손계산."""
    # w를 absmax로 나눠 가장 가까운 NF4 칸을 고른 뒤, 되돌린 값에 delta를 더해 다시 담았을 때 칸이 바뀌는지 본다
    k = min(range(16), key=lambda j: (abs(w / absmax - NF4[j]), j))
    back = absmax * NF4[k]
    k2 = min(range(16), key=lambda j: (abs((back + delta) / absmax - NF4[j]), j))
    # 위, 아래 이웃 칸까지 간격의 절반. delta가 이보다 작으면 칸이 안 바뀌어 사라진다(erased)
    up = absmax * (NF4[k + 1] - NF4[k]) / 2 if k < 15 else None
    down = absmax * (NF4[k] - NF4[k - 1]) / 2 if k > 0 else None
    return {"code": k, "table_value": NF4[k], "back": back, "code_after": k2, "erased": k2 == k,
            "half_gap_up": up, "half_gap_down": down}
