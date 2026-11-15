"""내 데이터로 LLM 파인튜닝하기 5장 검증. `uv run pytest -q finetune/ch05_gpu_memory` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다.
"""

import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path

import pytest

import ft_ch05_memory as M

HERE = Path(__file__).parent


# ---------- 사양표 ----------

def test_spec_file_has_source_and_date_everywhere():
    s = M.load_specs()
    assert s["checked"] == "2026-10-11" and M.SPEC_FILE.name == "specs_2026-10-11.json"
    for rec in s["recipes"].values():
        assert rec["url"].startswith("https://") and rec["checked"] == "2026-10-11"
    for g in s["gpus"]:
        assert g["url"].startswith("https://") and g["checked"] == "2026-10-11"
    for key in ("model", "activation", "colab", "pytorch_memory", "hf_training", "checkpointing_paper"):
        assert s[key]["checked"] == "2026-10-11"
    # Colab 무료 GPU 종류는 바뀔 수 있어 사양표에도 확인 필요로 적혀 있어야 한다
    assert "[확인 필요]" in s["colab"]["free_gpu_type"]


# ---------- 바이트 수 (4장 리캡) ----------

def test_bytes_per_number_from_struct():
    assert struct.calcsize("f") == 4 and struct.calcsize("e") == 2
    assert M.bytes_per_number("fp32") == 4
    assert M.bytes_per_number("fp16") == 2
    assert M.bytes_per_number("bf16") == 2


def test_bf16_is_top_half_of_fp32_by_truncation():
    # 1.0 = 0x3F800000 -> 0x3F80 ; 아래 16비트를 버리는 자르기
    # 1.0 = 0x3F800000이므로 위 16비트는 0x3F80
    assert M.to_bf16_bytes(1.0) == b"\x3f\x80"
    x = struct.unpack(">f", bytes.fromhex("3f80ffff"))[0]
    assert M.to_bf16_bytes(x) == b"\x3f\x80"  # 반올림하지 않고 잘라낸다


def test_first_screen_hand_numbers():
    nums = [i / 1000 for i in range(1000)]
    assert len(M.pack_numbers(nums, "fp32")) == 4000
    assert len(M.pack_numbers(nums, "fp16")) == 2000
    assert len(M.pack_numbers(nums, "bf16")) == 2000
    n = 10**9
    # 10억 개 x 2바이트 = 2GB(추론), x 16 = 16GB, x 18 = 18GB(학습 상태)
    assert 2 * n / M.GB == 2 and 16 * n / M.GB == 16 and 18 * n / M.GB == 18


# ---------- 파라미터 수 ----------

def test_param_count_matches_safetensors_and_card():
    s = M.load_specs()
    cfg = M.load_config(s)
    closed = M.count_closed_form(cfg)
    assert closed["total"] == M.count_by_tensors(cfg) == s["model"]["safetensors_total"] == 7_615_616_512
    # 7.6156...을 반올림하면 7.62지만 모델 카드는 7.61로 적었다(버림). 총합은 safetensors 헤더 값과 같다
    assert round(closed["total"] / 1e9, 2) == 7.62 and f"{closed['total'] / 1e9:.2f}".startswith("7.6")
    assert s["model"]["card_params"] == "7.61B"  # 카드는 7.6156...을 7.61로 적었다(버림)
    assert round(closed["non_embedding"] / 1e9, 2) == 6.53
    # 텐서 수 = 임베딩 1 + 층당 12개 x 28층 + 최종 정규화 1 + lm_head 1 = 339
    assert len(M.named_tensors(cfg)) == 1 + 28 * 12 + 1 + 1


def test_config_shape():
    cfg = M.load_config()
    assert (cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]) == (3584, 28, 28)
    assert cfg["num_key_value_heads"] == 4 and cfg["vocab_size"] == 152064
    assert cfg["tie_word_embeddings"] is False and cfg["torch_dtype"] == "bfloat16"


# ---------- 학습 상태 ----------

# 학습 방식별 파라미터 하나당 바이트(사양표의 값). 4비트 동결 가중치는 0.5바이트
@pytest.mark.parametrize("key,b", [("zero_mixed_adam", 16), ("hf_mixed_adam", 18), ("hf_mixed_adam8bit", 12),
                                   ("infer_bf16", 2), ("frozen_bf16", 2), ("frozen_4bit", F(1, 2))])
def test_recipe_bytes(key, b):
    assert M.recipe_bytes(M.load_specs()["recipes"][key]) == b


def test_state_gb_for_qwen():
    s = M.load_specs()
    n = M.count_closed_form(M.load_config(s))["total"]
    gb = {k: float(M.recipe_bytes(v) * n / M.GB) for k, v in s["recipes"].items()}
    assert round(gb["infer_bf16"], 2) == 15.23
    # 7.6156B개 x 16바이트 = 121.85GB. 18바이트면 137.08GB
    assert round(gb["zero_mixed_adam"], 2) == 121.85
    assert round(gb["hf_mixed_adam"], 2) == 137.08
    assert round(gb["hf_mixed_adam8bit"], 2) == 91.39
    assert round(gb["frozen_4bit"], 2) == 3.81


# ---------- 활성값 ----------

def test_activation_formula_equals_paper_breakdown():
    for s, b, h, a in [(1024, 1, 3584, 28), (2048, 4, 4096, 32), (7, 3, 64, 4)]:
        parts = M.activation_parts(s, b, h, a)
        # 논문 항목별 내역의 합이 한 줄 식 sbh(34 + 5as/h)와 같아야 한다. 어텐션 11 + MLP 19 + 정규화 4 = 34
        assert sum(parts.values()) == M.activation_per_layer(s, b, h, a)
        sbh, as2b = s * b * h, a * s * s * b
        assert sum(v for k, v in parts.items() if k.startswith("attn")) == 11 * sbh + 5 * as2b
        assert sum(v for k, v in parts.items() if k.startswith("mlp")) == 19 * sbh
        assert sum(v for k, v in parts.items() if k.startswith("layernorm")) == 4 * sbh


def test_activation_numbers_for_project():
    h, a, L, s = 3584, 28, 28, 1024
    per = M.activation_per_layer(s, 1, h, a)
    # s b h = 1024 x 3584 = 3,670,016. 34 + 5 x 28 x 1024 / 3584 = 34 + 40 = 74
    assert per == 3_670_016 * 74 == 271_581_184
    assert round(L * per / M.GB, 3) == 7.604
    assert round(M.activation_total(s, 4, h, a, L) / M.GB, 2) == 30.42
    assert round(2 * s * h * L / M.GB, 4) == 0.2055
    assert round(M.activation_total(s, 1, h, a, L, checkpoint=True) / M.GB, 4) == 0.4771
    # 묶음 크기에는 정비례, 길이에는 s항과 s²항이 섞인다
    assert M.activation_total(s, 4, h, a, L) == 4 * M.activation_total(s, 1, h, a, L)
    # 길이가 두 배면 s에 비례하는 항은 2배, s^2 항은 4배라 총합은 2배보다 크다
    assert M.activation_total(2 * s, 1, h, a, L) > 2 * M.activation_total(s, 1, h, a, L)


# ---------- 체크포인팅 흉내 ----------

@pytest.mark.parametrize("L,k", [(28, 1), (28, 2), (28, 4), (28, 7), (10, 3), (5, 5)])
def test_checkpoint_sim_peak_closed_form(L, k):
    inp, lay = 2, 34
    sim = M.simulate_checkpointing(L, k, inp, lay)
    starts = list(range(0, L, k))
    # 거꾸로 j번째 구간을 다시 계산할 때: 아직 남은 체크포인트 (j + 1)개 + 그 구간의 층 활성값
    # 거꾸로 j번째 구간을 다시 계산할 때: 아직 남은 체크포인트 (j+1)개 + 그 구간의 층 활성값(마지막 구간은 짧을 수 있다)
    expect = max((j + 1) * inp + (min(st + k, L) - st) * lay for j, st in enumerate(starts))
    assert sim["peak"] == expect
    if L % k == 0:
        assert sim["peak"] == (L // k) * inp + k * lay
    # 거꾸로 가며 모든 층을 한 번씩 다시 계산한다
    assert sim["recomputed_layers"] == L


def test_checkpoint_sim_equals_formula():
    h, a, L, s = 3584, 28, 28, 1024
    inp, lay = 2 * s * h, M.activation_per_layer(s, 1, h, a)
    assert M.simulate_checkpointing(L, 1, inp, lay)["peak"] == M.activation_total(s, 1, h, a, L, checkpoint=True)
    assert M.simulate_checkpointing(L, None, inp, lay)["peak"] == M.activation_total(s, 1, h, a, L)


def test_equal_units_follow_sqrt_rule():
    # 입력과 층 활성값의 크기가 같으면 ceil(L/k) + k가 가장 작은 k는 √L 근처
    L = 36
    peaks = {k: M.simulate_checkpointing(L, k, 1, 1)["peak"] for k in range(1, L + 1)}
    low = min(peaks.values())
    # L=36이면 k=6(=sqrt(36))에서 6 + 6 = 12로 최소. k=1이면 36 + 1 = 37, k=36이면 1 + 36 = 37
    assert low == 12 and peaks[6] == 12 and peaks[1] == 37 and peaks[36] == 37


# ---------- 기울기 누적 ----------

def test_grad_accumulation_equals_big_batch():
    batch = [(F(1), F(2)), (F(2), F(3)), (F(3), F(7)), (F(4), F(8))]
    full = M.grad_mse(F(1), batch)
    # w=1에서 4개 평균: 2(1x1-2)x1=-2, 2(2-3)x2=-4, 2(3-7)x3=-24, 2(4-8)x4=-32 이므로 합 -62, 평균 -15.5 = -31/2
    assert full == F(-31, 2)
    assert M.grad_accumulated(F(1), batch, 4) == full
    assert M.grad_accumulated(F(1), batch, 2) == full
    assert M.grad_accumulated(F(1), batch, 4, divide=False) == 4 * full


# ---------- 결과 파일 ----------

def test_results_file_is_current():
    res = json.loads((HERE / "results" / "ch05.json").read_text(encoding="utf-8"))
    assert res["e1_params"]["matches_safetensors"] is True
    assert res["e3_activations"]["formula_equals_sum"] is True
    assert res["e4_checkpoint_sim"]["k1_equals_formula"] is True
    assert res["e5_grad_accum"]["equal"] is True
    rows = {r["plan"]: r for r in res["e6_project"]["rows"]}
    assert rows["full_zero_b1_ckpt"]["fits"]["NVIDIA A100 80GB"] is False
    assert rows["qlora_base_b1_ckpt"]["fits"]["NVIDIA T4"] is True
    # T4 16GB에 LoRA(bf16 기본 가중치, 묶음 1, 체크포인팅)가 들어가고 남는 여유 0.29GB(결과 파일 기준, 계산식의 추정이라 실제 측정값이 아니다)
    assert rows["lora_base_b1_ckpt"]["margin_gb"]["NVIDIA T4"] == 0.29
