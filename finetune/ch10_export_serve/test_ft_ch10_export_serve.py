"""내 데이터로 LLM 파인튜닝하기 10장 검증. `uv run pytest -q finetune/ch10_export_serve` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다.
"""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import pytest

import ft_ch10_export as M

HERE = Path(__file__).parent
SPECS = M.load_specs()
CFG = M.load_config()
SC = M.load_scenario()
BLOCKS = SPECS["llama_cpp"]["blocks"]
SIZES = M.load_sizes()["files"]


# ---------- 사양표 ----------

def test_spec_files_have_date():
    assert SPECS["checked"] == "2026-10-11" and M.SPEC_FILE.name == "specs_2026-10-11.json"
    assert M.SIZES_FILE.name == "gguf_sizes_2026-10-11.json"
    for key in ("model", "peft", "transformers", "unsloth", "hub", "llama_cpp", "ollama", "vllm", "aws_prices"):
        assert SPECS[key]["checked"] == "2026-10-11", key
    assert SPECS["model"]["license"] == "apache-2.0"
    assert SPECS["ollama"]["adapter_removed_in"] == "0.34.1"
    assert SPECS["vllm"]["max_lora_rank_default"] == 16 and SPECS["vllm"]["max_loras_default"] == 1
    assert SPECS["vllm"]["gpu_memory_utilization_default"] == 0.92


def test_config_same_as_ch08_and_param_count():
    other = HERE.parent / "ch08_unsloth_colab" / "data" / "qwen2.5-7b-instruct_config.json"
    if other.exists():
        assert json.loads(other.read_text(encoding="utf-8")) == CFG
    assert sum(t[2] for t in M.tensors(CFG)) == SPECS["model"]["safetensors_total"] == 7_615_616_512


def test_small_model_param_count():
    cfg = M.load_config("small_model")
    assert sum(t[2] for t in M.tensors(cfg)) == SPECS["small_model"]["safetensors_total"]


# ---------- 손으로 합치기 ----------

def test_hand_merge_matches_ch07():
    ex = SPECS["from_earlier_chapters"]["ch07_merge_example"]
    h = M.merge_hand(ex["W0"], ex["B"], ex["A"], ex["alpha"], ex["r"], ex["x"])
    assert h["s"] == 2
    assert h["merged"] == [[7, -2], [12, -3]]
    assert h["via_merged"] == h["via_side"] == [12, 21] == ex["out"]
    assert (h["mults_merged"], h["mults_side"]) == (4, 8)


def test_b_zero_merge_is_noop():
    h = M.merge_hand([[1, 2], [3, 4]], [[0], [0]], [[5, 6]], 16, 8, [1, 1])
    assert h["merged"] == [[1, 2], [3, 4]]


def test_storage():
    st = M.storage(3, 15.23, 80.7)
    assert st["separate_gb"] == pytest.approx(15.4721) and st["merged_gb"] == pytest.approx(45.69)


# ---------- bf16 반올림 ----------

def test_bf16_rne_vs_trunc():
    assert M.bf16_rne(1.0) == M.bf16_trunc(1.0) == 1.0
    # 1 + 2^-8은 bf16 두 값(1, 1 + 2^-7)의 딱 가운데: 짝수 쪽(1)으로
    assert M.bf16_rne(1 + 2**-8) == 1.0
    # 1 + 3·2^-8은 1 + 2^-7과 1 + 2^-6의 가운데: 마지막 비트가 짝수인 1 + 2^-6으로
    assert M.bf16_rne(1 + 3 * 2**-8) == 1 + 2**-6
    assert M.bf16_trunc(1 + 3 * 2**-8) == 1 + 2**-7
    x = 0.0123456
    assert abs(M.bf16_rne(x) - x) <= abs(M.bf16_trunc(x) - x)


# ---------- 4비트 어댑터 합치기 장난감 ----------

def test_nf4_roundtrip_codes():
    row = [0.0, 0.5, -1.0, 1.0]
    am, codes = M.nf4_quant(row)
    assert am == 1.0 and M.nf4_dequant(am, codes) == [0.0, M.NF4[12], -1.0, 1.0]  # 0.5는 0.4407 쪽이 더 가까워요


def test_nf4_cell_hand():
    h = M.nf4_cell_hand(0.05, 0.0123, 0.0001)
    assert h["code"] == 10 and h["erased"]
    assert round(h["back"], 6) == 0.012306 and round(h["half_gap_down"], 5) == 0.00213
    assert not M.nf4_cell_hand(0.05, 0.0123, 0.0025)["erased"]


@pytest.fixture(scope="module")
def toy():
    return {rho: M.merge_toy(rho) for rho in SC["toy"]["rho_sweep"]}


def test_toy_requant_erases_small_delta(toy):
    t = toy[0.01]
    assert t["codes_unchanged"] == t["codes_total"] == 512
    assert t["D_requant_nf4"]["vs_effect"] > 1.0  # 변화가 사라지고 블록 기준값만 조금 흔들림


def test_toy_dequant_merge_keeps_delta(toy):
    for rho, t in toy.items():
        assert t["B_dequant_bf16"]["vs_effect"] <= t["B_dequant_bf16_trunc"]["vs_effect"]
        assert t["B_dequant_bf16"]["vs_output"] < 0.005
    assert toy[0.01]["B_dequant_bf16"]["vs_effect"] < 0.2


def test_toy_original_merge_is_quant_error(toy):
    for t in toy.values():
        assert t["C_original_bf16"]["vs_output"] == pytest.approx(0.097, abs=0.003)
    assert toy[0.01]["quant_err_vs_w"] == pytest.approx(0.0946, abs=0.0005)


def test_toy_requant_keeps_more_when_delta_large(toy):
    keep = [toy[r]["codes_unchanged"] for r in SC["toy"]["rho_sweep"]]
    assert keep == sorted(keep, reverse=True) and keep[-1] < 512


# ---------- GGUF ----------

def test_block_bits():
    bits = {k: F(v[1] * 8, v[0]) for k, v in BLOCKS.items()}
    assert bits["Q4_0"] == bits["Q4_K"] == F(9, 2)
    assert bits["Q8_0"] == F(17, 2) and bits["Q6_K"] == F(105, 16) and bits["Q5_K"] == F(11, 2)
    assert bits["Q2_K"] == F(21, 8) and bits["Q3_K"] == F(55, 16)


def test_q4km_half_of_attn_v_and_ffn_down():
    for cat in ("attn_v", "ffn_down"):
        layers = M.q6k_layers("Q4_K_M", cat)
        assert len(layers) == 14 == CFG["num_hidden_layers"] // 2
        assert layers == [0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27]


def test_all_shapes_divisible():
    for ft in M.BASE_TYPE:
        M.gguf_tensor_bytes(CFG, ft, BLOCKS)  # 안에서 블록 나눔을 확인한다


@pytest.mark.parametrize("ft", ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"])
def test_gguf_size_matches_qwen_files(ft):
    t = M.gguf_table(CFG, BLOCKS, SIZES)
    row = t["rows"][ft]
    assert abs(row["diff_bytes"]) <= 512, (ft, row["diff_bytes"])
    assert t["meta_bytes"] == 5_953_888


def test_q4km_size_and_bpw():
    t = M.gguf_table(CFG, BLOCKS, SIZES)
    q = t["rows"]["Q4_K_M"]
    assert round(q["actual_bytes"] / M.GB, 3) == 4.683 and round(q["bpw"], 4) == 4.9132
    assert round(t["rows"]["F16"]["actual_bytes"] / M.GB, 3) == 15.238


# ---------- 띄우기 ----------

def test_kv_per_token():
    assert M.kv_bytes_per_token(CFG) == 2 * 28 * 4 * 128 * 2 == 57_344


def test_vllm_budget():
    kv = M.kv_bytes_per_token(CFG)
    w = SPECS["model"]["bf16_file_bytes"]
    l4 = M.vllm_budget(24, 0.92, w, kv, 750, 16, int(3 * 80.7e6))
    assert l4["fits"] and l4["per_request_tokens"] == 752 and l4["kv_tokens"] == 115_210 and l4["requests"] == 153
    assert not M.vllm_budget(16, 0.92, w, kv, 750, 16)["fits"]


def test_speed_and_cost():
    w = SPECS["model"]["bf16_file_bytes"]
    tok = M.tokens_per_s_upper(300, w)
    assert round(tok, 2) == 19.7
    c = M.monthly_cost(3000, 200, round(tok, 2), 0.8048, 730)
    assert c["tokens"] == 600_000 and round(c["busy_hours"], 2) == 8.46
    assert round(c["always_on_usd"], 2) == 587.5 and round(c["busy_only_usd"], 2) == 6.81


# ---------- 예시 파일 ----------

def test_modelfile_example():
    cmds = M.parse_modelfile((HERE / "data" / "Modelfile.example").read_text(encoding="utf-8"))
    names = [c[0] for c in cmds]
    assert names[0] == "FROM" and cmds[0][1].endswith(".gguf")
    assert "ADAPTER" not in names  # Ollama 0.34.1부터 LoRA 어댑터를 받지 않는다
    template = next(c[1] for c in cmds if c[0] == "TEMPLATE")
    assert "<|im_start|>assistant" in template and "{{ .Prompt }}" in template
    params = {c[1].split()[0]: c[1].split(maxsplit=1)[1] for c in cmds if c[0] == "PARAMETER"}
    assert params["num_ctx"] == str(SC["assumed"]["ctx_local"]) and params["stop"] == '"<|im_end|>"'


def test_model_card_example():
    meta = M.card_front_matter((HERE / "data" / "model_card_example.md").read_text(encoding="utf-8"))
    assert meta["license"] == SPECS["model"]["license"]
    assert meta["base_model"] == SPECS["model"]["name"]
    assert meta["base_model_relation"] in ("adapter", "merge", "quantized", "finetune")
    assert "datasets" not in meta


# ---------- 결과 파일과 노트북 ----------

def test_results_file_matches_module():
    res = json.loads((HERE / "results" / "ch10.json").read_text(encoding="utf-8"))
    assert res["meta"]["spec_file"] == M.SPEC_FILE.name
    assert res["e3_gguf"]["rows"]["Q4_K_M"]["actual_bytes"] == sum(SIZES["Q4_K_M"])
    t = M.merge_toy(SC["toy"]["rho_main"])
    assert res["e2_toy"]["main"]["codes_unchanged"] == t["codes_unchanged"]
    assert math.isclose(res["e2_toy"]["main"]["B_vs_effect"], t["B_dequant_bf16"]["vs_effect"], abs_tol=1e-4)
    assert res["e6_speed"]["L4_bf16"] == 19.7 and res["e7_cost"]["rows"]["g6_L4_bf16_vllm"]["always_on_usd"] == 587.5


def test_notebook_outputs_empty_and_settings():
    nb = json.loads((HERE / "colab" / "ch10_merge_gguf_serve.ipynb").read_text(encoding="utf-8"))
    src = ""
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
        src += "".join(c["source"]) + "\n"
    assert "merge_and_unload" in src and "llama-quantize" in src and "Q4_K_M" in src
    assert "b11552" in src and "visibility" not in src and "private=True" in src
    assert "LoRA adapters are no longer supported" in src  # ADAPTER 줄이 거절되는지 확인하는 칸
