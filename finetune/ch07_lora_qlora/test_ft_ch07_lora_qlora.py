"""내 데이터로 LLM 파인튜닝하기 7장 검증. `uv run pytest -q finetune/ch07_lora_qlora` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다.
"""

import json
import math
import struct
from fractions import Fraction as F
from pathlib import Path

import pytest

import ft_ch07_lora as M

HERE = Path(__file__).parent
CFG = M.load_config()


# ---------- 사양표 ----------

def test_spec_file_has_source_and_date():
    s = M.load_specs()
    assert s["checked"] == "2026-10-11" and M.SPEC_FILE.name == "specs_2026-10-11.json"
    for key in ("model", "lora_paper", "qlora_paper", "peft", "transformers", "bitsandbytes"):
        assert s[key]["checked"] == "2026-10-11"
    assert s["peft"]["version"] == "0.21.2"
    assert s["peft"]["LoraConfig_defaults"]["r"] == 8 and s["peft"]["LoraConfig_defaults"]["lora_alpha"] == 8
    assert s["peft"]["default_targets_qwen2"] == M.TARGETS["qv"]
    assert s["lora_paper"]["v1"] == "2021-06-17" and s["qlora_paper"]["v1"] == "2023-05-23"


def test_config_is_same_as_ch05():
    other = HERE.parent / "ch05_gpu_memory" / "data" / "qwen2.5-7b_config.json"
    if other.exists():
        assert json.loads(other.read_text(encoding="utf-8")) == CFG
    assert M.total_params(CFG) == 7_615_616_512


# ---------- 첫 화면 손계산 ----------

def test_hand_rank1():
    B, A, x = [2, 1, 0, -1], [1, 2, 3, 4], [1, 0, 0, 1]
    dW = M.outer(B, A)
    assert len([v for row in dW for v in row]) == 16 and len(B) + len(A) == 8
    ax = sum(a * b for a, b in zip(A, x))
    assert ax == 5 and M.matvec(dW, x) == [10, 5, 0, -5] == [b * ax for b in B]


# ---------- LoRA 파라미터 수 ----------

def test_linear_shapes():
    s = M.linear_shapes(CFG)
    assert s["q_proj"] == (3584, 3584) and s["k_proj"] == (3584, 512) and s["v_proj"] == (3584, 512)
    assert s["o_proj"] == (3584, 3584) and s["gate_proj"] == (3584, 18944) and s["down_proj"] == (18944, 3584)


@pytest.mark.parametrize("t,coef", [("qv", 11264), ("qkvo", 22528), ("mlp", 67584), ("all_linear", 90112)])
def test_per_layer_coeff(t, coef):
    assert M.lora_per_layer_coeff(CFG, M.TARGETS[t]) == coef
    for r in (1, 8, 16, 64):
        assert M.lora_params(CFG, M.TARGETS[t], r) == coef * r * 28


def test_named_counts():
    assert M.lora_params(CFG, M.TARGETS["qv"], 8) == 2_523_136
    assert M.lora_params(CFG, M.TARGETS["all_linear"], 16) == 40_370_176
    assert M.lora_params(CFG, M.TARGETS["all_linear"], 64) == 161_480_704
    assert M.lora_params(CFG, M.TARGETS["all_linear"], 1) == M.lora_params(CFG, M.TARGETS["qv"], 8)
    assert round(100 * 40_370_176 / 7_615_616_512, 2) == 0.53


def test_lora_module_count_is_r_times_sum():
    for d_in, d_out, r in [(4, 4, 1), (3584, 512, 8), (18944, 3584, 16)]:
        assert M.lora_params_module(d_in, d_out, r) == r * (d_in + d_out)
        assert M.lora_params_module(d_in, d_out, r) < M.full_params_module(d_in, d_out)


# ---------- 기본 모델 나누기 ----------

def test_base_split():
    sp = M.base_split(CFG)
    assert sp["linear_4bit"] == 6_525_288_448 and sp["kept"] == 1_090_328_064
    assert sp["linear_4bit"] + sp["kept"] == 7_615_616_512
    assert sp["embed"] == sp["lm_head"] == 544_997_376
    assert sp["qkv_bias"] == 28 * (3584 + 1024) and sp["norms"] == 28 * 7168 + 3584


def test_nf4_bits_from_ch06():
    assert M.nf4_bits() == F(9, 2)
    assert M.nf4_bits(double_quant=True) == F(8, 64) + F(32, 64 * 256) + 4
    assert float(F(7 * 10**9) * M.nf4_bits() / 8 / M.GB) == 3.9375


# ---------- 메모리 ----------

def test_lora_bf16_totals():
    x = M.lora_bf16_total(CFG, 8, "qv")
    assert round((x["base"] + x["act"]) / M.GB, 2) == 15.71  # 5장 미리 보기
    assert round(x["total"] / M.GB, 2) == 15.75 and x["total"] <= 16 * M.GB
    y = M.lora_bf16_total(CFG, 8, "all_linear")
    assert y["total"] > 16 * M.GB


def test_qlora_ladder_bridges_ch05():
    L = M.qlora_ladder(CFG)
    steps = dict(L["steps"])
    assert round(float(steps["5장 미리 보기: 모든 숫자 0.5바이트"]) / M.GB, 2) == 3.81
    pre_act = steps["5장 미리 보기: 모든 숫자 0.5바이트"] + steps["5장 활성값 추정(체크포인팅)"]
    assert round(float(pre_act) / M.GB, 2) == 4.28
    assert round(float(L["total"]) / M.GB, 2) == 6.67
    assert L["total"] == pre_act + L["alpha_part"]
    assert round(float(L["alpha_part"]) / M.GB, 3) == 2.385
    assert round(float(M.qlora_ladder(CFG, double_quant=False)["total"]) / M.GB, 2) == 6.97
    assert round(float(M.qlora_ladder(CFG, kept_bytes=4)["total"]) / M.GB, 2) == 8.85
    assert round(float(M.qlora_ladder(CFG, lora_path="paper_bf16")["total"]) / M.GB, 2) == 6.51


def test_activation_same_as_ch05():
    assert round(M.activation_ckpt_bytes(1024, 1, 3584, 28, 28) / M.GB, 4) == 0.4771


# ---------- 배율 ----------

def test_scaling():
    assert M.scaling(8, 8) == 1 and M.scaling(16, 64) == F(1, 4) and M.scaling(16, 8) == 2
    assert M.scaling(16, 64, rslora=True) == 2.0
    assert M.scaling(16, 32) == M.scaling(16, 16) / 2


def test_random_sign_growth_is_sqrt():
    g = M.random_sign_growth()
    for r in (4, 16, 64):
        assert abs(g[r] / math.sqrt(r * 256) - 1) < 0.05


# ---------- 작은 행렬 ----------

def test_toy_matrix_eckart_young():
    sig = (F(3), F(1), F(1, 10))
    Mx = M.toy_matrix(sig)
    assert M.frob2(Mx) == sum(s * s for s in sig) == F(1001, 100)
    for k in (1, 2, 3):
        assert M.frob2(M.add(Mx, M.truncated(sig, k), -1)) == sum(s * s for s in sig[k:])


def test_toy_matrix_entries():
    Mx = M.toy_matrix()
    assert Mx[0] == [F(41, 40), F(-21, 40), F(19, 40), F(-39, 40)]


def test_power_iteration_finds_sigma1():
    s1, _ = M.power_iteration(M.toy_matrix())
    assert abs(s1 - 3) < 1e-9


@pytest.mark.parametrize("r,best", [(1, 1.01), (2, 0.01)])
def test_gd_reaches_svd_optimum(r, best):
    out = M.train_lora_toy(M.toy_matrix(), r)
    assert out["start_is_zero"] and abs(out["start_err2"] - 10.01) < 1e-12
    assert abs(out["end_err2"] - best) < 1e-6


def test_merge_equals_separate_and_zero_init():
    W0 = [[F(1), F(0)], [F(0), F(1)]]
    A = [[F(3), F(-1)]]
    x = [F(2), F(1)]
    merged, sep = M.merged_equals_separate(W0, [[F(1)], [F(2)]], A, x, F(2))
    assert merged == sep == [F(12), F(21)]
    m0, _ = M.merged_equals_separate(W0, [[F(0)], [F(0)]], A, x, F(2))
    assert m0 == x


# ---------- 바이트와 페이징 ----------

def test_bytes_per_number():
    assert struct.calcsize("e") == 2 and M.bytes_per_number("fp16") == 2
    assert M.bytes_per_number("bf16") == 2 and M.bytes_per_number("fp32") == 4
    x = struct.unpack(">f", bytes.fromhex("3f80ffff"))[0]
    assert M.to_bf16_bytes(x) == b"\x3f\x80"  # 반올림하지 않고 잘라낸다


def test_paging_overhead_doc_example():
    assert M.paging_overhead_ms(1.0) == 125.0


def test_paged_window():
    L = M.qlora_ladder(CFG)
    fixed = L["total"] - M.activation_ckpt_bytes(1024, 1, 3584, 28, 28)
    w = M.paged_window(fixed, L["lora_params"] * 8, 16 * M.GB, 3584, 28, 28)
    assert w["max_len_no_paging"] == 7168 and w["max_len_paged"] == 7424


# ---------- 결과 파일 ----------

def test_results_file_is_current():
    res = json.loads((HERE / "results" / "ch07.json").read_text(encoding="utf-8"))
    assert res["e0_hand"]["same"] is True
    assert res["e1_params"]["allr1_equals_qv_r8"] is True
    assert all(row["eckart_young_holds"] and row["gd_matches_svd"] for row in res["e3_toy"]["rows"])
    assert res["e4_merge"]["equal"] is True and res["e4_merge"]["B_zero_is_noop"] is True
    assert res["e6_qlora"]["main"]["total_gb"] == 6.67
    assert res["e6_qlora"]["main"]["lora_params"] == 40_370_176
    assert res["e7_paged"]["window"] == {"max_len_no_paging": 7168, "max_len_paged": 7424}
