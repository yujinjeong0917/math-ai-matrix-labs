"""내 데이터로 LLM 파인튜닝하기 8장 검증. `uv run pytest -q finetune/ch08_unsloth_colab` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다. torch·unsloth를 불러오지 않는다.
"""

import ast
import csv
import json
import math
from fractions import Fraction as F
from pathlib import Path

import ft_ch08_unsloth as M

HERE = Path(__file__).parent
NB = HERE / "colab" / "ch08_unsloth_qlora_t4.ipynb"
RES = HERE / "results" / "ch08.json"


# ---------- 사양표 ----------

def test_spec_file_dated_and_sourced():
    s = M.load_specs()
    assert s["checked"] == "2026-10-11" and M.SPEC_FILE.name == "specs_2026-10-11.json"

    def walk(d):
        if isinstance(d, dict):
            if "checked" in d:
                assert d["checked"] == "2026-10-11"
            for k, v in d.items():
                if k.startswith("url"):
                    assert v.startswith("https://"), (k, v)
                walk(v)
    walk(s)
    assert "[확인 필요]" in s["colab"]["free_gpu_type"]
    # 회사 주장 표시
    for k in ("kernels", "gradient_checkpointing", "grad_accum_fix", "prequantized"):
        assert s["mechanisms"][k]["claim"] is True
    assert s["requirements"]["claim"] is True


def test_t4_capability_below_bf16_threshold():
    s = M.load_specs()
    assert s["gpu_t4"]["cuda_capability"] == 7.5 < 8
    assert s["requirements"]["core_min_cuda_capability"] <= s["gpu_t4"]["cuda_capability"]


# ---------- 파라미터와 4bit 파일 ----------

def test_param_counts_match_hub_for_three_models():
    s = M.load_specs()
    for key, m in s["models"].items():
        cfg = M.load_config(key, s)
        assert M.count_params(cfg) == m["safetensors_total"], key


def test_kept_16bit_equals_hub_bf16_count():
    """양자화하지 않는 텐서(임베딩, lm_head, 정규화, 편향) 개수가 Hub의 BF16 개수와 한 개도 다르지 않다."""
    s = M.load_specs()
    for key, m in s["models"].items():
        cfg = M.load_config(key, s)
        assert M.count_params(cfg) - M.count_quantized_linear(cfg) == m["bnb_hub_bf16_params"], key
    cfg7 = M.load_config("qwen2.5-7b-instruct", s)
    assert M.count_quantized_linear(cfg7) == 28 * 233_046_016 == 6_525_288_448


def test_bnb_file_size_within_0_05_percent():
    s = M.load_specs()
    for key, m in s["models"].items():
        b = M.bnb4_bytes(M.load_config(key, s))
        diff = m["bnb_file_bytes"] - b["total"]
        assert 0 < diff < 0.0005 * m["bnb_file_bytes"], (key, diff)  # 남는 건 헤더와 작은 상태값
    b7 = M.bnb4_bytes(M.load_config("qwen2.5-7b-instruct", s))
    assert b7["total"] == 5_546_851_072


def test_bits_per_param_matches_ch06():
    assert M.bits_per_param_double_quant() == F(2113, 512)
    assert round(float(M.bits_per_param_double_quant()), 6) == 4.126953


def test_hub_u8_count_is_not_bytes():
    """Hub API의 U8 '파라미터' 수(6,729,271,260)는 파일 바이트가 아니다. 그대로 더하면 파일보다 커진다."""
    s = M.load_specs()
    m = s["models"]["qwen2.5-7b-instruct"]
    as_if_bytes = 6_729_271_260 + 2 * m["bnb_hub_bf16_params"] + 4 * m["bnb_hub_f32_params"]
    assert as_if_bytes > m["bnb_file_bytes"] * 1.5


# ---------- LoRA ----------

def test_lora_closed_form_equals_tensor_sum():
    cfg = M.load_config("qwen2.5-7b-instruct")
    for r in (8, 16, 32, 64):
        closed = M.lora_params_closed(cfg, r)
        tens = M.lora_tensors(cfg, r)
        assert closed == sum(a * b for _, (a, b) in tens) == r * 2_523_136
        assert len(tens) == 28 * 7 * 2
    assert M.lora_params_closed(cfg, 16) == 40_370_176


def test_lora_hand_numbers():
    # q_proj: 3584 -> 3584, 랭크 16이면 16 × (3584 + 3584) = 114,688
    cfg = M.load_config("qwen2.5-7b-instruct")
    assert M.lora_params_closed(cfg, 16, ("q_proj",)) == 28 * 16 * 7168
    assert 16 * (3584 + 3584) == 114_688


# ---------- 메모리 ----------

def test_main_config_memory():
    sc = M.load_scenario()
    cfg = M.load_config(sc["model_key"])
    p = M.memory_parts(cfg, 1024, 2, 16)
    assert round(p["weights"] / 1e9, 3) == 5.547
    assert round(p["lora"] / 1e9, 3) == 0.404
    assert round(p["activations"] / 1e9, 3) == 0.954
    assert round(p["logits"] / 1e9, 3) == 1.246
    assert round(p["total"] / 1e9, 2) == 8.15


def test_activation_matches_ch05_at_b1():
    cfg = M.load_config("qwen2.5-7b-instruct")
    assert round(M.activation_ckpt_bytes(1024, 1, cfg) / 1e9, 4) == 0.4771  # 5장 체크포인팅 값


def test_lora16_7b_does_not_fit_t4_but_qlora_does():
    cfg = M.load_config("qwen2.5-7b-instruct")
    assert M.memory_parts(cfg, 1024, 2, 16, method="lora16")["total"] > 16e9
    assert M.memory_parts(cfg, 1024, 2, 16, method="qlora")["total"] < 16e9 - 1e9


def test_gc_claim_only_touches_activations():
    cfg = M.load_config("qwen2.5-7b-instruct")
    a = M.memory_parts(cfg, 1024, 2, 16)
    b = M.memory_parts(cfg, 1024, 2, 16, gc_claim=0.30)
    assert a["weights"] == b["weights"] and a["logits"] == b["logits"]
    assert b["activations"] == a["activations"] * F(7, 10)


# ---------- 스텝·채우기·시간 ----------

def test_steps_hand_example():
    assert F(540, 16) == F(135, 4)  # 33.75
    assert M.steps_per_epoch(540, 2, 8) == 34
    for b, g in [(1, 16), (4, 4), (8, 2), (16, 1)]:
        assert M.steps_per_epoch(540, b, g) == 34
    assert M.examples_seen(60, 2, 4) == 480 < 540


def test_padding_hand_example_and_unsloth_formula():
    assert M.unsloth_packing_speedup(1024, 256) == F(8, 5)
    # 1024토큰 하나와 256토큰 하나를 1024칸 두 줄에 채우면 2048칸 중 1280칸만 진짜
    assert F(2048 - 1280, 2048) == F(3, 8)


def test_expected_batch_max_exact():
    h = {"100": 1, "300": 1}
    # 두 개를 뽑으면 (100,100)만 최댓값 100, 나머지 셋은 300: (100 + 3·300)/4 = 250
    assert M.expected_batch_max(h, 2) == 250
    assert M.expected_batch_max(h, 1) == 200


def test_bfd_packing_never_overflows():
    sc = M.load_scenario()
    bins = M.pack_bfd(M.expand_hist(sc["length_histogram"]), sc["max_seq_length"])
    assert all(x <= sc["max_seq_length"] for x in bins)
    assert sum(bins) == M.tokens_real(sc["length_histogram"]) == 212_544
    assert len(bins) >= math.ceil(212_544 / 1024)


def test_epoch_time_formula():
    n = 7_615_616_512
    sec = M.epoch_seconds(552_960, n, 65, 0.2)
    assert round(float(sec) / 60, 1) == 32.4
    assert M.mfu_from_runtime(552_960, n, 65, sec) == F(1, 5)


# ---------- 결과 파일과 노트북 ----------

def test_results_file_matches_model():
    d = json.loads(RES.read_text(encoding="utf-8"))
    assert d["e0_hand"]["steps_per_epoch"] == 34 and d["e0_hand"]["total_steps"] == 102
    assert d["e1_params_files"]["rows"][0]["model_bytes"] == 5_546_851_072
    assert d["e3_memory"]["main"]["total"] == 8.15
    assert d["e4_steps"]["packed_steps_per_epoch"] == 14
    assert d["e6_checkpoints"]["saves_before_end_default"] == 0
    ex = d["e4_steps"]["exercise_900"]
    assert (ex["minibatches"], ex["raw"], ex["steps_per_epoch"], ex["total_steps"]) == (225, 56.25, 57, 114)
    assert d["e4_steps"]["exercise_qv_only"]["params"] == 5_046_272
    rows = {row["mode"]: row for row in d["e5_time"]["rows"]}
    assert rows["pad_to_batch_max"]["seconds_per_epoch_mfu0.2"] == 941.1
    assert d["e5_time"]["input_embedding_share_pct"] == 7.2


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def test_notebook_outputs_empty():
    nb = _nb()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def _settings():
    nb = _nb()
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "MODEL_NAME =" in src:
            vals = {}
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    vals[node.targets[0].id] = ast.literal_eval(node.value)
            return vals
    raise AssertionError("설정 칸이 없다")


def test_notebook_settings_equal_scenario():
    sc = M.load_scenario()
    v = _settings()
    assert v["MODEL_NAME"] == sc["model_repo"]
    assert v["MAX_SEQ_LENGTH"] == sc["max_seq_length"]
    assert v["MICRO_BATCH"] == sc["micro_batch"] and v["GRAD_ACCUM"] == sc["grad_accum"]
    assert v["LORA_R"] == sc["lora_r"] and v["LORA_ALPHA"] == sc["lora_alpha"]
    assert v["EPOCHS"] == sc["epochs"] and v["LEARNING_RATE"] == sc["learning_rate"]
    assert v["SEED"] == sc["seed"] and v["OPTIM"] == sc["optim"]
    assert v["PAIRS_TOTAL"] == sc["pairs_total"] and v["VAL_PAIRS"] == sc["val_pairs"]
    d = json.loads(RES.read_text(encoding="utf-8"))
    assert v["CALC"]["weights_gb"] == d["e1_params_files"]["rows"][0]["model_gb"]
    assert v["CALC"]["lora_params"] == d["e2_lora"]["rows"][1]["params"]
    assert v["CALC"]["total_gb"] == d["e3_memory"]["main"]["total"]
    assert v["CALC"]["steps_per_epoch"] == d["e4_steps"]["steps_per_epoch"]


def test_notebook_sets_precision_explicitly_and_saves_each_epoch():
    src = "".join("".join(c["source"]) for c in _nb()["cells"])
    assert "fp16 = USE_FP16" in src and "bf16 = USE_BF16" in src
    assert "save_steps = steps_per_epoch" in src
    assert '"<|im_start|>assistant\\n"' in src
    assert "max_memory_allocated" in src and "max_memory_reserved" in src


def test_notebook_dataset_builder_makes_600_pairs():
    """데이터 칸에서 순수 파이썬 부분(DOCS, ASK, ITEMS, SYSTEM, make_pairs)만 떼어 돌려 본다."""
    for c in _nb()["cells"]:
        src = "".join(c["source"])
        if "def make_pairs" in src:
            tree = ast.parse(src)
            keep = [n for n in tree.body if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("DOCS", "ASK", "ITEMS", "SYSTEM"))
                    or (isinstance(n, ast.FunctionDef) and n.name == "make_pairs")]
            ns = {}
            exec(compile(ast.Module(body=keep, type_ignores=[]), "nb", "exec"), ns)
            pairs = ns["make_pairs"]()
            assert len(pairs) == 600 and len({p[1]["content"] for p in pairs}) == 600
            assert all(p[2]["content"].startswith("안녕하세요, 고객님.") for p in pairs)
            return
    raise AssertionError("데이터 칸이 없다")


def test_checks_csv_columns_match_notebook_row():
    rows = list(csv.reader((HERE / "checks" / "ch08.csv").open(encoding="utf-8")))
    assert len(rows[0]) == 17 and all(len(r) == 17 for r in rows)
    src = "".join("".join(c["source"]) for c in _nb()["cells"] if "row = [" in "".join(c["source"]))
    row = [n for n in ast.parse(src).body if isinstance(n, ast.Assign) and n.targets[0].id == "row"][0]
    assert len(row.value.elts) == 17


def test_no_em_dash_in_text_files():
    for p in [NB, HERE / "README.md", HERE / "checks" / "ch08.csv"]:
        assert "—" not in p.read_text(encoding="utf-8"), p
