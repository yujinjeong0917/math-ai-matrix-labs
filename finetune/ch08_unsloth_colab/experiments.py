"""내 데이터로 LLM 파인튜닝하기 8장 실험. `uv run python finetune/ch08_unsloth_colab/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 한 에폭의 스텝 수, 공식 노트북 60스텝이 본 문의 수, 채우기 손 예시
E1 config.json으로 파라미터 세기와 4bit 파일 크기 대조(7B·3B·1.5B, Hub 값과 비교)
E2 LoRA 학습 파라미터 수(랭크별), 텐서 하나씩 더한 값과 대조, 바이트
E3 T4 16GB 메모리 어림: 묶음 × 길이 격자, 랭크별, 모델·방식별, 회사 주장 적용 줄
E4 스텝 수: 같은 실제 묶음의 여러 조합, 묶어 담기 후 스텝 수, 워밍업
E5 채우기·묶어 담기 토큰 수와 에폭 시간 어림(가정한 활용률별), Colab 12시간과 비교
E6 체크포인트: 기본 save_steps와 전체 스텝 비교

표준 라이브러리만 쓴다. 무작위 값을 쓰지 않아 결과는 매번 같다. GPU 학습은 하지 않는다.
"""

import json
import math
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch08_unsloth as M  # noqa: E402

OUT = HERE / "results" / "ch08.json"


def r(x, nd=3):
    return round(float(x), nd)


def gb(x, nd=3):
    return r(F(x) / M.GB, nd)


def e0_hand(sc, specs):
    n, b, g = sc["train_pairs"], sc["micro_batch"], sc["grad_accum"]
    nb = specs["official_notebook"]
    seen = M.examples_seen(nb["max_steps"], nb["per_device_train_batch_size"], nb["gradient_accumulation_steps"])
    L, S = 1024, 256
    return {"train_pairs": n, "micro_batch": b, "grad_accum": g, "effective_batch": b * g,
            "raw_division": r(F(n, b * g), 4), "steps_per_epoch": M.steps_per_epoch(n, b, g),
            "epochs": sc["epochs"], "total_steps": sc["epochs"] * M.steps_per_epoch(n, b, g),
            "official_max_steps": nb["max_steps"], "official_examples_seen": seen,
            "official_epoch_fraction": r(F(seen, n), 3),
            "pad_example": {"long": L, "short": S, "slots": 2 * L, "real": L + S, "wasted": L - S,
                            "waste_pct": r(100 * F(L - S, 2 * L), 1),
                            "unsloth_speedup": r(M.unsloth_packing_speedup(L, S), 3)}}


def e1_params_files(specs):
    rows = []
    for key, m in specs["models"].items():
        cfg = M.load_config(key, specs)
        n = M.count_params(cfg)
        q = M.count_quantized_linear(cfg)
        bb = M.bnb4_bytes(cfg)
        rows.append({"model": key, "bnb_repo": m["bnb_repo"], "license": m["license"],
                     "tie_word_embeddings": cfg["tie_word_embeddings"],
                     "params": n, "safetensors_total": m["safetensors_total"], "params_match": n == m["safetensors_total"],
                     "quantized_linear": q, "kept_16bit": n - q, "hub_bf16_params": m["bnb_hub_bf16_params"],
                     "kept_match_hub": n - q == m["bnb_hub_bf16_params"],
                     "kept_share_pct": r(100 * F(n - q, n), 1),
                     "bytes_parts": bb, "model_bytes": bb["total"], "file_bytes": m["bnb_file_bytes"],
                     "diff_bytes": m["bnb_file_bytes"] - bb["total"],
                     "diff_pct": r(100 * F(m["bnb_file_bytes"] - bb["total"], m["bnb_file_bytes"]), 3),
                     "model_gb": gb(bb["total"], 3), "file_gb": gb(m["bnb_file_bytes"], 3),
                     "bf16_gb": gb(2 * n, 2),
                     "naive_half_byte_gb": gb(F(n, 2), 2)})
    k7 = rows[0]
    q7 = k7["quantized_linear"]
    return {"rows": rows,
            "bits_dq": str(M.bits_per_param_double_quant()), "bits_dq_float": r(M.bits_per_param_double_quant(), 6),
            "q7_4bit_gb": gb(F(q7) * M.bits_per_param_double_quant() / 8, 3),
            "kept7_bf16_gb": gb(2 * k7["kept_16bit"], 3),
            "ch05_preview_weights_gb": gb(F(k7["params"], 2), 2),
            "embed_head_7b": 2 * 152064 * 3584}


def e2_lora(specs, sc):
    cfg = M.load_config(sc["model_key"], specs)
    n = M.count_params(cfg)
    per_layer = {name: din + dout for name, din, dout in M.linear_shapes(cfg)}
    rows = []
    for rk in sc["grid_rank"]:
        p = M.lora_params_closed(cfg, rk)
        tens = M.lora_tensors(cfg, rk)
        rows.append({"r": rk, "params": p, "by_tensors": sum(a * b for _, (a, b) in tens), "n_tensors": len(tens),
                     "share_pct": r(100 * F(p, n), 2),
                     "gb_adamw_8bit": gb(p * M.LORA_BYTES["adamw_8bit"], 3),
                     "gb_adamw_torch": gb(p * M.LORA_BYTES["adamw_torch"], 3)})
    q_only = M.lora_params_closed(cfg, 16, ("q_proj", "v_proj"))
    return {"per_layer_in_plus_out": per_layer, "per_layer_sum": sum(per_layer.values()),
            "layers": cfg["num_hidden_layers"], "rows": rows, "bytes_per_param": M.LORA_BYTES,
            "r16_qv_only": q_only, "r16_qv_share_pct": r(100 * F(q_only, n), 3)}


def e3_memory(specs, sc):
    key = sc["model_key"]
    cfg = M.load_config(key, specs)
    t4 = specs["gpu_t4"]["memory_gb"]
    grid = []
    for b in sc["grid_micro_batch"]:
        for s in sc["grid_seq_len"]:
            p = M.memory_parts(cfg, s, b, sc["lora_r"])
            tot = F(p["total"], M.GB)
            grid.append({"micro_batch": b, "seq_len": s, "parts_gb": {k: gb(v) for k, v in p.items()},
                         "total_gb": r(tot, 2), "margin_t4_gb": r(t4 - tot, 2),
                         "verdict": "fits" if t4 - tot >= 1 else ("tight" if t4 - tot >= 0 else "oom")})
    main = M.memory_parts(cfg, sc["max_seq_length"], sc["micro_batch"], sc["lora_r"])
    main_claim = M.memory_parts(cfg, sc["max_seq_length"], sc["micro_batch"], sc["lora_r"],
                                gc_claim=specs["mechanisms"]["gradient_checkpointing"]["memory_saving_claim"])
    main_adamw = M.memory_parts(cfg, sc["max_seq_length"], sc["micro_batch"], sc["lora_r"], optim="adamw_torch")
    ranks = []
    for rk in sc["grid_rank"]:
        p = M.memory_parts(cfg, sc["max_seq_length"], sc["micro_batch"], rk)
        ranks.append({"r": rk, "lora_gb": gb(p["lora"]), "total_gb": gb(p["total"], 2)})
    methods = []
    for mk in ("qwen2.5-7b-instruct", "qwen2.5-3b-instruct", "qwen2.5-1.5b-instruct"):
        c = M.load_config(mk, specs)
        for meth in ("qlora", "lora16"):
            p = M.memory_parts(c, sc["max_seq_length"], sc["micro_batch"], sc["lora_r"], method=meth)
            tot = F(p["total"], M.GB)
            methods.append({"model": mk, "method": meth, "weights_gb": gb(p["weights"], 2),
                            "rest_gb": gb(p["lora"] + p["activations"] + p["logits"], 2),
                            "total_gb": r(tot, 2), "margin_t4_gb": r(t4 - tot, 2),
                            "verdict": "fits" if t4 - tot >= 1 else ("tight" if t4 - tot >= 0 else "oom")})
    return {"t4_gb": t4, "lora_r": sc["lora_r"], "grid": grid,
            "main": {k: gb(v) for k, v in main.items()}, "main_margin_gb": r(t4 - F(main["total"], M.GB), 2),
            "main_gib": r(F(main["total"], M.GIB), 2),
            "main_with_gc_claim": {k: gb(v) for k, v in main_claim.items()},
            "main_adamw_torch_total_gb": gb(main_adamw["total"], 2),
            "ranks": ranks, "methods": methods,
            "unsloth_table_7b_qlora_gb": specs["requirements"]["vram_table_gb"]["7B"][0],
            "unsloth_table_7b_lora_gb": specs["requirements"]["vram_table_gb"]["7B"][1]}


def e4_steps(sc):
    n = sc["train_pairs"]
    combos = []
    for b, g in [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]:
        combos.append({"micro_batch": b, "grad_accum": g, "effective": b * g, "minibatches": math.ceil(n / b),
                       "steps_per_epoch": M.steps_per_epoch(n, b, g)})
    lens = M.expand_hist(sc["length_histogram"])
    bins = M.pack_bfd(lens, sc["max_seq_length"])
    spe = M.steps_per_epoch(n, sc["micro_batch"], sc["grad_accum"])
    spe_packed = M.steps_per_epoch(len(bins), sc["micro_batch"], sc["grad_accum"])
    total = sc["epochs"] * spe
    odd = M.steps_per_epoch(530, 2, 8)
    return {"combos": combos, "steps_per_epoch": spe, "total_steps": total,
            "warmup_5pct": math.ceil(0.05 * total), "warmup_10pct": math.ceil(0.10 * total),
            "packed_rows": len(bins), "packed_fill_pct": r(100 * F(sum(bins), len(bins) * sc["max_seq_length"]), 1),
            "packed_steps_per_epoch": spe_packed, "packed_total_steps": sc["epochs"] * spe_packed,
            "example_530": {"pairs": 530, "raw": r(F(530, 16), 4), "steps": odd},
            "exercise_900": {"pairs": 900, "micro_batch": 4, "grad_accum": 4, "epochs": 2,
                             "minibatches": math.ceil(900 / 4), "raw": r(F(math.ceil(900 / 4), 4), 4),
                             "steps_per_epoch": M.steps_per_epoch(900, 4, 4),
                             "total_steps": 2 * M.steps_per_epoch(900, 4, 4)},
            "exercise_qv_only": {"r": 16, "params": M.lora_params_closed(M.load_config(sc["model_key"]), 16, ("q_proj", "v_proj"))}}


def e5_time(specs, sc):
    key = sc["model_key"]
    cfg = M.load_config(key, specs)
    n = M.count_params(cfg)
    h = sc["length_histogram"]
    s = sc["max_seq_length"]
    tflops = specs["gpu_t4"]["fp16_tflops"]
    real = M.tokens_real(h)
    pad_max = M.tokens_pad_to_max(h, s)
    pad_b = M.tokens_pad_to_batch_max(h, sc["micro_batch"])
    lens = M.expand_hist(h)
    modes = {"pad_to_max": pad_max, "pad_to_batch_max": pad_b, "packed": real}
    rows = []
    for mode, tok in modes.items():
        row = {"mode": mode, "tokens_per_epoch": r(tok, 1), "real_share_pct": r(100 * F(real) / F(tok), 1),
               "seconds_per_epoch_mfu0.2": r(M.epoch_seconds(tok, n, tflops, 0.2), 1),
               "minutes_per_epoch": {}, "hours_total": {}}
        for mfu in sc["mfu_assumed"]:
            sec = M.epoch_seconds(tok, n, tflops, mfu)
            row["minutes_per_epoch"][str(mfu)] = r(sec / 60, 1)
            row["hours_total"][str(mfu)] = r(sec * sc["epochs"] / 3600, 2)
        rows.append(row)
    worst = max(v for row in rows for v in row["hours_total"].values())
    emb = cfg["vocab_size"] * cfg["hidden_size"]
    return {"params": n, "flops_per_token_ckpt": M.flops_per_token(n), "t4_fp16_tflops": tflops,
            "input_embedding_params": emb, "input_embedding_share_pct": r(100 * F(emb, n), 1),
            "pairs": sum(h.values()), "avg_len": r(F(real, sum(h.values())), 1),
            "expected_max_of_2": r(M.expected_batch_max(h, 2), 2),
            "speedup_pad_max_vs_packed": r(F(pad_max, real), 2),
            "unsloth_formula_example": r(M.unsloth_packing_speedup(896, 192), 3),
            "rows": rows, "colab_limit_hours": 12, "worst_total_hours": worst,
            "within_12h": worst < 12,
            "mfu_backout_example": {"note": "예: 패딩 없이 1에폭 212,544토큰을 1,200초에 돌았다면", "seconds": 1200,
                                    "mfu": r(M.mfu_from_runtime(real, n, tflops, 1200), 3)}}


def e6_checkpoints(specs, sc):
    total = sc["epochs"] * M.steps_per_epoch(sc["train_pairs"], sc["micro_batch"], sc["grad_accum"])
    default = specs["trl"]["save_steps_default"]
    spe = M.steps_per_epoch(sc["train_pairs"], sc["micro_batch"], sc["grad_accum"])
    return {"total_steps": total, "default_save_steps": default, "saves_before_end_default": total // default,
            "save_steps_per_epoch": spe, "saves_per_epoch_setting": total // spe,
            "official_nb_full_epochs": {"micro_batch": 2, "grad_accum": 4,
                                         "steps": 3 * M.steps_per_epoch(600, 2, 4), "pairs": 600}}


def main():
    specs = M.load_specs()
    sc = M.load_scenario()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": M.SPEC_FILE.name, "checked": specs["checked"], "gb": "10^9 bytes",
                 "note": "GPU 학습 결과가 아니라 계산 모형 값이다"},
        "e0_hand": e0_hand(sc, specs),
        "e1_params_files": e1_params_files(specs),
        "e2_lora": e2_lora(specs, sc),
        "e3_memory": e3_memory(specs, sc),
        "e4_steps": e4_steps(sc),
        "e5_time": e5_time(specs, sc),
        "e6_checkpoints": e6_checkpoints(specs, sc),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
