"""내 데이터로 LLM 파인튜닝하기 5장 실험. `uv run python finetune/ch05_gpu_memory/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 숫자 1,000개를 형식별로 담아 바이트 재기, 10억 개 모델의 추론·학습 메모리
E1 config.json으로 파라미터 수 세기: 닫힌 식 / 텐서 하나씩 / 모델 카드·safetensors 값
E2 학습 방식별 모델 상태 메모리(ZeRO 16바이트, HF 18바이트, 8비트 Adam 12바이트, 추론, LoRA·QLoRA 미리 보기)
E3 활성값: 논문 식 (1)과 항목별 내역, 길이·묶음 크기별 크기, 체크포인팅
E4 체크포인팅 흉내: 구간 크기 k별 최대 메모리와 다시 계산한 층 수, 식과 대조
E5 기울기 누적 장난감: 묶음 4개 한 번 vs 1개씩 4번(분수로 정확히)
E6 관통 프로젝트 설정별 합계와 GPU(T4 16GB, L4 24GB, A100 80GB)

표준 라이브러리만 쓴다. 무작위 값을 쓰지 않아 결과는 매번 같다.
"""

import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch05_memory as M  # noqa: E402

OUT = HERE / "results" / "ch05.json"


def r(x, nd=4):
    return round(float(x), nd)


def e0_hand():
    nums = [i / 1000 for i in range(1000)]
    sizes = {fmt: len(M.pack_numbers(nums, fmt)) for fmt in ("fp32", "fp16", "bf16")}
    n = 10**9
    return {"numbers": len(nums), "bytes": sizes,
            "per_number": {fmt: M.bytes_per_number(fmt) for fmt in ("fp32", "fp16", "bf16")},
            "params_1b": n,
            "infer_bf16_gb": r(F(2 * n, M.GB)), "train_zero_gb": r(F(16 * n, M.GB)),
            "train_hf_gb": r(F(18 * n, M.GB))}


def e1_params(specs, cfg):
    closed = M.count_closed_form(cfg)
    by_tensor = M.count_by_tensors(cfg)
    total = closed["total"]
    return {"model": specs["model"]["name"],
            "config": {k: cfg[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                            "num_attention_heads", "num_key_value_heads", "vocab_size",
                                            "tie_word_embeddings", "torch_dtype")},
            "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
            "per_layer": closed["per_layer"], "attn_per_layer": closed["attn"], "mlp_per_layer": closed["mlp"],
            "embedding": closed["embedding"], "total_closed": total, "total_by_tensors": by_tensor,
            "n_tensors": len(M.named_tensors(cfg)), "safetensors_total": specs["model"]["safetensors_total"],
            "matches_safetensors": total == by_tensor == specs["model"]["safetensors_total"],
            "non_embedding": closed["non_embedding"], "card": specs["model"]["card_params"],
            "card_non_embedding": specs["model"]["card_params_non_embedding"],
            "embed_and_head_share_pct": r(100 * F(2 * closed["embedding"], total), 1),
            "norm_per_layer": 2 * cfg["hidden_size"],
            "kv_dim": cfg["num_key_value_heads"] * (cfg["hidden_size"] // cfg["num_attention_heads"]),
            "q_per_kv_head": cfg["num_attention_heads"] // cfg["num_key_value_heads"],
            "embed_and_head": 2 * closed["embedding"], "layers_total": cfg["num_hidden_layers"] * closed["per_layer"],
            "billions": r(F(total, 10**9), 2), "bf16_bytes": 2 * total, "bf16_gb": r(F(2 * total, M.GB), 2),
            "bf16_gib": r(F(2 * total, 2**30), 2), "gib_bytes": 2**30}


def e2_states(specs, n):
    rows = []
    for key, rec in specs["recipes"].items():
        parts = M.model_state_gb(n, rec)
        rows.append({"recipe": key, "label": rec["label"], "bytes_per_param": r(M.recipe_bytes(rec), 2),
                     "parts_gb": {k: r(v, 2) for k, v in parts.items()},
                     "total_gb": r(sum(parts.values()), 2)})
    return rows


def e3_activations(specs, cfg, sc):
    h, a, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    s, b = sc["seq_len"], 1
    parts = M.activation_parts(s, b, h, a)
    sbh, as2b = s * b * h, a * s * s * b
    attn = sum(v for k, v in parts.items() if k.startswith("attn"))
    mlp = sum(v for k, v in parts.items() if k.startswith("mlp"))
    ln = sum(v for k, v in parts.items() if k.startswith("layernorm"))
    per = M.activation_per_layer(s, b, h, a)
    grid = []
    for ss in sc["seq_lens_to_compare"]:
        for bb in sc["micro_batches_to_compare"]:
            grid.append({"seq_len": ss, "micro_batch": bb,
                         "all_gb": r(F(M.activation_total(ss, bb, h, a, L), M.GB), 3),
                         "no_scores_gb": r(F(M.activation_total(ss, bb, h, a, L, keep_scores=False), M.GB), 3),
                         "checkpoint_gb": r(F(M.activation_total(ss, bb, h, a, L, checkpoint=True), M.GB), 3)})
    return {"s": s, "b": b, "h": h, "a": a, "L": L, "sbh": sbh, "as2b": as2b,
            "parts_bytes": parts, "attn_bytes": attn, "attn_check": attn == 11 * sbh + 5 * as2b,
            "mlp_bytes": mlp, "mlp_check": mlp == 19 * sbh, "ln_bytes": ln, "ln_check": ln == 4 * sbh,
            "per_layer_formula": per, "per_layer_sum": sum(parts.values()),
            "formula_equals_sum": per == sum(parts.values()),
            "five_as_over_h": r(F(5 * a * s, h), 4),
            "per_layer_mb": r(F(per, 10**6), 2),
            "per_layer_gb": r(F(per, M.GB), 4), "paren_34_plus": r(34 + F(5 * a * s, h), 4),
            "total_gb": r(F(L * per, M.GB), 3),
            "checkpoint_inputs_gb": r(F(2 * sbh * L, M.GB), 4),
            "checkpoint_peak_gb": r(F(M.activation_total(s, b, h, a, L, checkpoint=True), M.GB), 4),
            "logits_fp32_gb_b1": r(F(s * cfg["vocab_size"] * 4, M.GB), 3),
            "grid": grid,
            "ratio_512_to_1024_all": r(F(M.activation_total(1024, 1, h, a, L), M.activation_total(512, 1, h, a, L)), 2),
            "ratio_1024_to_2048_all": r(F(M.activation_total(2048, 1, h, a, L), M.activation_total(1024, 1, h, a, L)), 2),
            "exercise_512x4": {"sbh": 512 * 4 * h, "paren": r(34 + F(5 * a * 512, h), 4),
                               "gb": r(F(M.activation_total(512, 4, h, a, L), M.GB), 3)},
            "exercise_3b": {"infer_gb": r(F(2 * 3 * 10**9, M.GB), 2), "train16_gb": r(F(16 * 3 * 10**9, M.GB), 2)}}


def e4_checkpoint_sim(cfg, sc):
    h, a, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    s, b = sc["seq_len"], 1
    inp = 2 * s * b * h
    lay = M.activation_per_layer(s, b, h, a)
    rows = []
    for k in [None, 1, 2, 4, 7, 14, 28]:
        sim = M.simulate_checkpointing(L, k, inp, lay)
        rows.append({"k": k, "peak_gb": r(F(sim["peak"], M.GB), 4), "recomputed_layers": sim["recomputed_layers"]})
    k1 = M.simulate_checkpointing(L, 1, inp, lay)["peak"]
    none = M.simulate_checkpointing(L, None, inp, lay)["peak"]
    best = min(rows[1:], key=lambda x: x["peak_gb"])
    return {"rows": rows, "k1_equals_formula": k1 == M.activation_total(s, b, h, a, L, checkpoint=True),
            "none_equals_formula": none == M.activation_total(s, b, h, a, L),
            "best_k": best["k"], "best_peak_gb": best["peak_gb"],
            "time_ratio_assumed": "앞 1 : 뒤 2 가정에서 (1 + 2 + 1) / (1 + 2)",
            "time_ratio": r(F(4, 3), 4)}


def e5_grad_accum():
    batch = [(F(1), F(2)), (F(2), F(3)), (F(3), F(7)), (F(4), F(8))]
    w = F(1)
    full = M.grad_mse(w, batch)
    acc = M.grad_accumulated(w, batch, 4)
    wrong = M.grad_accumulated(w, batch, 4, divide=False)
    return {"batch": [[int(x), int(y)] for x, y in batch], "w": 1,
            "per_example": [int(2 * (w * x - y) * x) for x, y in batch],
            "full_batch_grad": str(full), "accumulated_grad": str(acc), "equal": full == acc,
            "forgot_divide_grad": str(wrong), "forgot_ratio": str(wrong / full)}


def e6_project(specs, cfg, n, sc):
    h, a, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    s = sc["seq_len"]
    R = specs["recipes"]
    gpus = specs["gpus"]

    def act(b, ckpt):
        return F(M.activation_total(s, b, h, a, L, checkpoint=ckpt), M.GB)

    plans = [
        ("full_zero_b4", "전체 학습(16바이트), 묶음 4, 체크포인팅 없음", "zero_mixed_adam", 4, False),
        ("full_zero_b1_ckpt", "전체 학습(16바이트), 묶음 1 × 누적 16, 체크포인팅", "zero_mixed_adam", 1, True),
        ("full_hf_b1_ckpt", "전체 학습(18바이트), 묶음 1 × 누적 16, 체크포인팅", "hf_mixed_adam", 1, True),
        ("adam8_b1_ckpt", "8비트 Adam(12바이트), 묶음 1 × 누적 16, 체크포인팅", "hf_mixed_adam8bit", 1, True),
        ("lora_base_b1_ckpt", "LoRA 미리 보기: 얼린 bf16 + 활성값(LoRA 몫 제외)", "frozen_bf16", 1, True),
        ("qlora_base_b1_ckpt", "QLoRA 미리 보기: 얼린 4비트 + 활성값(보정 상수·LoRA 몫 제외)", "frozen_4bit", 1, True),
    ]
    rows = []
    for key, label, rk, b, ck in plans:
        st = M.recipe_bytes(R[rk]) * n / M.GB
        ac = act(b, ck)
        tot = st + ac
        rows.append({"plan": key, "label": label, "states_gb": r(st, 2), "activations_gb": r(ac, 2),
                     "total_gb": r(tot, 2),
                     "fits": {g["name"]: M.fits(tot, g["memory_gb"]) for g in gpus},
                     "margin_gb": {g["name"]: r(g["memory_gb"] - tot, 2) for g in gpus}})
    return {"seq_len": s, "effective_batch": sc["effective_batch"],
            "gpus": [{"name": g["name"], "memory_gb": g["memory_gb"]} for g in gpus], "rows": rows,
            "infer_bf16_gb": r(M.recipe_bytes(R["infer_bf16"]) * n / M.GB, 2),
            "infer_margin_t4_gb": r(16 - M.recipe_bytes(R["infer_bf16"]) * n / M.GB, 2)}


def main():
    specs = M.load_specs()
    cfg = M.load_config(specs)
    sc = M.load_scenario()
    n = M.count_closed_form(cfg)["total"]
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": M.SPEC_FILE.name, "checked": specs["checked"], "gb": "10^9 bytes"},
        "e0_hand": e0_hand(),
        "e1_params": e1_params(specs, cfg),
        "e2_states": e2_states(specs, n),
        "e3_activations": e3_activations(specs, cfg, sc),
        "e4_checkpoint_sim": e4_checkpoint_sim(cfg, sc),
        "e5_grad_accum": e5_grad_accum(),
        "e6_project": e6_project(specs, cfg, n, sc),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
