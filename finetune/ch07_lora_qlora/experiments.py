"""내 데이터로 LLM 파인튜닝하기 7장 실험. `uv run python finetune/ch07_lora_qlora/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 4×4 랭크 1 변화량을 B(4×1)·A(1×4)로, 곱셈 횟수, 같은 결과
E1 Qwen2.5-7B 선형층 모양과 LoRA 학습 파라미터 수(붙이는 곳 × 랭크)
E2 배율 α/r, rsLoRA α/√r, 무작위 부호 장난감으로 본 r에 따른 크기
E3 작은 행렬 분해 장난감: 특이값을 아는 4×4, 랭크별 근사 오차, 거듭제곱법, 기울기 하강으로 B·A 학습
E4 합치기: (W0 + sBA)x = W0x + sB(Ax), 시작할 때 BA = 0
E5 LoRA(얼린 bf16) 합계: 5장 15.71GB + LoRA 몫
E6 QLoRA 사다리: 5장 3.81GB(4.28GB + α)에서 실제에 가까운 합계로, 변형(두 번 양자화, fp32로 올리기, bf16 어댑터)
E7 페이징 옵티마이저 장난감: 길이를 늘릴 때 GPU에 들어가는 한계, 내보내는 시간 어림
E8 어댑터 파일 크기

표준 라이브러리만 쓴다. 무작위 값은 고정 시드(7, 11)라 결과는 매번 같다.
"""

import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch07_lora as M  # noqa: E402

OUT = HERE / "results" / "ch07.json"


def r(x, nd=4):
    return round(float(x), nd)


def gb(x, nd=2):
    return round(float(F(x) / M.GB), nd)


def e0_hand():
    Bv = [2, 1, 0, -1]
    Av = [1, 2, 3, 4]
    dW = M.outer(Bv, Av)
    x = [1, 0, 0, 1]
    ax = sum(a * b for a, b in zip(Av, x))
    return {"B": Bv, "A": Av, "deltaW": dW, "numbers_full": 16, "numbers_lora": 8, "x": x,
            "Ax": ax, "deltaW_x": M.matvec(dW, x), "B_Ax": [b * ax for b in Bv],
            "same": M.matvec(dW, x) == [b * ax for b in Bv],
            "mults_full": 16, "mults_lora": 8,
            "big_example": {"d": 4096, "r": 8, "full": 4096 * 4096, "lora": 2 * 4096 * 8,
                            "pct": r(100 * F(2 * 4096 * 8, 4096 * 4096), 2)}}


def e1_params(cfg, n):
    shapes = M.linear_shapes(cfg)
    ranks = [1, 4, 8, 16, 32, 64]
    table = {}
    for t, mods in M.TARGETS.items():
        table[t] = {"modules": mods, "per_layer_coeff": M.lora_per_layer_coeff(cfg, mods),
                    "by_r": {str(k): M.lora_params(cfg, mods, k) for k in ranks},
                    "pct_by_r": {str(k): r(100 * F(M.lora_params(cfg, mods, k), n), 3) for k in ranks},
                    "millions_by_r": {str(k): r(F(M.lora_params(cfg, mods, k), 10**6), 2) for k in ranks}}
    q = shapes["q_proj"]
    return {"shapes": {k: list(v) for k, v in shapes.items()}, "layers": cfg["num_hidden_layers"],
            "q_full": q[0] * q[1], "q_lora_r8": M.lora_params_module(*q, 8),
            "q_pct_r8": r(100 * F(M.lora_params_module(*q, 8), q[0] * q[1]), 3),
            "k_full": shapes["k_proj"][0] * shapes["k_proj"][1], "k_lora_r8": M.lora_params_module(*shapes["k_proj"], 8),
            "down_full": shapes["down_proj"][0] * shapes["down_proj"][1],
            "down_lora_r8": M.lora_params_module(*shapes["down_proj"], 8),
            "total_params": n, "table": table,
            "allr1_equals_qv_r8": table["all_linear"]["by_r"]["1"] == table["qv"]["by_r"]["8"]}


def e2_scaling():
    rows = []
    for alpha, rr, note in [(8, 8, "PEFT 기본값"), (16, 8, ""), (16, 16, ""), (16, 64, "QLoRA 논문 부록 B.2"),
                            (32, 16, ""), (64, 64, "")]:
        rows.append({"alpha": alpha, "r": rr, "alpha_over_r": r(M.scaling(alpha, rr)),
                     "rslora": r(M.scaling(alpha, rr, rslora=True)), "note": note})
    growth = M.random_sign_growth()
    g = {str(k): r(v, 2) for k, v in growth.items()}
    alpha = 16
    return {"rows": rows, "random_sign_rms": g, "d": 256, "trials": 200, "seed": 11,
            "after_alpha_over_r": {str(k): r(v * alpha / k, 2) for k, v in growth.items()},
            "after_rslora": {str(k): r(v * alpha / k ** 0.5, 2) for k, v in growth.items()},
            "exercise": {"r": 32, "alpha": 64, "alpha_over_r": r(M.scaling(64, 32)), "rslora": r(M.scaling(64, 32, rslora=True), 2)},
            "aligned_example": {"c": 1, "x_sum": 1, "r_values": [1, 2, 4],
                                "BAx": [k * 1 for k in (1, 2, 4)], "scaled_alpha_over_r_alpha2": [2, 2, 2]}}


def e3_toy():
    sig = (F(3), F(1), F(1, 10))
    Mx = M.toy_matrix(sig)
    rows = []
    for k in (1, 2, 3):
        err2 = M.frob2(M.add(Mx, M.truncated(sig, k), -1))
        dropped = sum(s * s for s in sig[k:])
        tr = M.train_lora_toy(Mx, k)
        rows.append({"r": k, "params": k * (4 + 4), "err2": str(err2), "err2_float": r(err2, 6),
                     "dropped_sigma2": str(dropped), "eckart_young_holds": err2 == dropped,
                     "share_kept_pct": r(100 * (1 - F(err2) / M.frob2(Mx)), 3),
                     "gd_start_err2": r(tr["start_err2"], 6), "gd_end_err2": r(tr["end_err2"], 6),
                     "gd_start_is_zero": tr["start_is_zero"],
                     "gd_matches_svd": abs(tr["end_err2"] - float(err2)) < 1e-6})
    s1, v = M.power_iteration(Mx)
    return {"sigmas": [str(s) for s in sig], "matrix": [[str(a) for a in row] for row in Mx],
            "matrix_float": [[r(a, 3) for a in row] for row in Mx], "full_params": 16,
            "frob2": str(M.frob2(Mx)), "rows": rows, "power_steps": 60, "power_sigma1": r(s1, 6), "power_v1": [r(t, 4) for t in v],
            "gd": {"steps": 4000, "lr": 0.05, "seed": 7, "init_std": 0.1}}


def e4_merge():
    W0 = [[F(1), F(0)], [F(0), F(1)]]
    B = [[F(1)], [F(2)]]
    A = [[F(3), F(-1)]]
    x = [F(2), F(1)]
    s = F(16, 8)
    merged, sep = M.merged_equals_separate(W0, B, A, x, s)
    Bz = [[F(0)], [F(0)]]
    m0, _ = M.merged_equals_separate(W0, Bz, A, x, s)
    return {"W0": "단위 행렬 2×2", "B": [1, 2], "A": [3, -1], "x": [2, 1], "s": "16/8 = 2",
            "BA": [[str(a) for a in row] for row in M.matmul(B, A)],
            "merged": [str(t) for t in merged], "separate": [str(t) for t in sep], "equal": merged == sep,
            "B_zero_output": [str(t) for t in m0], "B_zero_is_noop": m0 == M.matvec(W0, x)}


def e5_lora_bf16(cfg):
    rows = []
    for rr, t in [(8, "qv"), (16, "qv"), (8, "all_linear"), (16, "all_linear"), (64, "all_linear")]:
        x = M.lora_bf16_total(cfg, rr, t)
        rows.append({"r": rr, "targets": t, "lora_params": x["lora_params"], "base_gb": gb(x["base"]),
                     "lora_gb": gb(x["lora"], 3), "act_gb": gb(x["act"]), "total_gb": gb(x["total"]),
                     "margin_t4_gb": r(16 - F(x["total"]) / M.GB, 2), "fits_t4": x["total"] <= 16 * M.GB})
    return {"rows": rows, "ch05_preview_gb": gb(M.lora_bf16_total(cfg, 8, "qv")["base"] +
                                                    M.lora_bf16_total(cfg, 8, "qv")["act"])}


def ladder_row(cfg, **kw):
    L = M.qlora_ladder(cfg, **kw)
    return {"steps": [{"label": k, "gb": r(F(v) / M.GB, 4)} for k, v in L["steps"]],
            "total_gb": gb(L["total"]), "total_gb4": r(F(L["total"]) / M.GB, 4),
            "alpha_gb": r(F(L["alpha_part"]) / M.GB, 4), "margin_t4_gb": r(16 - F(L["total"]) / M.GB, 2),
            "lora_params": L["lora_params"], "fits_t4": L["total"] <= 16 * M.GB}


def e6_qlora(cfg):
    sp = M.base_split(cfg)
    main = ladder_row(cfg)  # 두 번 양자화, 남는 층 bf16, all-linear r=16, PEFT fp32 어댑터
    variants = {
        "no_dq": ladder_row(cfg, double_quant=False),
        "kbit_prep_fp32": ladder_row(cfg, kept_bytes=4),
        "paper_bf16_adapter": ladder_row(cfg, lora_path="paper_bf16"),
        "qv_r8": ladder_row(cfg, r=8, targets="qv"),
        "all_r64": ladder_row(cfg, r=64),
    }
    n = sp["total"]
    nf4 = M.nf4_bits()
    dq = M.nf4_bits(double_quant=True)
    return {"split": sp, "kept_pct": r(100 * F(sp["kept"], n), 1),
            "kept_bf16_gb": gb(2 * sp["kept"]), "kept_fp32_gb": gb(4 * sp["kept"]),
            "linear_nf4_gb": gb(sp["linear_4bit"] * nf4 / 8), "linear_nf4_dq_gb": gb(sp["linear_4bit"] * dq / 8),
            "linear_code_only_gb": gb(F(sp["linear_4bit"], 2)),
            "all_params_nf4_gb": gb(n * nf4 / 8), "ch06_7e9_nf4_gb": r(F(7 * 10**9) * nf4 / 8 / M.GB, 4),
            "nf4_bits": str(nf4), "dq_bits": r(dq, 6),
            "base_weights_main_gb": gb(sp["linear_4bit"] * dq / 8 + 2 * sp["kept"]),
            "ch05_preview_gb": gb(F(n, 2)),
            "ch05_preview_plus_act_gb": gb(F(n, 2) + M.activation_ckpt_bytes(1024, 1, 3584, 28, 28)),
            "linear_nf4_dq_gb3": r(F(sp["linear_4bit"]) * dq / 8 / M.GB, 3),
            "kept_bf16_gb3": r(F(2 * sp["kept"], M.GB), 3),
            "base_weights_main_gb3": r((F(sp["linear_4bit"]) * dq / 8 + 2 * sp["kept"]) / M.GB, 3),
            "kept_extra_vs_preview_gb": r(F(2 * sp["kept"] - F(sp["kept"], 2)) / M.GB, 4),
            "bars_pct_of_16gb": {"linear": r(100 * F(sp["linear_4bit"]) * dq / 8 / (16 * M.GB), 2),
                                 "kept": r(100 * F(2 * sp["kept"], 16 * M.GB), 2),
                                 "lora": r(100 * F(main["lora_params"] * 16, 16 * M.GB), 2),
                                 "act": r(100 * F(M.activation_ckpt_bytes(1024, 1, 3584, 28, 28), 16 * M.GB), 2),
                                 "total": r(100 * F(main["total_gb4"]) / 16, 2)},
            "main": main, "variants": variants,
            "lora_optim_gb": gb(main["lora_params"] * 8, 3)}


def e7_paged(cfg, main_total_bytes, lora_params):
    h, a, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    act1024 = M.activation_ckpt_bytes(1024, 1, h, a, L)
    fixed = main_total_bytes - act1024
    optim = lora_params * 8
    w = M.paged_window(fixed, optim, 16 * M.GB, h, a, L)
    acts = {str(s): gb(M.activation_ckpt_bytes(s, 1, h, a, L)) for s in (1024, 2048, 4096, 6144, 7168, 8192)}
    return {"fixed_gb": gb(fixed), "optim_gb": gb(optim, 3), "act_by_len_gb": acts,
            "total_by_len_gb": {s: r((F(fixed) + M.activation_ckpt_bytes(int(s), 1, h, a, L)) / M.GB, 2) for s in acts},
            "window": w, "overhead_ms_for_optim": r(M.paging_overhead_ms(float(F(optim, M.GB))), 1),
            "overhead_ms_1gb_doc": r(M.paging_overhead_ms(1.0), 1)}


def e8_files(cfg):
    out = {}
    for rr, t in [(8, "qv"), (16, "all_linear"), (64, "all_linear")]:
        p = M.lora_params(cfg, M.TARGETS[t], rr)
        out[f"{t}_r{rr}"] = {"params": p, "bf16_mb": r(F(p * M.bytes_per_number("bf16"), 10**6), 1),
                             "fp32_mb": r(F(p * M.bytes_per_number("fp32"), 10**6), 1)}
    out["base_bf16_gb"] = gb(2 * M.total_params(cfg))
    return out


def main():
    specs = M.load_specs()
    cfg = M.load_config(specs)
    n = M.total_params(cfg)
    assert n == specs["model"]["safetensors_total"]
    e6 = e6_qlora(cfg)
    mainL = M.qlora_ladder(cfg)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": M.SPEC_FILE.name, "checked": specs["checked"], "gb": "10^9 bytes",
                 "main_recipe": "NF4 + 두 번 양자화(선형층), 나머지 bf16, LoRA all-linear r=16, PEFT 기본 fp32 어댑터 + AdamW, 길이 1,024, 묶음 1, 체크포인팅"},
        "e0_hand": e0_hand(),
        "e1_params": e1_params(cfg, n),
        "e2_scaling": e2_scaling(),
        "e3_toy": e3_toy(),
        "e4_merge": e4_merge(),
        "e5_lora_bf16": e5_lora_bf16(cfg),
        "e6_qlora": e6,
        "e7_paged": e7_paged(cfg, mainL["total"], mainL["lora_params"]),
        "e8_files": e8_files(cfg),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
