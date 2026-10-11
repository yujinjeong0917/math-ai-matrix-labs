"""내 데이터로 LLM 파인튜닝하기 10장 실험. `uv run python finetune/ch10_export_serve/experiments.py` 로 실행한다.

E0 첫 화면 손계산: W' = W0 + (α/r)BA, 합친 표와 옆길 계산이 같은지, 곱셈 횟수
E1 저장 크기: 어댑터 3개를 따로 두기 vs 합친 모델 3벌, safetensors 파일 크기 대조
E2 4비트로 학습한 어댑터 합치기 장난감: 따로 두기 / 되돌린 W에 bf16 / 원래 W에 bf16 / 다시 NF4, rho 훑기
E3 GGUF 파일 크기: 블록 비트, 텐서별 형식 규칙, Qwen 공식 GGUF 파일과 대조, README의 Llama-3.1-8B 비트와 비교
E4 내 컴퓨터에서 띄울 때 메모리: GGUF 파일 + KV 캐시(문맥 4,096토큰)
E5 vLLM 메모리 예산: L4 24GB·T4 16GB, 어댑터 3개, 동시 요청 수 상한
E6 토큰/초 상한: 메모리 대역폭 ÷ 토큰당 읽는 가중치 바이트
E7 한 달 서빙 비용 어림: 1장의 월 3,000건, AWS 온디맨드 가격(2026-10-09 발행 파일)

표준 라이브러리만 쓴다. 무작위 값은 고정 시드(10)라 결과는 매번 같다.
"""

import json
import platform
import struct
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch10_export as M  # noqa: E402

OUT = HERE / "results" / "ch10.json"


def r(x, nd=4):
    return round(float(x), nd)


def fr(M_):
    return [[str(v) for v in row] for row in M_]


def e0_hand(specs):
    ex = specs["from_earlier_chapters"]["ch07_merge_example"]
    h = M.merge_hand(ex["W0"], ex["B"], ex["A"], ex["alpha"], ex["r"], ex["x"])
    return {"W0": ex["W0"], "B": ex["B"], "A": ex["A"], "alpha": ex["alpha"], "r": ex["r"], "x": ex["x"],
            "s": str(h["s"]), "BA": fr(h["BA"]), "sBA": fr(M.scale(h["BA"], h["s"])), "merged": fr(h["merged"]),
            "via_merged": [str(v) for v in h["via_merged"]], "via_side": [str(v) for v in h["via_side"]],
            "same": h["via_merged"] == h["via_side"], "mults_merged": h["mults_merged"], "mults_side": h["mults_side"]}


def e1_storage(specs, sc, gguf_q4km_gb):
    prev = specs["from_earlier_chapters"]["ch07_adapter_files"]
    n = len(sc["assumed"]["adapters"])
    base, ad = prev["base_bf16_gb"], prev["all_linear_r16_bf16_mb"]
    st = M.storage(n, base, ad)
    st_q = M.storage(n, base, ad, merged_gb=gguf_q4km_gb)
    m = specs["model"]
    calc = m["safetensors_total"] * len(struct.pack("<e", 1.0))  # bf16도 fp16처럼 2바이트
    return {"n_adapters": n, "adapters": sc["assumed"]["adapters"], "base_gb": base, "adapter_mb": ad,
            "adapter_share_pct": r(100 * ad / 1000 / base, 2),
            "separate_gb": r(st["separate_gb"], 2), "merged_bf16_gb": r(st["merged_gb"], 2),
            "merged_q4km_gb": r(st_q["merged_gb"], 2),
            "safetensors_calc_bytes": calc, "safetensors_actual_bytes": m["bf16_file_bytes"],
            "safetensors_header_bytes": m["bf16_file_bytes"] - calc}


def e2_toy(sc):
    rows = []
    for rho in sc["toy"]["rho_sweep"]:
        t = M.merge_toy(rho)
        rows.append({"rho": rho,
                     "B_vs_effect": r(t["B_dequant_bf16"]["vs_effect"]), "B_vs_output": r(t["B_dequant_bf16"]["vs_output"], 5),
                     "Btrunc_vs_effect": r(t["B_dequant_bf16_trunc"]["vs_effect"]),
                     "C_vs_effect": r(t["C_original_bf16"]["vs_effect"]), "C_vs_output": r(t["C_original_bf16"]["vs_output"], 5),
                     "D_vs_effect": r(t["D_requant_nf4"]["vs_effect"]), "D_vs_output": r(t["D_requant_nf4"]["vs_output"], 5),
                     "codes_unchanged": t["codes_unchanged"], "codes_total": t["codes_total"],
                     "delta_abs_mean": r(t["delta_abs_mean"], 7)})
    t = M.merge_toy(sc["toy"]["rho_main"])
    h = M.nf4_cell_hand(0.05, 0.0123, 0.0001)
    hand = {"absmax": 0.05, "w": 0.0123, "delta": 0.0001, "ratio": r(0.0123 / 0.05, 3), "code": h["code"],
            "table_value": r(h["table_value"], 4), "back": r(h["back"], 6), "back_plus_delta": r(h["back"] + 0.0001, 6),
            "ratio_after": r((h["back"] + 0.0001) / 0.05, 4), "erased": h["erased"],
            "next_up": r(M.NF4[h["code"] + 1], 4), "next_down": r(M.NF4[h["code"] - 1], 4),
            "half_gap_up": r(h["half_gap_up"], 5), "half_gap_down": r(h["half_gap_down"], 5)}
    return {"setup": sc["toy"], "hand": hand, "quant_err_vs_w_pct": r(100 * t["quant_err_vs_w"], 2),
            "half_gap_near0": r(t["half_gap_near0"], 6), "rows": rows,
            "main": next(x for x in rows if x["rho"] == sc["toy"]["rho_main"])}


def e3_gguf(cfg, specs, sizes):
    blocks = specs["llama_cpp"]["blocks"]
    t = M.gguf_table(cfg, blocks, sizes["files"])
    bits = {k: r(F(v[1] * 8, v[0]), 4) for k, v in blocks.items() if k not in ("F32", "BF16")}
    rows = {}
    for k, v in t["rows"].items():
        rows[k] = {"pred_bytes": v["pred_bytes"], "actual_bytes": v["actual_bytes"], "diff_bytes": v["diff_bytes"],
                   "pred_gb": r(v["pred_gb"], 3), "actual_gb": r(v["actual_bytes"] / M.GB, 3), "bpw": r(v["bpw"], 4),
                   "shards": v["shards"], "by_type_params": v["by_type"],
                   "pct_of_f16": r(100 * v["actual_bytes"] / t["rows"]["F16"]["actual_bytes"], 1)}
    q4 = t["rows"]["Q4_K_M"]["by_type"]
    readme = specs["llama_cpp"]["readme_llama31_8b_bpw"]
    gib = specs["llama_cpp"]["readme_llama31_8b_gib"]
    return {"n_params": t["n_params"], "meta_bytes": t["meta_bytes"], "block_bits": bits, "rows": rows,
            "max_abs_diff_bytes": max(abs(v["diff_bytes"]) for v in t["rows"].values()),
            "q4km_attn_v_q6k_layers": M.q6k_layers("Q4_K_M", "attn_v"),
            "q4km_ffn_down_q6k_layers": M.q6k_layers("Q4_K_M", "ffn_down"),
            "q4km_q6k_share_pct": r(100 * q4["Q6_K"] / t["n_params"], 2),
            "q4km_q4k_share_pct": r(100 * q4["Q4_K"] / t["n_params"], 2),
            "output_params": cfg["vocab_size"] * cfg["hidden_size"],
            "readme_llama31_8b_bpw_q4km": readme["Q4_K_M"], "ours_bpw_q4km": r(t["rows"]["Q4_K_M"]["bpw"], 4),
            "readme_gib_to_gb": {k: r(v * 2**30 / M.GB, 2) for k, v in gib.items()}}


def e4_local(cfg, sc, g):
    kv = M.kv_bytes_per_token(cfg)
    ctx = sc["assumed"]["ctx_local"]
    kv_ctx = kv * ctx
    out = {"kv_bytes_per_token": kv, "ctx": ctx, "kv_ctx_bytes": kv_ctx, "kv_ctx_gb": r(kv_ctx / M.GB, 3), "rows": {}}
    for ft in ["F16", "Q8_0", "Q4_K_M"]:
        a = g["rows"][ft]["actual_bytes"]
        out["rows"][ft] = {"file_gb": r(a / M.GB, 3), "total_gb": r((a + kv_ctx) / M.GB, 2)}
    return out


def e5_vllm(cfg, specs, sc):
    kv = M.kv_bytes_per_token(cfg)
    util = specs["vllm"]["gpu_memory_utilization_default"]
    w = specs["model"]["bf16_file_bytes"]
    req = sc["from_ch01"]["input_tokens"] + sc["from_ch01"]["answer_reserve_tokens"]
    n_ad = len(sc["assumed"]["adapters"])
    ad_bytes = int(n_ad * specs["from_earlier_chapters"]["ch07_adapter_files"]["all_linear_r16_bf16_mb"] * 10**6)
    block = sc["assumed"]["paged_block"]
    out = {"util": util, "weights_gb": r(w / M.GB, 2), "req_tokens": req, "adapters_gb": r(ad_bytes / M.GB, 3)}
    for gpu in ["L4", "T4"]:
        b = M.vllm_budget(specs["gpus"][gpu]["memory_gb"], util, w, kv, req, block, ad_bytes)
        out[gpu] = {k: (r(v, 2) if isinstance(v, float) else v) for k, v in b.items()}
    return out


def e6_speed(specs, g):
    w16 = specs["model"]["bf16_file_bytes"]
    q4 = g["rows"]["Q4_K_M"]["actual_bytes"]
    L4, T4 = specs["gpus"]["L4"]["bandwidth_gb_s"], specs["gpus"]["T4"]["bandwidth_gb_s"]
    tg = specs["llama_cpp"]["readme_llama31_8b_tg"]
    gib = specs["llama_cpp"]["readme_llama31_8b_gib"]
    return {"L4_bf16": r(M.tokens_per_s_upper(L4, w16), 2), "L4_q4km": r(M.tokens_per_s_upper(L4, q4), 2),
            "T4_q4km": r(M.tokens_per_s_upper(T4, q4), 2), "T4_bf16_if_fit": r(M.tokens_per_s_upper(T4, w16), 2),
            "L4_bf16_batch16": r(M.tokens_per_s_upper(L4, w16, batch=16), 1),
            "readme_speed_ratio_q4km_over_f16": r(tg["Q4_K_M"] / tg["F16"], 2),
            "readme_size_ratio_f16_over_q4km": r(gib["F16"] / gib["Q4_K_M"], 2)}


def e7_cost(specs, sc, s6):
    p = specs["aws_prices"]
    calls = sc["from_ch01"]["monthly_calls"]
    ans = sc["assumed"]["answer_tokens_avg"]
    hpm = p["hours_per_month"]
    rows = {}
    for name, inst, tok_s in [("g6_L4_bf16_vllm", "g6.xlarge", s6["L4_bf16"]),
                              ("g4dn_T4_q4km_llamacpp", "g4dn.xlarge", s6["T4_q4km"])]:
        c = M.monthly_cost(calls, ans, tok_s, p[inst]["usd_per_hour"], hpm)
        rows[name] = {"instance": inst, "usd_per_hour": p[inst]["usd_per_hour"], "tok_s_upper": tok_s,
                      "tokens": c["tokens"], "busy_hours": r(c["busy_hours"], 2), "always_on_usd": r(c["always_on_usd"], 2),
                      "busy_only_usd": r(c["busy_only_usd"], 2), "busy_share_pct": r(100 * c["busy_share"], 2),
                      "always_on_per_call_usd": r(c["always_on_per_call_usd"], 3)}
    per_m = M.usd_per_million_tokens(p["g6.xlarge"]["usd_per_hour"], s6["L4_bf16_batch16"])
    return {"monthly_calls": calls, "answer_tokens": ans, "hours_per_month": hpm, "price_published": p["published"],
            "rows": rows, "g6_batch16_usd_per_million": r(per_m, 3),
            "g6_batch16_month_capacity_tokens": int(s6["L4_bf16_batch16"] * 3600 * hpm)}


def main():
    specs, sc, sizes = M.load_specs(), M.load_scenario(), M.load_sizes()
    cfg = M.load_config()
    g = e3_gguf(cfg, specs, sizes)
    s6 = e6_speed(specs, g)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": M.SPEC_FILE.name, "sizes_file": M.SIZES_FILE.name, "checked": specs["checked"],
                 "gb": "10^9 bytes", "model": specs["model"]["name"]},
        "e0_hand": e0_hand(specs),
        "e1_storage": e1_storage(specs, sc, g["rows"]["Q4_K_M"]["actual_bytes"] / M.GB),
        "e2_toy": e2_toy(sc),
        "e3_gguf": g,
        "e4_local": e4_local(cfg, sc, g),
        "e5_vllm": e5_vllm(cfg, specs, sc),
        "e6_speed": s6,
        "e7_cost": e7_cost(specs, sc, s6),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
