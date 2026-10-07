"""13장 비교 실험. `uv run python llm/ch13_scaling/experiments.py` 로 실행한다.

과제: ch13_data.py의 '열쇠-값' 언어(어휘 64, 열쇠 4,096개, 지프 분포). 이론 최저 손실을 식으로 안다.
모델: 9장 Pre-LN TinyLM을 6가지 크기로. 학습: AdamW, warmup 12스텝(모든 학습 공통) + 코사인(최대의 1/10까지), 배치 8문장.
데이터는 한 번씩만 본다.

E0 이론 최저 손실과 균등 추측의 손실
E1 세기: 파라미터 수(식 vs PyTorch), 학습 FLOP(6N vs 행렬곱을 센 값 vs PyTorch FlopCounterMode)
E2 학습률 탐색: 크기마다 D=2^20에서 학습률 5개, 시드 0. 가장 좋은 값을 이후 실험에 고정한다
E3 격자(스케줄을 D에 맞춤): 크기 6 × D 6단계 × 시드 3
E4 격자(스케줄 하나): 크기마다 코사인을 D=2^21에 맞춘 한 번의 학습에서 중간 D의 손실을 읽는다 × 시드 3
E4b 흔한 실패: warmup을 스케줄 길이의 5%로 두면 두 프로토콜 차이에 warmup 차이가 섞인다
E5 같은 계산 예산(IsoFLOP): C = 6ND 세 값에서 크기를 바꿔 가며, 시드 3. '큰 모델이 늘 낫다'의 반례
E6 피팅: Kaplan 꼴 L(N), Chinchilla 꼴 L(N,D)(NumPy 격자 / PyTorch L-BFGS·Huber, E 자유 / E=이론 바닥 고정), 부트스트랩,
    E3과 E4 각각의 a = β/(α+β), 미리 정한 판정 규칙
E7 IsoFLOP의 실측 최적 크기와 피팅이 예측한 N_opt 비교

결과는 results/ch13.json 에 저장하고, 챕터 본문은 이 파일의 값을 그대로 인용한다.
학습은 프로세스 여러 개로 나눠 돌리고, 프로세스마다 PyTorch 스레드를 1로 고정한다.
"""

import json
import math
import platform
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import ch13_data as cd
import scaling_np as sn

OUT = Path(__file__).parent / "results" / "ch13.json"
SIZES = [
    {"name": "S1", "d": 16, "L": 1, "h": 2},
    {"name": "S2", "d": 24, "L": 2, "h": 2},
    {"name": "S3", "d": 32, "L": 2, "h": 4},
    {"name": "S4", "d": 48, "L": 3, "h": 4},
    {"name": "S5", "d": 64, "L": 4, "h": 4},
    {"name": "S6", "d": 96, "L": 4, "h": 4},
]
D_GRID = [2**16, 2**17, 2**18, 2**19, 2**20, 2**21]
SEEDS = (0, 1, 2)
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ISO_C = (3e10, 1.2e11, 4.8e11)
ISO_D_RANGE = (2**14, 2**23)
WORKERS = 10
RULE = (
    "실험 전에 정한 판정 규칙: 두 프로토콜(E3 스케줄을 D에 맞춤, E4 긴 스케줄 하나를 중간에서 읽기)에서 "
    "피팅한 a = β/(α+β)의 부트스트랩 95% 구간이 겹치지 않으면 '배분 결론이 바뀌었다'고 쓴다. 겹치면 "
    "'이 규모에서는 결론을 가를 만큼 차이가 나지 않았다'고 쓴다. 손실 자체의 비교는 같은 (N, D, 시드) 칸끼리 한다."
)


def _job(args):
    import scaling_torch as st

    torch.set_num_threads(1)
    kind, i, tokens, seed, lr, extra = args
    r = st.train_run(SIZES[i], tokens, seed, lr, **extra)
    r.update({"kind": kind, "size": SIZES[i]["name"], "seed": seed, "lr": lr})
    return r


def run_pool(jobs):
    jobs = sorted(jobs, key=lambda j: -j[2] * (SIZES[j[1]]["d"] ** 2 * SIZES[j[1]]["L"] + 2000))  # 큰 일부터
    with Pool(WORKERS) as p:
        return list(p.imap_unordered(_job, jobs))


# ---------------------------------------------------------------- E0, E1


def e0_floor():
    f = cd.entropy_floor()
    cov = {str(D): cd.key_coverage(D) for D in D_GRID}
    return {**f, "V": cd.V, "n_keys": cd.N_KEYS, "zipf": cd.ZIPF, "p_rule": cd.P_RULE, "n_ctx": cd.N_CTX,
            "top_key_prob": float(cd.P_KEY.max()), "top10_key_mass": float(np.sort(cd.P_KEY)[::-1][:10].sum()),
            "key_coverage": cov}


def e1_counts():
    import scaling_torch as st
    from torch.utils.flop_counter import FlopCounterMode

    rows = []
    for cfg in SIZES:
        m = st.build(cfg, 0)
        n_t, tot_t = st.torch_param_counts(m)
        n, tot = sn.count_params(cfg)
        f = sn.flops_per_token(cfg)
        x = torch.randint(0, cd.V, (1, cd.N_CTX))
        with FlopCounterMode(display=False) as fc:
            m(x)
        rows.append({
            "size": cfg["name"], **{k: cfg[k] for k in ("d", "L", "h")},
            "N": n, "N_total": tot, "torch_N": n_t, "torch_N_total": tot_t,
            "approx_6N": f["approx_6N"], "exact_train": f["exact_train"],
            "ratio_exact_over_6N": f["ratio_exact_over_6N"], "context_share_of_forward": f["context_share_of_forward"],
            "flopcounter_forward_per_token": fc.get_total_flops() / cd.N_CTX, "exact_forward": f["exact_forward"],
        })
    return rows


# ---------------------------------------------------------------- 학습 실험


def e2_lr_sweep():
    res = run_pool([("lr", i, 2**20, 0, lr, {}) for i in range(len(SIZES)) for lr in LRS])
    table = {c["name"]: {str(r["lr"]): r["val_loss"] for r in res if r["size"] == c["name"]} for c in SIZES}
    best = {k: float(min(v, key=lambda lr: v[lr])) for k, v in table.items()}
    return {"D": 2**20, "seed": 0, "val_loss": table, "best_lr": best}


def e3_e4_grid(best_lr, warmup_frac=None):
    jobs = []
    w = {} if warmup_frac is None else {"warmup_frac": warmup_frac}
    for i, c in enumerate(SIZES):
        lr = best_lr[c["name"]]
        for s in SEEDS:
            for D in D_GRID:
                jobs.append(("matched", i, D, s, lr, dict(w)))
            jobs.append(("single", i, D_GRID[-1], s, lr, {"eval_at_tokens": tuple(D_GRID[:-1]), **w}))
    res = run_pool(jobs)
    matched = [r for r in res if r["kind"] == "matched"]
    single = []
    for r in res:
        if r["kind"] != "single":
            continue
        for D in D_GRID:
            v = r["val_loss"] if D == D_GRID[-1] else r["intermediate_val_loss"][str(D)]
            single.append({"size": r["size"], "seed": r["seed"], "N": r["N"], "D": D, "val_loss": v, "lr": r["lr"]})
    key = lambda r: (r["size"], r["D"], r["seed"])  # noqa: E731
    matched.sort(key=key)
    single.sort(key=key)
    slim = [{k: r[k] for k in ("size", "seed", "N", "N_total", "D", "val_loss", "lr", "seconds", "nan")} for r in matched]
    return slim, single


def e5_isoflop(best_lr):
    jobs = []
    for C in ISO_C:
        for i, c in enumerate(SIZES):
            n, _ = sn.count_params(c)
            D = int(round(C / (6 * n) / (8 * cd.N_CTX))) * 8 * cd.N_CTX  # 배치 크기의 배수로
            if not (ISO_D_RANGE[0] <= D <= ISO_D_RANGE[1]):
                continue
            for s in SEEDS:
                jobs.append(("iso", i, D, s, best_lr[c["name"]], {"_C": C}))
    # _C는 학습 함수에 넘기지 않는다
    tagged = {(j[1], j[2], j[3]): j[5]["_C"] for j in jobs}
    jobs = [(k, i, D, s, lr, {}) for (k, i, D, s, lr, _) in jobs]
    res = run_pool(jobs)
    out = []
    for r in res:
        i = [c["name"] for c in SIZES].index(r["size"])
        out.append({"C": tagged[(i, r["D"], r["seed"])], "size": r["size"], "seed": r["seed"], "N": r["N"],
                    "D": r["D"], "C_6ND": 6 * r["N"] * r["D"], "val_loss": r["val_loss"]})
    out.sort(key=lambda r: (r["C"], r["N"], r["seed"]))
    summary = []
    for C in ISO_C:
        rows = [r for r in out if r["C"] == C]
        by_n = {}
        for r in rows:
            by_n.setdefault((r["size"], r["N"], r["D"]), []).append(r["val_loss"])
        cells = [{"size": k[0], "N": k[1], "D": k[2], "tokens_per_param": k[2] / k[1],
                  "mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v} for k, v in sorted(by_n.items(), key=lambda kv: kv[0][1])]
        best = min(cells, key=lambda c: c["mean"])
        summary.append({"C": C, "cells": cells, "best_size": best["size"], "best_N": best["N"], "best_D": best["D"]})
    return out, summary


# ---------------------------------------------------------------- 피팅


def _cell_means(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["N"], r["D"]), []).append(r["val_loss"])
    keys = sorted(cells)
    N = np.array([k[0] for k in keys], float)
    D = np.array([k[1] for k in keys], float)
    return N, D, cells, keys


def _fit_both(N, D, L, E_fixed=None):
    import scaling_torch as st

    p_np = sn.fit_chinchilla(N, D, L, E_fixed=E_fixed)
    p_t = st.fit_chinchilla_torch(N, D, L, E_fixed=E_fixed)
    return p_np, p_t


def bootstrap(rows, reps=200, seed=0, E_fixed=None):
    """칸마다 시드 3개를 복원추출해 평균을 다시 내고 다시 맞춘다(NumPy 피팅)."""
    N, D, cells, keys = _cell_means(rows)
    rng = np.random.default_rng(seed)
    out = {"alpha": [], "beta": [], "E": [], "a": []}
    for _ in range(reps):
        L = np.array([np.mean(rng.choice(cells[k], size=len(cells[k]))) for k in keys])
        p = sn.fit_chinchilla(N, D, L, refine=1, E_fixed=E_fixed)
        out["alpha"].append(p["alpha"])
        out["beta"].append(p["beta"])
        out["E"].append(p["E"])
        out["a"].append(p["beta"] / (p["alpha"] + p["beta"]))
    return {k: {"p2.5": float(np.percentile(v, 2.5)), "p50": float(np.median(v)), "p97.5": float(np.percentile(v, 97.5))}
            for k, v in out.items()}


def e6_fits(matched, single, floor):
    res = {"rule": RULE}
    for name, rows in (("matched", matched), ("single", single)):
        N, D, cells, keys = _cell_means(rows)
        L = np.array([np.mean(cells[k]) for k in keys])
        p_np, p_t = _fit_both(N, D, L)
        f_np, f_t = _fit_both(N, D, L, E_fixed=floor)
        a = lambda p: p["beta"] / (p["alpha"] + p["beta"])  # noqa: E731
        res[name] = {
            "numpy_fit": p_np, "torch_fit": p_t, "a_numpy": a(p_np), "a_torch": a(p_t),
            "E_minus_floor_numpy": p_np["E"] - floor, "bootstrap": bootstrap(rows),
            "fixedE_numpy_fit": f_np, "fixedE_torch_fit": f_t, "fixedE_a_numpy": a(f_np), "fixedE_a_torch": a(f_t),
            "fixedE_bootstrap": bootstrap(rows, E_fixed=floor),
            "cells": [{"N": k[0], "D": k[1], "mean": float(np.mean(cells[k])), "std": float(np.std(cells[k]))} for k in keys],
        }
    # Kaplan 꼴: 가장 큰 D에서 L(N), 가장 큰 N에서 L(D)
    m = res["matched"]["cells"]
    big_d = [c for c in m if c["D"] == D_GRID[-1]]
    big_n = max(c["N"] for c in m)
    res["kaplan_L_of_N_at_Dmax"] = sn.fit_power_law([c["N"] for c in big_d], [c["mean"] for c in big_d])
    res["kaplan_L_of_N_at_Dmax_minus_floor"] = sn.fit_power_law([c["N"] for c in big_d], [c["mean"] - floor for c in big_d])
    res["kaplan_L_of_D_at_Nmax"] = sn.fit_power_law([c["D"] for c in m if c["N"] == big_n], [c["mean"] for c in m if c["N"] == big_n])
    res["protocol_diff"] = protocol_diff(matched, single)
    verdict = {}
    for tag in ("bootstrap", "fixedE_bootstrap"):
        bm, bs = res["matched"][tag]["a"], res["single"][tag]["a"]
        overlap = not (bm["p97.5"] < bs["p2.5"] or bs["p97.5"] < bm["p2.5"])
        verdict[tag] = {"a_intervals_overlap": overlap, "conclusion_changed": not overlap}
    res["verdict"] = verdict
    return res


def protocol_diff(matched, single):
    """같은 (크기, D, 시드) 칸에서 '긴 스케줄의 중간 읽기 - D에 맞춘 스케줄'."""
    diffs = []
    s_by = {(r["size"], r["D"], r["seed"]): r["val_loss"] for r in single}
    for r in matched:
        if r["D"] == D_GRID[-1]:
            continue
        diffs.append({"size": r["size"], "D": r["D"], "seed": r["seed"], "single_minus_matched": s_by[(r["size"], r["D"], r["seed"])] - r["val_loss"]})
    dv = np.array([d["single_minus_matched"] for d in diffs])
    sizes = [c["name"] for c in SIZES]
    return {
        "n_cells": len(dv), "n_single_higher": int((dv > 0).sum()), "mean": float(dv.mean()),
        "by_D": {str(D): float(np.mean([d["single_minus_matched"] for d in diffs if d["D"] == D])) for D in D_GRID[:-1]},
        "by_size_D": {s: {str(D): float(np.mean([d["single_minus_matched"] for d in diffs if d["D"] == D and d["size"] == s]))
                          for D in D_GRID[:-1]} for s in sizes},
        "n_single_higher_by_D": {str(D): int(sum(d["single_minus_matched"] > 0 for d in diffs if d["D"] == D)) for D in D_GRID[:-1]},
    }


def e7_compare(iso_summary, fit):
    out = {}
    for tag in ("numpy_fit", "fixedE_numpy_fit"):
        p = fit["matched"][tag]
        rows = []
        for s in iso_summary:
            pred = sn.optimal_allocation(s["C"], p)
            rows.append({"C": s["C"], "measured_best_N": s["best_N"], "measured_best_size": s["best_size"],
                         "measured_best_tokens_per_param": s["best_D"] / s["best_N"],
                         "predicted_N_opt": float(pred["N_opt"]), "predicted_D_opt": float(pred["D_opt"]),
                         "predicted_tokens_per_param": float(pred["D_opt"] / pred["N_opt"])})
        out[tag] = {"a": sn.optimal_allocation(1.0, p)["a"], "rows": rows}
    best = [s["best_N"] for s in iso_summary]
    out["measured_a_from_isoflop_minima"] = float(np.polyfit(np.log(ISO_C), np.log(best), 1)[0])
    return out


def main():
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    results = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "torch_threads_per_worker": 1, "workers": WORKERS},
        "setup": {"sizes": SIZES, "D_grid": D_GRID, "seeds": list(SEEDS), "lrs": list(LRS), "batch_seqs": 8,
                  "schedule": "linear warmup 12 steps (all runs) + cosine to 0.1x peak over schedule length, AdamW wd=0", "iso_C": list(ISO_C)},
    }
    results["E0"] = e0_floor()
    results["E1"] = e1_counts()
    results["E2"] = e2_lr_sweep()
    print("E2", results["E2"]["best_lr"], flush=True)
    matched, single = e3_e4_grid(results["E2"]["best_lr"])
    results["E3"], results["E4"] = matched, single
    print("E3/E4 done", flush=True)
    m5, s5 = e3_e4_grid(results["E2"]["best_lr"], warmup_frac=0.05)
    results["E4b"] = {"note": "warmup을 스케줄 길이의 5%로 잡은 경우. 긴 스케줄은 warmup도 길어져 두 효과가 섞인다(7절 흔한 실패).",
                      "warmup_frac": 0.05, "protocol_diff": protocol_diff(m5, s5),
                      "matched": m5, "single": s5}
    print("E4b done", flush=True)
    iso, iso_sum = e5_isoflop(results["E2"]["best_lr"])
    results["E5"] = {"runs": iso, "summary": iso_sum}
    print("E5 done", flush=True)
    results["E6"] = e6_fits(matched, single, results["E0"]["floor"])
    results["E7"] = e7_compare(iso_sum, results["E6"])
    results["wall_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in ("E0", "E2", "E7", "wall_seconds")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in results["E6"].items() if k not in ("matched", "single")}, indent=1, ensure_ascii=False))
    for k in ("matched", "single"):
        e = results["E6"][k]
        print(k, {x: e[x] for x in e if x != "cells"})
    for s in results["E5"]["summary"]:
        print(s["C"], [(c["size"], c["D"], round(c["tokens_per_param"], 1), round(c["mean"], 3)) for c in s["cells"]])


if __name__ == "__main__":
    main()
