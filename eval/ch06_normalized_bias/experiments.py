"""모델 평가 6장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch06_normalized_bias/experiments.py` 로 실행한다.

E0 첫 화면: 4시간 사용량. MAE·RMSE가 같은 두 모델(번갈아 틀림 A, 늘 모자람 B)의 NMBE와 CV(RMSE)
E1 실패: 1년(8760시간) 합성 부하. A는 편향 없이 잡음 큼(표준편차 25), B는 8% 모자람 + 잡음 작음(표준편차 5).
   RMSE로는 B가 낫지만, 월 합계로 보면 A는 기준을 넘고 B는 떨어진다.
E2 시드 100개에서 ASHRAE 2002 기준 통과 횟수(시간 단위, 월 단위)
E3 B의 편향 크기, A의 잡음 크기를 바꿔 가며 통과 경계 찾기
E4 상수 보정: 평균 오차만큼 더하면 NMBE 0, CV(RMSE)는 편향 조각만큼 줄어든다
E5 부호와 분모 규약: ASHRAE(p=1, p=0), FEMP, Solar Forecast Arbiter를 같은 예측에 대기
E6 NMAE 분모 세 가지: 같은 계열에서는 순위가 안 바뀌고, 발전소가 다르면 바뀔 수 있다
E7 손 계산 fixture와 NumPy-PyTorch 대조
결과는 results/ch06.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
"""

import json
import os
import platform
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch06_cases as cases  # noqa: E402
import eval_ch06_metrics as m  # noqa: E402
import eval_ch06_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch06.json"
SEEDS = 100


def q(a, p=(5, 50, 95)):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), **{f"p{k}": float(np.percentile(a, k)) for k in p}}


def all_metrics(y, yhat, p=1):
    d = m.mse_decomposition(y, yhat)
    return {"mae": m.mae(y, yhat), "rmse": m.rmse(y, yhat), "mbe": m.mbe(y, yhat),
            "nmbe": m.nmbe(y, yhat, p=p), "cv_rmse": m.cv_rmse(y, yhat, p=p),
            "bias_sq": d["bias_sq"], "var": d["var"], "bias_share": d["bias_share"]}


def e0_first_screen():
    y, a, b = cases.first_screen()
    out = {"y": y.tolist(), "A": a.tolist(), "B": b.tolist(), "mean_y": float(y.mean())}
    for k, p in (("A", a), ("B", b)):
        out[k + "_errors"] = m.errors(y, p).tolist()
        out[k + "_n_denominator"] = all_metrics(y, p, p=0)
        out[k + "_ashrae_p1"] = {"nmbe": m.nmbe(y, p, p=1), "cv_rmse": m.cv_rmse(y, p, p=1)}
        corrected = p + m.mbe(y, p)
        out[k + "_after_offset"] = {"shift": m.mbe(y, p), "rmse": m.rmse(y, corrected), "nmbe": m.nmbe(y, corrected, p=0)}
    return out


def models(seed, bias=0.08, sd_a=25.0, sd_b=5.0):
    y = cases.base_load(seed=seed)
    return y, cases.case_unbiased_noisy(y, sd=sd_a, seed=seed), cases.case_biased_steady(y, bias=bias, sd=sd_b, seed=seed)


def e1_failure():
    y, a, b = models(0)
    ym, am = cases.aggregate_monthly(y, a)
    _, bm = cases.aggregate_monthly(y, b)
    return {
        "seed": 0, "hours": len(y), "mean_y_hourly": float(y.mean()), "mean_y_monthly": float(ym.mean()),
        "hourly": {"A": {**all_metrics(y, a), "check": m.ashrae_check(y, a, "hourly")},
                   "B": {**all_metrics(y, b), "check": m.ashrae_check(y, b, "hourly")}},
        "monthly": {"A": {**all_metrics(ym, am), "check": m.ashrae_check(ym, am, "monthly")},
                    "B": {**all_metrics(ym, bm), "check": m.ashrae_check(ym, bm, "monthly")}},
        "monthly_totals_first3": {"y": ym[:3].tolist(), "A": am[:3].tolist(), "B": bm[:3].tolist()},
    }


def run_seeds(bias=0.08, sd_a=25.0, seeds=SEEDS, base=1000):
    keys = ("A_hourly", "A_monthly", "B_hourly", "B_monthly")
    passes = {k: 0 for k in keys}
    fails_by = {k: {"nmbe_only": 0, "cv_only": 0, "both": 0} for k in keys}
    vals = {k: {"nmbe": [], "cv_rmse": []} for k in keys}
    rmse_picks_b = 0
    for s in range(base, base + seeds):
        y, a, b = models(s, bias=bias, sd_a=sd_a)
        ym, am = cases.aggregate_monthly(y, a)
        _, bm = cases.aggregate_monthly(y, b)
        rmse_picks_b += m.rmse(y, b) < m.rmse(y, a)
        for k, (yy, pp, res) in {"A_hourly": (y, a, "hourly"), "A_monthly": (ym, am, "monthly"),
                                 "B_hourly": (y, b, "hourly"), "B_monthly": (ym, bm, "monthly")}.items():
            c = m.ashrae_check(yy, pp, res)
            passes[k] += c["pass"]
            if not c["pass"]:
                fails_by[k]["both" if not (c["nmbe_ok"] or c["cv_ok"]) else ("nmbe_only" if c["cv_ok"] else "cv_only")] += 1
            vals[k]["nmbe"].append(c["nmbe"]); vals[k]["cv_rmse"].append(c["cv_rmse"])
    return {"passes": {k: int(v) for k, v in passes.items()}, "rmse_picks_B": int(rmse_picks_b), "fails_by": fails_by,
            "cv_max": {k: float(np.max(d["cv_rmse"])) for k, d in vals.items()},
            "summary": {k: {kk: q(vv) for kk, vv in d.items()} for k, d in vals.items()}}


def e2_seeds():
    return {"seeds": SEEDS, "bias": 0.08, "sd_a": 25.0, **run_seeds()}


def e3_sweeps():
    bias_rows = {}
    for bias in (0.02, 0.04, 0.045, 0.05, 0.06, 0.08, 0.10, 0.12):
        r = run_seeds(bias=bias, seeds=SEEDS, base=3000)
        bias_rows[str(bias)] = {"B_hourly_pass": r["passes"]["B_hourly"], "B_monthly_pass": r["passes"]["B_monthly"],
                                "B_hourly_nmbe_mean": r["summary"]["B_hourly"]["nmbe"]["mean"],
                                "B_monthly_nmbe_mean": r["summary"]["B_monthly"]["nmbe"]["mean"],
                                "B_hourly_cv_mean": r["summary"]["B_hourly"]["cv_rmse"]["mean"],
                                "B_monthly_cv_mean": r["summary"]["B_monthly"]["cv_rmse"]["mean"]}
    sd_rows = {}
    for sd in (10.0, 20.0, 25.0, 28.0, 30.0, 32.0, 40.0, 200.0):
        r = run_seeds(sd_a=sd, seeds=SEEDS, base=4000)
        sd_rows[str(sd)] = {"A_hourly_pass": r["passes"]["A_hourly"], "A_monthly_pass": r["passes"]["A_monthly"],
                            "A_hourly_cv_mean": r["summary"]["A_hourly"]["cv_rmse"]["mean"],
                            "A_monthly_cv_mean": r["summary"]["A_monthly"]["cv_rmse"]["mean"],
                            "A_monthly_nmbe_p5_p95": [r["summary"]["A_monthly"]["nmbe"]["p5"], r["summary"]["A_monthly"]["nmbe"]["p95"]],
                            "A_monthly_fails_by": r["fails_by"]["A_monthly"], "A_monthly_cv_max": r["cv_max"]["A_monthly"],
                            "rmse_picks_B": r["rmse_picks_B"]}
    return {"seeds": SEEDS, "bias_sweep_sd_b5": bias_rows, "noise_sweep_A": sd_rows}


def e4_offset():
    y, a, b = models(0)
    out = {}
    for k, p in (("A", a), ("B", b)):
        pc = m.offset_correct(y, p)
        ym, pm = cases.aggregate_monthly(y, p)
        _, pcm = cases.aggregate_monthly(y, pc)
        out[k] = {"shift": m.mbe(y, p),
                  "before": all_metrics(y, p), "after": all_metrics(y, pc),
                  "monthly_before": m.ashrae_check(ym, pm, "monthly"), "monthly_after": m.ashrae_check(ym, pcm, "monthly"),
                  "hourly_after_check": m.ashrae_check(y, pc, "hourly")}
    # B는 곱하기 꼴 편향(0.92y)이라 상수 보정 뒤에도 하루 주기 모양의 잔차가 남는다: 0.08*(y - mean y)
    resid_shape = 0.08 * (y - y.mean())
    out["B_residual_shape_sd"] = float(np.std(resid_shape))
    out["B_noise_floor_cv"] = float(100 * 5.0 / y.mean())
    return out


def e5_conventions():
    y, _, b = models(0)
    ym, bm = cases.aggregate_monthly(y, b)
    hourly_norm = float(np.max(y))
    rows = {}
    for name, (yy, pp, norm) in {"hourly": (y, b, hourly_norm), "monthly": (ym, bm, float(np.max(ym)))}.items():
        rows[name] = {"n": len(yy), "ashrae_p1": m.nmbe(yy, pp, p=1), "ashrae_p0": m.nmbe(yy, pp, p=0),
                      "femp": m.nmbe(yy, pp, convention="femp"), "sfa_norm_max": m.nmbe(yy, pp, convention="sfa", norm=norm),
                      "sfa_norm": norm, "cv_p1": m.cv_rmse(yy, pp, p=1), "cv_p0": m.cv_rmse(yy, pp, p=0),
                      "nmbe_inflation_n_over_n_minus_p": len(yy) / (len(yy) - 1)}
    # 오차가 모두 같을 때 p=1이면 |NMBE| > CV(RMSE)가 된다 (n=12, 오차 1, 평균 100)
    ye = np.full(12, 100.0)
    pe = ye - 1.0
    rows["equal_errors_n12"] = {"nmbe_p1": m.nmbe(ye, pe, p=1), "cv_p1": m.cv_rmse(ye, pe, p=1),
                                "nmbe_p0": m.nmbe(ye, pe, p=0), "cv_p0": m.cv_rmse(ye, pe, p=0)}
    return rows


def e6_nmae():
    y, a, b = models(0)
    same = {}
    for kind in ("capacity", "mean", "range"):
        nv = m.norm_value(y, kind)
        same[kind] = {"norm": nv, "A": m.nmae(y, a, nv), "B": m.nmae(y, b, nv)}
    flips_same = 0
    for s in range(1000, 1000 + SEEDS):
        y2, a2, b2 = models(s)
        picks = {kind: m.nmae(y2, a2, m.norm_value(y2, kind)) < m.nmae(y2, b2, m.norm_value(y2, kind))
                 for kind in ("capacity", "mean", "range")}
        flips_same += len(set(picks.values())) > 1
    # 서로 다른 발전소: 공급사 A는 흐린 발전소 P, 공급사 B는 맑은 발전소 Q의 예측을 맡았다(이 장에서 고른 설정)
    cap = 100.0
    wins = {"capacity": 0, "mean": 0, "range": 0}
    flips_cross = 0
    seed0 = None
    for s in range(SEEDS):
        yp = cases.solar_site(0.2, 0.6, capacity=cap, seed=5000 + s)
        yq = cases.solar_site(0.6, 1.0, capacity=cap, seed=6000 + s)
        fa = cases.solar_forecast(yp, 0.35, capacity=cap, seed=7000 + s)
        fb = cases.solar_forecast(yq, 0.25, capacity=cap, seed=8000 + s)
        row = {}
        for kind in wins:
            na = m.nmae(yp, fa, m.norm_value(yp, kind, capacity=cap if kind == "capacity" else None))
            nb = m.nmae(yq, fb, m.norm_value(yq, kind, capacity=cap if kind == "capacity" else None))
            row[kind] = {"A_on_P": na, "B_on_Q": nb}
            wins[kind] += na < nb
        picks = {kind: row[kind]["A_on_P"] < row[kind]["B_on_Q"] for kind in wins}
        flips_cross += len(set(picks.values())) > 1
        if s == 0:
            seed0 = {"row": row, "mae_A_on_P": m.mae(yp, fa), "mae_B_on_Q": m.mae(yq, fb),
                     "mean_P": float(yp.mean()), "mean_Q": float(yq.mean()), "max_P": float(yp.max()), "max_Q": float(yq.max()),
                     "daylight_share": float((yp > 0).mean())}
    return {"same_series_seed0": same, "same_series_flips": int(flips_same), "seeds": SEEDS,
            "cross_site": {"seed0": seed0, "A_wins": {k: int(v) for k, v in wins.items()}, "flips": int(flips_cross),
                           "setup": "P: 맑음 U(0.2,0.6), Q: U(0.6,1.0), 정격 100, A 상대잡음 0.35, B 0.25, 365일"}}


def e7_fixture_torch():
    y, p = cases.fixture_ashrae()
    yi, pi = [int(v) for v in y], [int(v) for v in p]
    e = [a - b for a, b in zip(yi, pi)]
    exact_nmbe = Fraction(100 * sum(e), (len(e) - 1) * Fraction(sum(yi), len(yi)))
    exact_cv_sq = Fraction(100 ** 2) * Fraction(sum(v * v for v in e), len(e) - 1) / Fraction(sum(yi), len(yi)) ** 2
    exact_femp = Fraction(100 * sum(-v for v in e), sum(yi))
    rng = np.random.default_rng(7)
    y2 = cases.base_load(seed=7)
    p2 = cases.case_biased_steady(y2, seed=7) + rng.normal(0, 1, len(y2))
    ty, tp = torch.tensor(y2), torch.tensor(p2)
    d_np = m.mse_decomposition(y2, p2)
    d_t = tm.mse_decomposition(ty, tp)
    return {"fixture": {"y": y.tolist(), "yhat": p.tolist(), "errors": e,
                        "exact_nmbe": str(exact_nmbe), "numpy_nmbe": m.nmbe(y, p),
                        "exact_cv_rmse_squared": str(exact_cv_sq), "numpy_cv_rmse": m.cv_rmse(y, p),
                        "exact_femp_mbe": str(exact_femp), "numpy_femp_mbe": m.nmbe(y, p, convention="femp"),
                        "numpy_femp_cv": m.cv_rmse(y, p, p=0)},
            "numpy_torch_abs_diff": {"nmbe": abs(m.nmbe(y2, p2) - tm.nmbe_ashrae(ty, tp)),
                                     "cv_rmse": abs(m.cv_rmse(y2, p2) - tm.cv_rmse(ty, tp)),
                                     "nmae_mean": abs(m.nmae(y2, p2, y2.mean()) - tm.nmae(ty, tp, float(y2.mean()))),
                                     "mse_vs_F_mse_loss": abs(d_np["mse"] - d_t["mse_builtin"]),
                                     "bias_sq_plus_var_vs_mse": abs(d_np["bias_sq"] + d_np["var"] - d_np["mse"])}}


def main():
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "E0": e0_first_screen(), "E1": e1_failure(), "E2": e2_seeds(), "E3": e3_sweeps(),
        "E4": e4_offset(), "E5": e5_conventions(), "E6": e6_nmae(), "E7": e7_fixture_torch(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
