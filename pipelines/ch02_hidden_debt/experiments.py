"""2장 비교 실험. `uv run python pipelines/ch02_hidden_debt/experiments.py` 로 실행한다.

E1 조용한 실패: v1로 학습한 모델에 v2 표(금액 천 원 단위 + 요금제 코드 순서 변경)를 넣는다. 시드 5개.
E2 같은 입력, 세 가지 설계: 검사 없음 / 값 검사(스키마 + 분포 비교) / 뜻이 담긴 계약(라벨·단위를 열에 적음)
   사례 다섯 개: 단위 변경, 코드 순서 변경, 비율이 고른 범주의 코드 순서 변경, 업스트림의 '개선', v2 전체
E3 금액 배율 곡선: 같은 고객 금액에 배율을 곱했을 때 정확도와 각 검사의 경보 여부
E4 검사의 대가: 아무것도 안 바뀐 v1 표에서 분포 경보가 울리는 비율(표 크기별), 검사 시간
E5 CACE: 피처 하나를 빼면 나머지 가중치가 얼마나 바뀌나, 단위를 학습·서빙 모두에서 바꾸면?
E6 시스템 안에서 모델 코드의 비중: 이 폴더의 파이썬 줄 수
E7 ML Test Score: 이 장 저장소의 점수(Breck 등 2017의 계산법)
E8 NumPy와 PyTorch 대조

결과는 results/ch02.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch02_checks as chk  # noqa: E402
import pipelines_ch02_data as dat  # noqa: E402
import pipelines_ch02_model_np as mnp  # noqa: E402
import pipelines_ch02_model_torch as mtt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results" / "ch02.json"
SEEDS = range(5)
N_TRAIN, N_SERVE = 4000, 2000
V2_CODEBOOK = dat.UPSTREAM[2]["plan_codebook"]


def _world(seed, plan_probs=(0.5, 0.3, 0.2)):
    return (
        dat.make_customers(seed, N_TRAIN, plan_probs),
        dat.make_customers(1000 + seed, N_SERVE, plan_probs),
    )


def _predict_safely(model, batch, y):
    """예외가 나는지, NaN이 나오는지까지 센다."""
    try:
        prep, w, b = model
        p = mnp.predict_proba(mnp.transform(batch, prep), w, b)
        return {"accuracy": float(np.mean((p >= 0.5) == y)), "exception": None, "nan": int(np.isnan(p).sum()),
                "mean_pred": float(p.mean())}
    except Exception as e:  # noqa: BLE001
        return {"accuracy": None, "exception": type(e).__name__, "nan": None, "mean_pred": None}


def e1_silent_failure():
    rows = []
    for s in SEEDS:
        tr, sv = _world(s)
        model = mnp.train(dat.emit_batch(tr, 1), tr["churned"])
        y = sv["churned"]
        majority = float(max(y.mean(), 1 - y.mean()))
        rows.append({
            "seed": s,
            "churn_rate_serve": float(y.mean()),
            "majority_baseline": majority,
            "v1": _predict_safely(model, dat.emit_batch(sv, 1), y),
            "unit_only": _predict_safely(model, dat.emit_batch(sv, 1, fee_divisor=1000.0), y),
            "codes_only": _predict_safely(model, dat.emit_batch(sv, 1, plan_codebook=V2_CODEBOOK), y),
            "v2": _predict_safely(model, dat.emit_batch(sv, 2), y),
        })
    summary = {}
    for key in ("v1", "unit_only", "codes_only", "v2"):
        accs = [r[key]["accuracy"] for r in rows]
        summary[key] = {
            "acc_mean": float(np.mean(accs)), "acc_min": float(np.min(accs)), "acc_max": float(np.max(accs)),
            "exceptions": sum(r[key]["exception"] is not None for r in rows),
            "nan_predictions": sum(r[key]["nan"] for r in rows),
            "mean_pred": float(np.mean([r[key]["mean_pred"] for r in rows])),
        }
    summary["majority_baseline_mean"] = float(np.mean([r["majority_baseline"] for r in rows]))
    summary["churn_rate_serve_mean"] = float(np.mean([r["churn_rate_serve"] for r in rows]))
    # 손으로 따라 할 숫자: 학습 평균·표준편차와 3만 원 고객 한 명의 z
    tr, _ = _world(0)
    prep = mnp.fit_preprocessor(dat.emit_batch(tr, 1))
    m, sd = prep["mean"]["monthly_fee"], prep["std"]["monthly_fee"]
    summary["hand_example_seed0"] = {
        "fee_mean": m, "fee_std": sd,
        "z_30000_krw": (30_000 - m) / sd, "z_30_thousand": (30 - m) / sd,
    }
    return {"per_seed": rows, "summary": summary}


CASES = {
    # 이름: (세상의 요금제 비율, 학습 쪽 보내는 방식, 서빙 쪽 보내는 방식)
    "unit": ((0.5, 0.3, 0.2), {}, {"fee_divisor": 1000.0}),
    "codes": ((0.5, 0.3, 0.2), {}, {"plan_codebook": V2_CODEBOOK}),
    "codes_uniform": ((1 / 3, 1 / 3, 1 / 3), {}, {"plan_codebook": V2_CODEBOOK}),
    "upstream_fix": ((0.5, 0.3, 0.2), {"usage_scale": 0.5}, {"usage_scale": 1.0}),
    "v2_all": ((0.5, 0.3, 0.2), {}, {"fee_divisor": 1000.0, "plan_codebook": V2_CODEBOOK}),
}


def _labeled_kwargs(cfg):
    return {"fee_divisor": cfg.get("fee_divisor", 1.0), "usage_scale": cfg.get("usage_scale", 1.0)}


def e2_designs():
    out = {}
    for name, (probs, tr_cfg, sv_cfg) in CASES.items():
        per_seed = []
        for s in SEEDS:
            tr, sv = _world(s, probs)
            y = sv["churned"]
            train_batch = dat.emit_batch(tr, 1, **tr_cfg)
            model = mnp.train(train_batch, tr["churned"])
            ref = chk.profile(train_batch)
            clean = _predict_safely(model, dat.emit_batch(sv, 1, **tr_cfg), y)["accuracy"]

            serve = dat.emit_batch(sv, 1, **sv_cfg)
            a = _predict_safely(model, serve, y)["accuracy"]
            errors = chk.validate(serve)
            alerts = chk.drift_alerts(ref, serve)

            # 설계 C: 라벨과 단위를 담아 보내고, 받는 쪽이 자기 코드표로 바꾼다
            lab_train = mnp.labeled_to_codes(dat.emit_labeled_batch(tr, **_labeled_kwargs(tr_cfg)),
                                             dat.PLAN_LABELS, dat.REGION_LABELS)
            model_c = mnp.train(lab_train, tr["churned"])
            try:
                lab_serve = mnp.labeled_to_codes(dat.emit_labeled_batch(sv, **_labeled_kwargs(sv_cfg)),
                                                 dat.PLAN_LABELS, dat.REGION_LABELS)
                c_blocked, c_acc = False, _predict_safely(model_c, lab_serve, y)["accuracy"]
            except mnp.ContractError as e:
                c_blocked, c_acc, _msg = True, None, str(e)

            majority = float(max(y.mean(), 1 - y.mean()))
            per_seed.append({
                "seed": s, "clean_acc": clean, "none_acc": a,
                "schema_errors": errors, "drift_alerts": alerts,
                "contract_blocked": c_blocked, "contract_acc": c_acc,
                "perf_gate_fail": bool(a < majority + 0.05),
            })
        out[name] = {
            "clean_acc_mean": float(np.mean([r["clean_acc"] for r in per_seed])),
            "none_acc_mean": float(np.mean([r["none_acc"] for r in per_seed])),
            "schema_blocked_seeds": sum(bool(r["schema_errors"]) for r in per_seed),
            "drift_alert_seeds": sum(bool(r["drift_alerts"]) for r in per_seed),
            "value_checks_blocked_seeds": sum(bool(r["schema_errors"] or r["drift_alerts"]) for r in per_seed),
            "contract_blocked_seeds": sum(r["contract_blocked"] for r in per_seed),
            "contract_acc_mean": (float(np.mean([r["contract_acc"] for r in per_seed if r["contract_acc"] is not None]))
                                  if any(r["contract_acc"] is not None for r in per_seed) else None),
            "perf_gate_fail_seeds": sum(r["perf_gate_fail"] for r in per_seed),
            "example_schema_errors_seed0": per_seed[0]["schema_errors"],
            "example_drift_alerts_seed0": per_seed[0]["drift_alerts"],
            "per_seed": per_seed,
        }
    return out


def e3_fee_multiplier():
    mults = [1.0, 1.1, 0.9, 0.5, 0.1, 0.01, 0.001]
    out = {}
    worlds = [_world(s) for s in SEEDS]
    models = [(mnp.train(dat.emit_batch(tr, 1), tr["churned"]), chk.profile(dat.emit_batch(tr, 1))) for tr, _ in worlds]
    for m in mults:
        accs, schema_hits, drift_hits, oor = [], 0, 0, []
        for (tr, sv), (model, ref) in zip(worlds, models):
            serve = dat.emit_batch(sv, 1, fee_divisor=1.0 / m)
            accs.append(mnp.accuracy(serve, sv["churned"], model))
            errs = chk.validate(serve)
            schema_hits += bool(errs)
            drift_hits += "monthly_fee" in chk.drift_alerts(ref, serve)
            f = serve["monthly_fee"]
            oor.append(float(np.mean((f < 10_000) | (f > 300_000))))
        out[str(m)] = {"acc_mean": float(np.mean(accs)), "schema_blocked_seeds": schema_hits,
                       "drift_alert_seeds": drift_hits, "out_of_range_frac_mean": float(np.mean(oor))}
    return out


def e3b_tighter_tolerance():
    """연습 1의 정답: 이상치 허용치를 1%에서 0.1%로 낮추면?"""
    saved = chk.MAX_OUT_OF_RANGE
    try:
        chk.MAX_OUT_OF_RANGE = 0.001
        half = sum(bool(chk.validate(dat.emit_batch(dat.make_customers(1000 + s, N_SERVE), 1, fee_divisor=2.0)))
                   for s in SEEDS)
        false_reject_100 = sum(bool(chk.validate(dat.emit_batch(dat.make_customers(5000 + s, 100), 1)))
                               for s in range(40))
    finally:
        chk.MAX_OUT_OF_RANGE = saved
    return {"max_out_of_range": 0.001, "fee_x0.5_blocked_seeds": half, "false_reject_unchanged_100rows_of_40": false_reject_100}


def e4_cost():
    tr, _ = _world(0)
    ref = chk.profile(dat.emit_batch(tr, 1))
    false_alarm = {}
    for n in (100, 300, 1000, 3000):
        hits, schema_hits, cols = 0, 0, {}
        for s in range(40):
            b = dat.emit_batch(dat.make_customers(5000 + s, n), 1)
            a = chk.drift_alerts(ref, b)
            hits += bool(a)
            schema_hits += bool(chk.validate(b))
            for c in a:
                cols[c] = cols.get(c, 0) + 1
        false_alarm[str(n)] = {"batches": 40, "drift_alarm_batches": hits, "schema_reject_batches": schema_hits,
                               "alarms_by_column": cols}
    serve = dat.emit_batch(dat.make_customers(7, N_SERVE), 1)
    times = []
    for _ in range(21):
        t0 = time.perf_counter()
        chk.validate(serve)
        chk.drift_alerts(ref, serve)
        times.append((time.perf_counter() - t0) * 1000)
    t_train = []
    for _ in range(5):
        t0 = time.perf_counter()
        mnp.train(dat.emit_batch(tr, 1), tr["churned"])
        t_train.append((time.perf_counter() - t0) * 1000)
    return {"false_alarm_on_unchanged_v1": false_alarm,
            "check_ms_median_2000_rows": float(np.median(times)),
            "train_ms_median_4000_rows": float(np.median(t_train))}


def e5_cace():
    out = []
    names = dat.NUMERIC
    for s in SEEDS:
        tr, sv = _world(s)
        full = dat.emit_batch(tr, 1)
        prep, w, b = mnp.train(full, tr["churned"])
        # support_calls 열만 빼고 다시 학습한다
        reduced_prep = mnp.fit_preprocessor(full)
        X = mnp.transform(full, reduced_prep)
        keep = [i for i in range(X.shape[1]) if i != names.index("support_calls")]
        w2, _ = mnp.fit_logistic(X[:, keep], tr["churned"])
        w_full = {n: float(w[i]) for i, n in enumerate(names)}
        w_drop = {n: float(w2[keep.index(i)]) for i, n in enumerate(names) if n != "support_calls"}
        # 단위를 학습·서빙 모두 천 원으로: 표준화 덕분에 결과가 같아야 한다
        both_k = mnp.train(dat.emit_batch(tr, 1, fee_divisor=1000.0), tr["churned"])
        acc_krw = mnp.accuracy(dat.emit_batch(sv, 1), sv["churned"], (prep, w, b))
        acc_k = mnp.accuracy(dat.emit_batch(sv, 1, fee_divisor=1000.0), sv["churned"], both_k)
        out.append({"seed": s, "w_full": w_full, "w_drop_calls": w_drop,
                    "corr_tenure_calls": float(np.corrcoef(full["tenure_months"], full["support_calls"])[0, 1]),
                    "acc_consistent_krw": acc_krw, "acc_consistent_thousand": acc_k})
    rel = {n: float(np.mean([(r["w_drop_calls"][n] - r["w_full"][n]) / abs(r["w_full"][n]) for r in out]))
           for n in names if n != "support_calls"}
    mean_full = {n: float(np.mean([r["w_full"][n] for r in out])) for n in names}
    mean_drop = {n: float(np.mean([r["w_drop_calls"][n] for r in out])) for n in names if n != "support_calls"}
    return {"per_seed": out, "mean_relative_weight_change_when_dropping_support_calls": rel,
            "mean_weights_full": mean_full, "mean_weights_drop_calls": mean_drop,
            "mean_corr_tenure_calls": float(np.mean([r["corr_tenure_calls"] for r in out])),
            "max_abs_acc_diff_consistent_unit": float(max(abs(r["acc_consistent_krw"] - r["acc_consistent_thousand"]) for r in out))}


def _count_lines(path):
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))


def e6_line_counts():
    src = (HERE / "pipelines_ch02_model_np.py").read_text().splitlines()
    a = next(i for i, l in enumerate(src) if "모델 코드 시작" in l)
    z = next(i for i, l in enumerate(src) if "모델 코드 끝" in l)
    model_lines = sum(1 for l in src[a + 1:z] if l.strip() and not l.strip().startswith("#"))
    files = {p.name: _count_lines(p) for p in sorted(HERE.glob("*.py"))}
    total = sum(files.values())
    product = sum(v for k, v in files.items() if k != "experiments.py" and not k.startswith("test_"))
    return {"model_code_lines": model_lines, "files": files, "total_lines": total,
            "model_share": model_lines / total,
            "lines_without_experiments_and_tests": product,
            "model_share_without_experiments_and_tests": model_lines / product,
            "note": "빈 줄과 # 주석 줄은 세지 않음. 독스트링은 셈.",
            "fit_logistic_source_lines": len(inspect.getsource(mnp.fit_logistic).splitlines())}


def e7_score():
    now = chk.ml_test_score(chk.THIS_CHAPTER)
    after_ci = chk.ml_test_score([(s, n, "automated") for s, n, _ in chk.THIS_CHAPTER])
    return {"items": [list(t) for t in chk.THIS_CHAPTER], "now": now, "if_all_automated": after_ci}


def e8_np_vs_torch():
    tr, _ = _world(0)
    batch = dat.emit_batch(tr, 1)
    prep = mnp.fit_preprocessor(batch)
    X, y = mnp.transform(batch, prep), tr["churned"]
    w1, b1 = mnp.fit_logistic(X, y)
    w2, b2 = mtt.fit_logistic(X, y)
    return {"max_abs_weight_diff": float(np.max(np.abs(w1 - w2))), "bias_diff": abs(b1 - b2),
            "max_abs_prob_diff": float(np.max(np.abs(mnp.predict_proba(X, w1, b1) - mtt.predict_proba(X, w2, b2))))}


def slices():
    out = {}
    for s in SEEDS:
        tr, sv = _world(s)
        model = mnp.train(dat.emit_batch(tr, 1), tr["churned"])
        b = dat.emit_batch(sv, 1)
        for code, lab in enumerate(dat.PLAN_LABELS):
            m = b["plan"] == code
            sub = {k: v[m] for k, v in b.items()}
            out.setdefault(lab, []).append(mnp.accuracy(sub, sv["churned"][m], model))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    results = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "threads": 2, "n_train": N_TRAIN, "n_serve": N_SERVE,
                "seeds": list(SEEDS)},
        "E1": e1_silent_failure(),
        "E2": e2_designs(),
        "E3": e3_fee_multiplier(),
        "E3b": e3b_tighter_tolerance(),
        "E4": e4_cost(),
        "E5": e5_cace(),
        "E6": e6_line_counts(),
        "E7": e7_score(),
        "E8": e8_np_vs_torch(),
        "slice_acc_v1_by_plan": slices(),
    }
    results["env"]["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    brief = {k: results[k] for k in ("E3", "E3b", "E4", "E6", "E7", "E8")}
    brief["E1"] = results["E1"]["summary"]
    brief["E2"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"} for k, v in results["E2"].items()}
    brief["E5"] = {k: v for k, v in results["E5"].items() if k != "per_seed"}
    brief["slices"] = results["slice_acc_v1_by_plan"]
    print(json.dumps(brief, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
