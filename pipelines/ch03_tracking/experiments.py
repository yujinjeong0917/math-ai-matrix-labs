"""3장 비교 실험. `uv run python pipelines/ch03_tracking/experiments.py` 로 실행한다.

E1 손 기록의 실패: 하이퍼파라미터 12조합을 돌리며 (설정, 점수)만 results.csv에 적는다.
   6번째 실행 뒤 동료가 데이터를 새로 뽑았고(data_seed 변경) 아무도 적지 않았다. 바깥 시드 20개.
E2 log_run 기록: 같은 12번을 (코드, 데이터 해시, 설정, 시드, 환경, 지표, 결과 파일)로 기록하고 기록만으로 다시 실행한다.
E3 데이터 해시의 성질: 같은 표, 셀 하나, 열 순서, dtype, CSV 왕복, 파일 이름으로 신원 판정하기
E4 해시 계산 시간: 행 1e3, 1e5, 1e6
E5 저장 공간: 기록만 / 가중치 / 검증 예측 / 데이터 사본까지
E6 기록이 못 막는 것: 12개 중 검증 점수 1등을 고르면 그 점수는 새 데이터에서 얼마나 낮아지나(선택 편향)
E7 NumPy와 PyTorch 대조: 같은 기록으로 두 구현을 돌린다

결과는 results/ch03.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch03_data as dat  # noqa: E402
import pipelines_ch03_hashing as hsh  # noqa: E402
import pipelines_ch03_model_np as mnp  # noqa: E402
import pipelines_ch03_model_torch as mtt  # noqa: E402
import pipelines_ch03_tracking as trk  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch03.json"
GRID = [{"lr": lr, "steps": st, "l2": l2} for lr in (0.05, 0.5) for st in (50, 400) for l2 in (0.0, 0.01, 0.1)]
N_ROWS, N_TRAIN, SPLIT_SEED = 3000, 2000, 7
OUTER = range(20)
SWITCH_AT = 6  # 이 번호의 실행부터 데이터가 바뀌어 있다


def data_seeds(outer):
    return 100 + outer, 200 + outer  # 처음 데이터, 동료가 새로 뽑은 데이터


def make_data(params):
    table = dat.make_table(params["data_seed"], N_ROWS)
    train, valid = dat.split(table, N_TRAIN, params["split_seed"])
    return train, valid, hsh.dataset_hash(table)


def run(params, fit=mnp.fit_logistic):
    train, valid, _ = make_data(params)
    return mnp.train_and_eval(train, valid, params["lr"], params["steps"], params["l2"], fit=fit)


def train_fn(train, valid, params):
    return mnp.train_and_eval(train, valid, params["lr"], params["steps"], params["l2"])[1]


def sweep_params(outer):
    a, b = data_seeds(outer)
    return [{**g, "data_seed": a if i < SWITCH_AT else b, "split_seed": SPLIT_SEED} for i, g in enumerate(GRID)]


def e1_hand_log():
    per_outer, example_csv = [], None
    for o in OUTER:
        a, b = data_seeds(o)
        ps = sweep_params(o)
        accs = [run(p)[1]["val_acc"] for p in ps]
        csv_rows = [{"lr": p["lr"], "steps": p["steps"], "l2": p["l2"], "val_acc": acc} for p, acc in zip(ps, accs)]
        best = int(np.argmax(accs))  # 동점이면 먼저 적힌 행
        # 손 기록으로 '최고 행'을 다시 돌린다: 지금 디스크에 있는 데이터(b)로
        rerun = run({**GRID[best], "data_seed": b, "split_seed": SPLIT_SEED})[1]["val_acc"]
        # 모든 조합을 같은 데이터로 돌렸다면 누가 1등이었나
        acc_a = [run({**g, "data_seed": a, "split_seed": SPLIT_SEED})[1]["val_acc"] for g in GRID]
        acc_b = [run({**g, "data_seed": b, "split_seed": SPLIT_SEED})[1]["val_acc"] for g in GRID]
        same_cfg_gap = float(np.mean(np.abs(np.array(acc_a) - np.array(acc_b))))
        per_outer.append({
            "outer": o, "csv_val_acc": accs, "acc_all_on_a": acc_a, "acc_all_on_b": acc_b, "best_row": best, "best_from_first_half": best < SWITCH_AT,
            "recorded": accs[best], "rerun": rerun, "gap": accs[best] - rerun,
            "reproduced": abs(accs[best] - rerun) < 1e-12,
            "winner_on_a": int(np.argmax(acc_a)), "winner_on_b": int(np.argmax(acc_b)),
            "csv_winner_is_winner_on_a_or_b": best in (int(np.argmax(acc_a)), int(np.argmax(acc_b))),
            "config_spread_on_a": float(max(acc_a) - min(acc_a)),
            "same_config_data_gap_mean": same_cfg_gap,
        })
        if o == 0:
            example_csv = csv_rows
    first = [r for r in per_outer if r["best_from_first_half"]]
    return {
        "grid": GRID, "switch_at": SWITCH_AT, "n_rows": N_ROWS, "n_train": N_TRAIN, "example_csv_outer0": example_csv,
        "per_outer": per_outer,
        "n_outer": len(per_outer),
        "n_best_from_first_half": len(first),
        "n_not_reproduced": sum(not r["reproduced"] for r in per_outer),
        "mean_abs_gap_when_not_reproduced": float(np.mean([abs(r["gap"]) for r in per_outer if not r["reproduced"]])) if any(not r["reproduced"] for r in per_outer) else 0.0,
        "max_abs_gap": float(max(abs(r["gap"]) for r in per_outer)),
        "n_rerun_lower": sum(r["gap"] > 1e-12 for r in per_outer),
        "n_csv_winner_not_true_winner": sum(not r["csv_winner_is_winner_on_a_or_b"] for r in per_outer),
        "mean_config_spread": float(np.mean([r["config_spread_on_a"] for r in per_outer])),
        "mean_same_config_data_gap": float(np.mean([r["same_config_data_gap_mean"] for r in per_outer])),
    }


def e2_tracked():
    code, env = trk.code_identity(), trk.env_snapshot()
    out = []
    for o in OUTER:
        with tempfile.TemporaryDirectory() as d:
            for p in sweep_params(o):
                train, valid, h = make_data(p)
                model, metrics = mnp.train_and_eval(train, valid, p["lr"], p["steps"], p["l2"])
                trk.log_run(d, p, metrics, artifacts={"model": {"w": model["w"], "b": np.array(model["b"])}},
                            data={"train_valid": h}, code=code, env=env)
            runs = trk.load_runs(d)
            groups = trk.comparable_groups(runs)
            best = max(runs, key=lambda r: r["metrics"]["val_acc"])
            exact = sum(trk.rerun_from_record(r, make_data, train_fn)["val_acc"] == r["metrics"]["val_acc"] for r in runs)
            # 기록대로가 아니라 '지금 디스크의 데이터(b)'로 다시 돌리려 하면: 해시가 다른 기록은 학습 전에 멈춘다
            _, b = data_seeds(o)
            blocked = 0
            for r in runs:
                try:
                    trk.rerun_from_record(r, lambda p: make_data({**p, "data_seed": b}), train_fn)
                except trk.RecordMismatch:
                    blocked += 1
            best_group = [x for x in runs if x["data"]["train_valid"] == best["data"]["train_valid"]]
            out.append({
                "outer": o, "n_groups": len(groups), "group_sizes": sorted(len(v) for v in groups.values()),
                "exact_reruns": exact, "blocked_on_wrong_data": blocked,
                "best_run_id": best["run_id"], "best_data_seed": best["params"]["data_seed"],
                "best_group_size": len(best_group),
            })
            if o == 0:
                example = runs[0]
                log_bytes = (Path(d) / "runs.jsonl").stat().st_size
    return {
        "per_outer": out,
        "total_exact_reruns": sum(r["exact_reruns"] for r in out), "total_runs": 12 * len(out),
        "n_outer_with_two_groups": sum(r["n_groups"] == 2 for r in out),
        "total_blocked_on_wrong_data": sum(r["blocked_on_wrong_data"] for r in out),
        "example_record_outer0": example, "runs_jsonl_bytes_outer0": log_bytes,
        "code": code, "env": env,
    }


def e3_hash_props():
    a, b = data_seeds(0)
    t = dat.make_table(a, N_ROWS)
    t_b = dat.make_table(b, N_ROWS)
    h = hsh.dataset_hash(t)
    one = {k: v.copy() for k, v in t.items()}
    one["usage_hours"][123] = np.nextafter(one["usage_hours"][123], np.inf)  # 표현 가능한 가장 작은 변화
    reordered = {k: t[k] for k in reversed(list(t))}
    f32 = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in t.items()}
    csv_t, csv_bytes = hsh.csv_roundtrip(t, digits=6)
    max_csv_diff = max(float(np.max(np.abs(csv_t[k] - t[k]))) for k in t)
    n_changed = int(sum(np.sum(csv_t[k] != t[k]) for k in t))

    def acc_of(table):
        tr, va = dat.split(table, N_TRAIN, SPLIT_SEED)
        return mnp.train_and_eval(tr, va, 0.5, 400, 0.01)[1]["val_acc"]

    f32_back = {k: v.astype(np.float64) if v.dtype == np.float32 else v for k, v in f32.items()}
    return {
        "same_table_same_hash": hsh.dataset_hash(dat.make_table(a, N_ROWS)) == h,
        "one_ulp_change_changes_hash": hsh.dataset_hash(one) != h,
        "column_order_same_hash": hsh.dataset_hash(reordered) == h,
        "float32_changes_hash": hsh.dataset_hash(f32) != h,
        "csv_roundtrip_changes_hash": hsh.dataset_hash(csv_t) != h,
        "csv_cells_changed": n_changed, "csv_total_cells": N_ROWS * len(t),
        "csv_max_abs_diff": max_csv_diff, "csv_bytes": csv_bytes,
        "val_acc_original": acc_of(t), "val_acc_csv_roundtrip": acc_of(csv_t), "val_acc_float32": acc_of(f32_back),
        "path_identity_same_for_both_data": hsh.path_identity("data/customers.csv") == hsh.path_identity("data/customers.csv"),
        "hash_differs_for_both_data": hsh.dataset_hash(t_b) != h,
        "hash_prefix_a": h[:12], "hash_prefix_b": hsh.dataset_hash(t_b)[:12],
    }


def _median_time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def e4_hash_time():
    rows = []
    for n in (1_000, 100_000, 1_000_000):
        t = dat.make_table(5, n)
        nbytes = int(sum(v.nbytes for v in t.values()))
        rows.append({"rows": n, "bytes": nbytes, "seconds": _median_time(lambda: hsh.dataset_hash(t), 5)})
    a, _ = data_seeds(0)
    tr, va = dat.split(dat.make_table(a, N_ROWS), N_TRAIN, SPLIT_SEED)
    train_s = _median_time(lambda: mnp.train_and_eval(tr, va, 0.5, 400, 0.01), 5)
    return {"rows": rows, "train_one_run_seconds_2000rows_400steps": train_s}


def e5_storage():
    a, _ = data_seeds(0)
    p = {**GRID[0], "data_seed": a, "split_seed": SPLIT_SEED}
    table = dat.make_table(a, N_ROWS)
    tr, va = dat.split(table, N_TRAIN, SPLIT_SEED)
    model, metrics = mnp.train_and_eval(tr, va, p["lr"], p["steps"], p["l2"])
    proba = mnp.predict_proba(mnp.transform(va, model["prep"]), model["w"], model["b"])
    variants = {
        "record_only": {},
        "plus_weights": {"model": {"w": model["w"], "b": np.array(model["b"])}},
        "plus_val_predictions": {"model": {"w": model["w"], "b": np.array(model["b"])}, "val_pred": {"p": proba}},
        "plus_data_copy": {"model": {"w": model["w"], "b": np.array(model["b"])}, "val_pred": {"p": proba}, "data": table},
    }
    out = {}
    for name, arts in variants.items():
        with tempfile.TemporaryDirectory() as d:
            trk.log_run(d, p, metrics, artifacts=arts, data={"train_valid": hsh.dataset_hash(table)},
                        code=trk.code_identity(), env=trk.env_snapshot())
            out[name] = int(sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()))
    return {"bytes_per_run": out, "bytes_12_runs": {k: 12 * v for k, v in out.items()}}


def e6_selection_bias():
    rows = []
    for o in OUTER:
        a, _ = data_seeds(o)
        table = dat.make_table(a, N_ROWS)
        tr, va = dat.split(table, N_TRAIN, SPLIT_SEED)
        test = dat.make_table(900 + o, 2000)
        vals, tests = [], []
        for g in GRID:
            model, m = mnp.train_and_eval(tr, va, g["lr"], g["steps"], g["l2"])
            vals.append(m["val_acc"])
            pt = mnp.predict_proba(mnp.transform(test, model["prep"]), model["w"], model["b"])
            tests.append(float(np.mean((pt >= 0.5) == test[dat.LABEL])))
        best = int(np.argmax(vals))
        rows.append({"outer": o, "best": best, "val_best": vals[best], "test_of_best": tests[best],
                     "best_test_any": max(tests), "val_mean": float(np.mean(vals)), "test_mean": float(np.mean(tests))})
    opt = [r["val_best"] - r["test_of_best"] for r in rows]
    return {"per_outer": rows, "mean_val_best": float(np.mean([r["val_best"] for r in rows])),
            "mean_test_of_best": float(np.mean([r["test_of_best"] for r in rows])),
            "mean_optimism": float(np.mean(opt)), "n_optimistic": int(sum(x > 0 for x in opt)),
            "mean_gap_all_configs": float(np.mean([r["val_mean"] - r["test_mean"] for r in rows])),
            "selection_part": float(np.mean(opt) - np.mean([r["val_mean"] - r["test_mean"] for r in rows])),
            "n_val": N_ROWS - N_TRAIN, "n_test": 2000}


def e7_torch():
    a, _ = data_seeds(0)
    p = {"lr": 0.5, "steps": 400, "l2": 0.01, "data_seed": a, "split_seed": SPLIT_SEED}
    m_np, met_np = run(p)
    m_t, met_t = run(p, fit=mtt.fit_logistic)
    return {"params": p, "max_abs_w_diff": float(np.max(np.abs(m_np["w"] - m_t["w"]))),
            "abs_b_diff": abs(m_np["b"] - m_t["b"]), "val_acc_numpy": met_np["val_acc"], "val_acc_torch": met_t["val_acc"]}


def main():
    t0 = time.perf_counter()
    res = {"E1": e1_hand_log(), "E2": e2_tracked(), "E3": e3_hash_props(), "E4": e4_hash_time(),
           "E5": e5_storage(), "E6": e6_selection_bias(), "E7": e7_torch()}
    res["env"] = {**trk.env_snapshot(), "threads": torch.get_num_threads(), "total_seconds": time.perf_counter() - t0}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k in ("E1", "E3", "E4", "E5", "E6", "E7"):
        v = {kk: vv for kk, vv in res[k].items() if kk not in ("per_outer", "example_csv_outer0", "grid")}
        print(k, json.dumps(v, ensure_ascii=False)[:900])
    e2 = {kk: vv for kk, vv in res["E2"].items() if kk not in ("per_outer", "example_record_outer0")}
    print("E2", json.dumps(e2, ensure_ascii=False)[:900])
    print("seconds", res["env"]["total_seconds"])


if __name__ == "__main__":
    main()
