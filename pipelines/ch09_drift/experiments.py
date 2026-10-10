"""9장 비교 실험. `uv run python pipelines/ch09_drift/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스·kfp 불필요).

E0 손 계산 예: 요금 다섯 개씩, 누적 비율 표와 KS 거리
E1 조용한 하락: 배포 시점 검사를 모두 통과한 모델에 30개 배치(배치당 500행)를 흘린다.
   요금 평균이 배치마다 0.04 표준편차씩 오르고(P(X) 변화), 요금이 이탈로 이어지는 정도도 0.04씩 약해진다(P(Y|X) 변화).
   배치별 실제 정확도, KS(열마다, 본페로니), MMD(열 전체, 순열 200회) 경보.
E2 같은 입력에서 두 검사: 이동 크기 0, 0.1, 0.25, 0.5 x 시드 20개, 그리고 짝 관계만 바뀐 경우, 관계만 바뀐 경우
E3 아무것도 안 바뀐 30개 배치의 헛경보: 보정 없음 / 본페로니 / MMD, 같은 값이 많은 열의 근사 p값
E4 계산량: n = 500, 2,000, 8,000에서 MMD와 KS 시간
E5 NumPy MMD vs PyTorch MMD vs 이중 루프
E6 감시·재학습·되돌리기 루프(정답 지연 3배치): 감시 없음 / 정답 온 데이터로 재학습 / 정답 안 온 행을 0으로 채운 재학습
   그리고 정답 지연 1, 3, 6배치에서 나쁜 모델이 서빙한 배치 수
E7 Jenkinsfile 규칙 검사

결과는 results/ch09.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch09_data as D  # noqa: E402
import pipelines_ch09_drift_np as K  # noqa: E402
import pipelines_ch09_drift_torch as KT  # noqa: E402
import pipelines_ch09_jenkinsfile as J  # noqa: E402
import pipelines_ch09_loop as LP  # noqa: E402
import pipelines_ch09_model_np as M  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch09.json"
N_BATCH, BATCH = 30, 500
STEP_SHIFT, STEP_COEF = 0.04, 0.04
GATE_ACC = 0.78  # 7장 Jenkinsfile Gate의 --min-accuracy


def r4(x):
    return round(float(x), 4)


def drift_batches(n_batches=N_BATCH, seed0=100, step_shift=STEP_SHIFT, step_coef=STEP_COEF):
    return [D.make_table(seed0 + t, BATCH, fee_shift=step_shift * t, fee_coef=1.6 - step_coef * t)
            for t in range(n_batches)]


def baseline():
    train = D.make_table(0, 2000)
    hold = D.make_table(1, 2000)
    model = M.train(train)
    return train, hold, model


def e0():
    ref = np.array([3, 4, 5, 6, 7.0])
    cur = np.array([5, 6, 6, 7, 8.0])
    grid = np.arange(3, 9.0)
    f_ref = [float(np.mean(ref <= g)) for g in grid]
    f_cur = [float(np.mean(cur <= g)) for g in grid]
    lam = math.sqrt(-0.5 * math.log(0.05 / 6 / 2))
    return {
        "ref": ref.tolist(), "cur": cur.tolist(), "grid": grid.tolist(), "F_ref": f_ref, "F_cur": f_cur,
        "gap": [r4(abs(a - b)) for a, b in zip(f_ref, f_cur)], "D": r4(K.ks_stat(ref, cur)),
        "crit_first_term": {"alpha": 0.05, "K": 6, "lambda": r4(lam), "n": 2000, "m": 500,
                            "D_crit": r4(lam * math.sqrt((2000 + 500) / (2000 * 500)))},
    }


def e1(train, hold, model):
    batches = drift_batches()
    ref = D.numeric_matrix(train)
    rows = []
    for t, b in enumerate(batches):
        cur = D.numeric_matrix(b)
        ks = K.ks_check(ref, cur, names=D.NUMERIC)
        mmd = K.mmd_permutation_test(ref, cur, n_perm=200, seed=t)
        rows.append({
            "t": t, "acc": r4(M.accuracy(model, b)), "fee_mean": r4(b["monthly_fee"].mean()),
            "churn_rate": r4(b[D.LABEL].mean()),
            "ks_D_fee": r4(ks["D"]["monthly_fee"]), "ks_min_p": float(f"{ks['min_p']:.3g}"), "ks_worst": ks["worst"],
            "ks_alarm": ks["alarm"], "mmd2": float(f"{mmd['mmd2']:.4g}"), "mmd_p": r4(mmd["p"]), "mmd_alarm": mmd["p"] < 0.05,
        })
    first = lambda key: next((r["t"] for r in rows if r[key]), None)  # noqa: E731
    below = next((r["t"] for r in rows if r["acc"] < GATE_ACC), None)
    nan_rows = int(sum(np.isnan(D.numeric_matrix(b)).sum() for b in batches))
    return {
        "deploy_check": {"holdout_acc": r4(M.accuracy(model, hold)), "gate": GATE_ACC,
                         "passed": M.accuracy(model, hold) >= GATE_ACC, "nan_values_in_batches": nan_rows,
                         "train_fee_mean": r4(train["monthly_fee"].mean())},
        "rows": rows,
        "first_ks_alarm": first("ks_alarm"), "first_mmd_alarm": first("mmd_alarm"), "first_acc_below_gate": below,
        "acc_first5_mean": r4(np.mean([r["acc"] for r in rows[:5]])),
        "acc_last5_mean": r4(np.mean([r["acc"] for r in rows[-5:]])),
        "n_ks_alarms": sum(r["ks_alarm"] for r in rows), "n_mmd_alarms": sum(r["mmd_alarm"] for r in rows),
    }


def e2(model):
    seeds = range(20)
    cases = [("fee_shift", d, dict(fee_shift=d)) for d in (0.0, 0.1, 0.25, 0.5)]
    cases += [("fee_usage_corr", 0.7, dict(fee_usage_corr=0.7)),
              ("fee_coef", 0.4, dict(fee_coef=0.4)), ("shift", 1.0, dict(shift=1.0))]
    out = []
    for kind, val, kw in cases:
        ks_a, mmd_a, acc, acc0 = [], [], [], []
        for s in seeds:
            ref_t = D.make_table(1000 + s, 500)
            cur_t = D.make_table(2000 + s, 500, **kw)
            ref, cur = D.numeric_matrix(ref_t), D.numeric_matrix(cur_t)
            ks_a.append(K.ks_check(ref, cur)["alarm"])
            mmd_a.append(K.mmd_permutation_test(ref, cur, n_perm=200, seed=s)["p"] < 0.05)
            acc.append(M.accuracy(model, cur_t))
            acc0.append(M.accuracy(model, D.make_table(2000 + s, 500)))
        out.append({"kind": kind, "value": val, "ks_rate": r4(np.mean(ks_a)), "mmd_rate": r4(np.mean(mmd_a)),
                    "acc": r4(np.mean(acc)), "acc_same_rows_no_change": r4(np.mean(acc0))})
    return {"seeds": 20, "n_ref": 500, "n_cur": 500, "alpha": 0.05, "n_perm": 200, "cases": out}


def e3():
    seeds, nb = 20, N_BATCH
    unadj, bonf, mmd = [], [], []
    for s in seeds_range(seeds):
        ref = D.numeric_matrix(D.make_table(3000 + s, 500))
        u = b = mm = 0
        for t in range(nb):
            cur = D.numeric_matrix(D.make_table(10_000 + 100 * s + t, 500))
            ks = K.ks_check(ref, cur, correction="none")
            u += ks["alarm"]
            b += ks["min_p"] < 0.05 / 6
            mm += K.mmd_permutation_test(ref, cur, n_perm=200, seed=t)["p"] < 0.05
        unadj.append(int(u))
        bonf.append(int(b))
        mmd.append(int(mm))
    # 기준 표본을 매번 새로 뽑으면(서로 독립인 300쌍) 한 번 검사의 헛경보율
    ind_ks, ind_mmd = [], []
    for s in range(300):
        ref = D.numeric_matrix(D.make_table(60_000 + s, 500))
        cur = D.numeric_matrix(D.make_table(70_000 + s, 500))
        ind_ks.append(K.ks_check(ref, cur)["alarm"])
        ind_mmd.append(K.mmd_permutation_test(ref, cur, n_perm=200, seed=s)["p"] < 0.05)
    # 열 하나의 오경보율: 연속 열(할인율)과 같은 값이 많은 열(상담 전화 수), 근사 p vs 순열 p
    col = {}
    for name in ("discount_rate", "support_calls"):
        j = D.NUMERIC.index(name)
        asym, perm = [], []
        for s in range(200):
            a = D.make_table(20_000 + s, 200)[name]
            bb = D.make_table(30_000 + s, 200)[name]
            asym.append(K.ks_pvalue(K.ks_stat(a, bb), 200, 200) < 0.05)
            if s < 100:
                perm.append(K.ks_perm_pvalue(a, bb, n_perm=200, seed=s) < 0.05)
        col[name] = {"seeds_asym": 200, "asym_rate": r4(np.mean(asym)), "seeds_perm": 100, "perm_rate": r4(np.mean(perm)),
                     "distinct_values_in_200": int(len(np.unique(D.make_table(20_000, 200)[name]))), "col_index": j}
    return {
        "seeds": seeds, "batches": nb, "n": 500,
        "unadjusted_mean_alarms": r4(np.mean(unadj)), "bonferroni_mean_alarms": r4(np.mean(bonf)),
        "mmd_mean_alarms": r4(np.mean(mmd)),
        "unadjusted_expected_if_independent_continuous": r4(nb * (1 - 0.95 ** 6)),
        "bonferroni_expected_upper": r4(nb * 0.05), "single_column": col,
        "per_seed": {"unadjusted": unadj, "bonferroni": bonf, "mmd": mmd},
        "median_per_seed": {"unadjusted": float(np.median(unadj)), "bonferroni": float(np.median(bonf)), "mmd": float(np.median(mmd))},
        "independent_pairs": {"n_pairs": 300, "ks_bonferroni_rate": r4(np.mean(ind_ks)), "mmd_rate": r4(np.mean(ind_mmd))},
    }


def seeds_range(n):
    return range(n)


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def e4():
    rows = []
    for n in (500, 2000, 8000):
        x = D.numeric_matrix(D.make_table(40_000, n))
        y = D.numeric_matrix(D.make_table(40_001, n, fee_shift=0.25))
        z = K.standardize_pooled(x, y)
        xs, ys = z[:n], z[n:]
        sigma = K.median_bandwidth(z, max_rows=2000)
        reps = 3 if n < 8000 else 1
        t_mmd = timeit(lambda: K.mmd2_blockwise(xs, ys, sigma), reps)
        t_ks = timeit(lambda: K.ks_check(x, y), 3)
        rows.append({"n": n, "pairs": 2 * n * n, "mmd_seconds": r4(t_mmd), "ks_seconds": r4(t_ks)})
    x = D.numeric_matrix(D.make_table(40_000, 500))
    y = D.numeric_matrix(D.make_table(40_001, 500))
    t_perm = timeit(lambda: K.mmd_permutation_test(x, y, n_perm=200, seed=0), 3)
    return {"rows": rows, "perm_test_n500_200perm_seconds": r4(t_perm),
            "ratio_8000_over_2000": r4(rows[2]["mmd_seconds"] / rows[1]["mmd_seconds"]),
            "ratio_2000_over_500": r4(rows[1]["mmd_seconds"] / rows[0]["mmd_seconds"])}


def e5():
    x = D.numeric_matrix(D.make_table(50_000, 300))
    y = D.numeric_matrix(D.make_table(50_001, 300, fee_shift=0.5))
    z = K.standardize_pooled(x, y)
    xs, ys = z[:300], z[300:]
    sigma = K.median_bandwidth(z)
    a = K.mmd2_unbiased(xs, ys, sigma)
    b = KT.mmd2_unbiased(xs, ys, sigma)
    c = K.mmd2_loop(xs[:60], ys[:60], sigma)
    d = K.mmd2_unbiased(xs[:60], ys[:60], sigma)
    e = K.mmd_permutation_test(x, y, n_perm=10, seed=0, sigma=sigma)["mmd2"]
    return {"numpy": a, "torch": b, "abs_diff_np_torch": float(abs(a - b)),
            "loop_n60": c, "vec_n60": d, "abs_diff_loop_vec": float(abs(c - d)),
            "perm_matmul_obs": e, "abs_diff_perm_vs_direct": float(abs(e - a)), "sigma": r4(sigma)}


def e6(train, model):
    batches = drift_batches()
    ref = D.numeric_matrix(train)
    out = {}
    for name, kw in [("none", dict(monitor=False)), ("labeled", dict(policy="labeled")), ("naive", dict(policy="naive")),
                     ("naive_pause", dict(policy="naive", pause_after_rollback=True))]:
        log = LP.run_loop(batches, model, ref, label_delay=3, window=2, **kw)
        base = LP.digest(model)
        out[name] = {
            "mean_acc": r4(np.mean(log["acc"])), "last10_mean_acc": r4(np.mean(log["acc"][-10:])),
            "acc": [r4(a) for a in log["acc"]], "alarms": log["alarms"], "runs": log["runs"],
            "decisions": [{k: (r4(v) if isinstance(v, float) else v) for k, v in d.items()} for d in log["decisions"]],
            "n_rollbacks": sum(d["verdict"] == "rollback" for d in log["decisions"]),
            "n_promotes": sum(d["verdict"] == "promote" for d in log["decisions"]),
            "distinct_digests_served": len(set(log["serving"])),
            "batches_served_by_original": sum(s == base for s in log["serving"]),
        }
        if name.startswith("naive"):
            out[name]["batches_served_by_rejected"] = bad_batches(log)
    sweep = []
    for L in (1, 3, 6):
        log = LP.run_loop(batches, model, ref, label_delay=L, window=2, policy="naive")
        rej = bad_batches(log)
        first = log["decisions"][0] if log["decisions"] else None
        sweep.append({"label_delay": L, "n_runs": len(log["runs"]), "n_rollbacks": sum(d["verdict"] == "rollback" for d in log["decisions"]),
                      "batches_served_by_rejected": rej, "mean_acc": r4(np.mean(log["acc"])),
                      "first_bad_model_acc": r4(first["acc_new"]) if first else None,
                      "first_bad_model_old_acc": r4(first["acc_old"]) if first else None})
    out["naive_label_delay_sweep"] = sweep
    return out


def bad_batches(log):
    """되돌려진(rollback) 모델이 서빙한 배치 수."""
    rejected = set()
    for run, dec in zip(log["runs"], log["decisions"]):
        if dec["verdict"] == "rollback":
            rejected.add(run["digest"])
    return sum(s[:19] in rejected for s in log["serving"])


def e7():
    text = (HERE / "Jenkinsfile").read_text()
    bad_tag = text.replace('"$SERVING@$(cat results/previous_digest.txt)"', '"$SERVING:latest"')
    bad_train = text.replace("python ci/submit_retrain.py", "python -c 'import M; M.train(load())' && python ci/submit_retrain.py")
    bad_cron = text.replace("cron('H * * * *')", "cron('0 * * * *')")
    return {"ours": J.lint(text), "rollback_by_tag": J.lint(bad_tag), "train_in_jenkins": J.lint(bad_train),
            "cron_without_H": J.lint(bad_cron), "stages": [n for n, _ in J.stages(text)]}


def main():
    t0 = time.perf_counter()
    train, hold, model = baseline()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                   "platform": f"{platform.system()} {platform.machine()}", "threads": 2}}
    res["E0"] = e0()
    res["E1"] = e1(train, hold, model)
    res["E2"] = e2(model)
    res["E3"] = e3()
    res["E4"] = e4()
    res["E5"] = e5()
    res["E6"] = e6(train, model)
    res["E7"] = e7()
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump({k: res[k] for k in ("E0",)}, sys.stdout, ensure_ascii=False)
    print("\nsaved", OUT, res["total_seconds"], "s")


if __name__ == "__main__":
    main()
