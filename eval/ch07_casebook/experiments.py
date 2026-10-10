"""모델 평가 7장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch07_casebook/experiments.py` 로 실행한다.

E0 첫 화면: 배달 5건. MAE·RMSE가 엇갈리는 두 모델을 목적 두 가지(10분 넘으면 쿠폰, 분당 할인)로 고른다
E1 시드 0의 사례 7개(대조군 + 판독 사례 6개): 지표 표, 목적 손실, 판독 깃발
E2 시드 100개: 사례 x 지표 6개의 '목적과 다른 모델을 고른 비율' 표, 두 번째 목적에서의 같은 표
E3 시드 100개: 판독 깃발이 켜진 비율(사례 x 깃발)
E4 짝지은 부트스트랩 1000번(시드 0): 지표마다 A - B 차이의 95% 구간, 0 포함 여부. 시드 100개에서 '구별 불가' 비율
E5 같은 평가 데이터에서 순위가 늘 같은 지표 짝(R^2와 RMSE, MASE와 MAE)이 정말 한 번도 안 갈리는지
E6 사례 5 보정 뒤 다시 재기, 사례 6 학습 R^2와 평가 R^2
E7 손 계산 fixture와 NumPy-PyTorch 대조
결과는 results/ch07.json, results/report.md 에 저장한다. 시간 측정값은 넣지 않아서 같은 시드면 바이트 단위로 같다.
"""

import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch07_cases as cases  # noqa: E402
import eval_ch07_casebook as cb  # noqa: E402
import eval_ch07_metrics as m  # noqa: E402
import eval_ch07_torch as tm  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
SEEDS = 100
N_BOOT = 1000


def e0_first_screen():
    a, b = cases.first_screen()
    out = {"A_errors": a.tolist(), "B_errors": b.tolist()}
    for k, e in (("A", a), ("B", b)):
        out[k] = {"mae": float(np.mean(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
                  "coupon_won": int(5000 * np.sum(e > 10)),  # 10분 넘게 틀리면 5,000원 쿠폰
                  "per_minute_won": int(100 * np.sum(e))}     # 틀린 1분마다 100원 할인
    return out


def e1_seed0():
    out = {}
    for name in cases.CASE_NAMES:
        c = cases.MAKERS[name](0)
        t = cb.metric_table(c)
        boot = cb.paired_bootstrap(c, n_boot=N_BOOT, seed=0)
        out[name] = {"objective": c["objective"], "alt_objective": c.get("alt_objective"), "intended_flag": c["intended_flag"],
                     "n_test": len(c["y_test"]), "table": t, "flags": cb.read_table(t, boot), "bootstrap": boot}
    return out


def e2_e3_e4_seeds():
    flips, flips_alt, flag_rate, indistinct_metric = {}, {}, {}, {}
    for name in cases.CASE_NAMES:
        fn = cases.MAKERS[name]
        miss = {k: 0 for k in m.METRICS}
        miss_alt = {k: 0 for k in m.METRICS}
        fl = {k: 0 for k in cb.FLAGS}
        ind = {k: 0 for k in m.METRICS}
        obj_picks = {"A": 0, "B": 0}
        for s in range(SEEDS):
            c = fn(s)
            t = cb.metric_table(c)
            obj_picks[t["objective_pick"]] += 1
            for k in m.METRICS:
                miss[k] += t["picks"][k] != t["objective_pick"]
                if "alt_objective" in c:
                    miss_alt[k] += t["picks"][k] != t["alt_objective_pick"]
            boot = cb.paired_bootstrap(c, n_boot=N_BOOT, seed=s)
            for k in m.METRICS:
                ind[k] += boot[k]["contains_zero"]
            for k, v in cb.read_table(t, boot).items():
                fl[k] += v
        flips[name] = {"miss": miss, "objective_picks": obj_picks}
        if "alt_objective" in fn(0):
            flips_alt[name] = miss_alt
        flag_rate[name] = fl
        indistinct_metric[name] = ind
    return ({"seeds": SEEDS, "cases": flips}, {"seeds": SEEDS, "cases": flips_alt},
            {"seeds": SEEDS, "cases": flag_rate}, {"seeds": SEEDS, "n_boot": N_BOOT, "contains_zero_count": indistinct_metric})


def e5_rank_equivalence():
    """같은 평가 데이터라면 R^2 = 1 - SSE/SST에서 SST가 두 모델에 같고, MASE = MAE / (같은 분모)라서 순위가 같아야 한다."""
    total, r2_vs_rmse, mase_vs_mae, cv_vs_rmse = 0, 0, 0, 0
    for name in cases.CASE_NAMES:
        for s in range(SEEDS):
            c = cases.MAKERS[name](s)
            t = cb.metric_table(c)
            total += 1
            r2_vs_rmse += t["picks"]["r2"] != t["picks"]["rmse"]
            mase_vs_mae += t["picks"]["mase"] != t["picks"]["mae"]
            cv_vs_rmse += m.better("rmse", t["A"]["cv_rmse"], t["B"]["cv_rmse"]) != t["picks"]["rmse"]
    return {"comparisons": total, "r2_vs_rmse_disagree": int(r2_vs_rmse), "mase_vs_mae_disagree": int(mase_vs_mae),
            "cv_rmse_vs_rmse_disagree": int(cv_vs_rmse)}


def e6_followups():
    # 사례 5: 검증 달의 평균 오차만큼 더한 뒤 다시 잰다
    picks_after = {k: {"A": 0, "B": 0} for k in m.METRICS}
    for s in range(SEEDS):
        c = cases.make_case_fixable_bias(s)
        fixed = dict(c)
        for w in "AB":
            fixed[w] = c[w] + float(np.mean(c["y_val"] - c[w + "_val"]))
        t = cb.metric_table(fixed)
        for k in m.METRICS:
            picks_after[k][t["picks"][k]] += 1
    c = cases.make_case_fixable_bias(0)
    shift = {w: float(np.mean(c["y_val"] - c[w + "_val"])) for w in "AB"}
    seed0_after = {w: m.metric_row(c["y_train"], c["y_test"], c[w] + shift[w]) for w in "AB"}
    # 사례 6: 학습 구간 R^2 와 평가 구간 R^2
    tr, te = [], []
    for s in range(SEEDS):
        c6 = cases.make_case_extrapolation(s)
        tr.append(m.r2(c6["y_train"], c6["A_train_fit"]))
        te.append(m.r2(c6["y_test"], c6["A"]))
    tr, te = np.array(tr), np.array(te)
    # 연습 문제용: 사례 3의 시드 1 표, 사례 1을 평가 50개로 줄였을 때 시드 0의 부트스트랩, 문턱 25에서의 사례 1 목적
    c3 = cases.make_case_flat_level(1)
    ex_flat = {w: {k: v for k, v in m.metric_row(c3["y_train"], c3["y_test"], c3[w]).items() if k in m.METRICS} for w in "AB"}
    c1s = cases.make_case_spiky(0, n_test=50)
    ex_small = cb.paired_bootstrap(c1s, metrics=("mae", "rmse"), n_boot=N_BOOT, seed=0)
    c1 = cases.make_case_spiky(0)
    ex_tau = {w: cases.big_miss_cost(c1["y_test"] - c1[w], tau=25.0) for w in "AB"}
    return {"exercise": {"flat_level_seed1": ex_flat, "spiky_n50_seed0_bootstrap": ex_small, "spiky_tau25_seed0": ex_tau},
            "fixable_bias": {"shift_seed0": shift, "seed0_after": seed0_after, "picks_after_100_seeds": picks_after},
            "extrapolation": {"A_train_r2_seed0": float(tr[0]), "A_test_r2_seed0": float(te[0]),
                              "A_train_r2_min": float(tr.min()), "A_test_r2_max": float(te.max()),
                              "A_test_r2_median": float(np.median(te)), "test_r2_below_zero": int(np.sum(te < 0))}}


def e7_fixture_torch():
    ytr, yte, a, b = cases.fixture_small()
    fx = {w: m.metric_row(ytr, yte, p) for w, p in (("A", a), ("B", b))}
    worst = 0.0
    per_case = {}
    for name in cases.CASE_NAMES:
        c = cases.MAKERS[name](0)
        d = 0.0
        for w in "AB":
            nps = m.metric_row(c["y_train"], c["y_test"], c[w])
            ts = tm.metrics(torch.tensor(c["y_train"]), torch.tensor(c["y_test"]), torch.tensor(c[w]))
            d = max(d, max(abs(nps[k] - ts[k]) / max(1.0, abs(nps[k])) for k in m.METRICS))
        per_case[name] = d
        worst = max(worst, d)
    return {"fixture": {"y_train": ytr.tolist(), "y_test": yte.tolist(), "A": a.tolist(), "B": b.tolist(),
                        "A_metrics": {k: fx["A"][k] for k in m.METRICS}, "B_metrics": {k: fx["B"][k] for k in m.METRICS}},
            "numpy_torch_max_rel_diff": per_case, "numpy_torch_max_rel_diff_all": worst}


def build():
    e2, e2_alt, e3, e4 = e2_e3_e4_seeds()
    return {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "torch_threads": torch.get_num_threads()},
        "E0": e0_first_screen(), "E1": e1_seed0(), "E2": e2, "E2_alt": e2_alt, "E3": e3, "E4": e4,
        "E5": e5_rank_equivalence(), "E6": e6_followups(), "E7": e7_fixture_torch(),
    }


def main():
    res = build()
    (HERE / "results").mkdir(exist_ok=True)
    text = cb.dumps(res)
    (HERE / "results" / "ch07.json").write_text(text)
    (HERE / "results" / "report.md").write_text(cb.render_report(res))
    print(text)


if __name__ == "__main__":
    main()
