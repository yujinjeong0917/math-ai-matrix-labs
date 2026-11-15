"""모델 평가 5장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch05_mase/experiments.py` 로 실행한다.

E0 첫 화면: 도서관 방문자 5일 학습, 2일 평가. 평균 예측과 '마지막 값 그대로' 예측
E1 실패: 수준 1000 랜덤워크(하루 변화 표준편차 5). MAPE는 작은데 naive보다 못한 평균 예측.
   비교 방식 두 가지를 구분한다: (가) 학습 끝에서 한 번 예측(100일 앞까지), (나) 매일 전날 실제값을 보고 다시 예측
E2 예측 시계 h = 1, 5, 20의 naive 예측 MASE: 1스텝 분모와 h스텝 분모, 이론값 sqrt(h)
E3 여러 계열 평균: 큰 가게(랜덤워크, 수준 1000) + 작은 가게(평균 20 근처), 평균 MAPE와 평균 MASE의 승자
E4 0이 섞인 간헐 수요: MAPE 정의 여부, eps 정책의 승자, MASE의 승자
E5 분모의 한계: 학습 구간이 상수이거나 거의 평평할 때
E6 무엇과 비교하나: 요일 효과가 큰 계열에서 1스텝 naive 분모와 계절 naive(lag 7) 분모
E7 손 계산 fixture와 NumPy-PyTorch 대조
결과는 results/ch05.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
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
import eval_ch05_cases as cases  # noqa: E402
import eval_ch05_metrics as m  # noqa: E402
import eval_ch05_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch05.json"
SEEDS = 100


# 평균과 5·50·95 백분위수를 한 번에 요약
def q(a, p=(5, 50, 95)):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), **{f"p{k}": float(np.percentile(a, k)) for k in p}}


def e0_first_screen():
    tr, te = cases.first_screen()
    d = m.naive_mae_in_sample(tr)
    a = m.mean_forecast(tr, 2)
    nv = m.naive_forecast(tr, 2)
    return {
        "train": tr.tolist(), "test": te.tolist(), "naive_diffs": np.abs(np.diff(tr)).tolist(), "denominator": d,
        "mean_forecast": float(a[0]), "mean_abs_err": np.abs(te - a).tolist(), "mean_mae": m.mae(te, a),
        "mean_mape": m.mape(te, a), "mean_mase": m.mase(tr, te, a),
        "naive_forecast": float(nv[0]), "naive_abs_err": np.abs(te - nv).tolist(), "naive_mae": m.mae(te, nv),
        "naive_mape": m.mape(te, nv), "naive_mase": m.mase(tr, te, nv),
        "with_zero_day": {"train": [0.0, 2.0, 1.0, 0.0, 3.0], "test": [0.0, 2.0], "pred": [1.0, 1.0],
                          "mase": m.mase([0, 2, 1, 0, 3], [0, 2], [1, 1]),
                          "mape": str(m.ape([0, 2], [1, 1]).tolist())},
    }


# 세 예측: 평균 고정(A), 마지막 값 고정(N_fixed), 매일 전날 실제값(N_rolling)
def _models_rw(tr, te):
    h = len(te)
    return {"A_mean_fixed": m.mean_forecast(tr, h), "N_fixed": m.naive_forecast(tr, h), "N_rolling": m.naive_rolling(tr, te)}


def e1_random_walk():
    tr, te = cases.case_random_walk_level(seed=0)
    one = {"seed": 0, "level_train_mean": float(tr.mean()), "test_min_max": [float(te.min()), float(te.max())],
           "denominator": m.naive_mae_in_sample(tr), "in_sample_naive_mase": m.mase(tr, tr[1:], tr[:-1])}
    for k, p in _models_rw(tr, te).items():
        one[k] = {"mape": m.mape(te, p), "mae": m.mae(te, p), "mase": m.mase(tr, te, p)}
    many = {k: {"mape": [], "mase": []} for k in ("A_mean_fixed", "N_fixed", "N_rolling")}
    a_mape_below_1pct_and_mase_above_1 = 0
    a_beats_nfixed = 0
    for s in range(SEEDS):
        tr, te = cases.case_random_walk_level(seed=1000 + s)
        ps = _models_rw(tr, te)
        v = {k: (m.mape(te, p), m.mase(tr, te, p)) for k, p in ps.items()}
        for k in v:
            many[k]["mape"].append(v[k][0]); many[k]["mase"].append(v[k][1])
        # MAPE는 1% 미만인데 MASE는 1 초과인 시드 수. 수준이 크면 MAPE만으로는 못한 모델이 드러나지 않는다
        a_mape_below_1pct_and_mase_above_1 += v["A_mean_fixed"][0] < 1.0 and v["A_mean_fixed"][1] > 1.0
        a_beats_nfixed += v["A_mean_fixed"][1] < v["N_fixed"][1]
    summ = {k: {"mape": q(d["mape"]), "mase": q(d["mase"])} for k, d in many.items()}
    return {"seed0": one, "seeds": SEEDS, "summary": summ,
            "A_mape_below_1pct_and_mase_above_1": int(a_mape_below_1pct_and_mase_above_1),
            "A_beats_N_fixed_by_mase": int(a_beats_nfixed)}


def e2_horizon():
    out = {}
    for h in (1, 5, 20):
        one_step, h_step = [], []
        for s in range(SEEDS):
            tr, te = cases.case_random_walk_level(seed=2000 + s)
            p = m.naive_rolling(tr, te, lag=h)  # 평가 시점 t를 h일 전 실제값으로 예측
            # 같은 예측 p를 1스텝 분모와 h스텝 분모로 각각 나눠 비교. 랜덤워크에서 h일 전 값 예측의 오차는 약 sqrt(h)배라 1스텝 분모 MASE의 이론값은 sqrt(h), h스텝 분모는 약 1
            one_step.append(m.mase(tr, te, p, lag=1))
            h_step.append(m.mase(tr, te, p, lag=h))
        out[str(h)] = {"theory_sqrt_h": float(np.sqrt(h)), "scaled_by_1step": q(one_step), "scaled_by_hstep": q(h_step)}
    return out


def _panel(seed):
    # 큰 가게(랜덤워크, 수준 1000)와 작은 가게(수준 20)를 시드 대역을 달리해 독립으로 만든다
    big = cases.case_random_walk_level(seed=3000 + seed)
    small = cases.case_iid_level(seed=4000 + seed)
    res = {}
    for name, (tr, te) in (("big", big), ("small", small)):
        h = len(te)
        res[name] = {}
        for mk, p in (("A_mean", m.mean_forecast(tr, h)), ("B_naive", m.naive_forecast(tr, h))):
            res[name][mk] = {"mape": m.mape(te, p), "mase": m.mase(tr, te, p), "mae": m.mae(te, p)}
    # 계열 평균(MASE는 단위가 없어 큰 가게가 지배하지 않지만, MAE 평균은 단위가 큰 가게가 지배한다)
    avg = {mk: {k: float(np.mean([res[s][mk][k] for s in ("big", "small")])) for k in ("mape", "mase", "mae")}
           for mk in ("A_mean", "B_naive")}
    return res, avg


def e3_panel():
    res0, avg0 = _panel(0)
    counts = {"mape_picks_A": 0, "mase_picks_A": 0, "mae_picks_A": 0, "mape_A_mase_B": 0,
              "big_mase_picks_A": 0, "small_mase_picks_A": 0, "mase_mae_pick_differ": 0}
    share_of_big_in_A_mase = []
    for s in range(SEEDS):
        res, avg = _panel(s)
        ma = avg["A_mean"]["mape"] < avg["B_naive"]["mape"]
        sa = avg["A_mean"]["mase"] < avg["B_naive"]["mase"]
        counts["mape_picks_A"] += ma
        counts["mase_picks_A"] += sa
        mae_a = avg["A_mean"]["mae"] < avg["B_naive"]["mae"]
        counts["mae_picks_A"] += mae_a
        counts["mase_mae_pick_differ"] += sa != mae_a
        counts["mape_A_mase_B"] += ma and not sa
        counts["big_mase_picks_A"] += res["big"]["A_mean"]["mase"] < res["big"]["B_naive"]["mase"]
        counts["small_mase_picks_A"] += res["small"]["A_mean"]["mase"] < res["small"]["B_naive"]["mase"]
        share_of_big_in_A_mase.append(res["big"]["A_mean"]["mase"] / (res["big"]["A_mean"]["mase"] + res["small"]["A_mean"]["mase"]))
    return {"seed0": {"per_series": res0, "average": avg0}, "seeds": SEEDS,
            "counts": {k: int(v) for k, v in counts.items()},
            "share_of_big_series_in_A_mase_sum": q(share_of_big_in_A_mase)}


def e4_intermittent(n=120, n_train=80):
    out = {}
    for pz in (0.0, 0.05, 0.2):
        c = {"mape_undefined": 0, "eps_Z_best": 0, "mase_A_best": 0, "mase_Z_best": 0, "mase_finite": 0}
        mase_vals = {"A": [], "B": [], "Z": []}
        for s in range(SEEDS):
            y = cases.case_intermittent(pz, n=n, seed=5000 + s)
            tr, te = y[:n_train], y[n_train:]
            a = float(tr.mean())
            preds = {"A": np.full(len(te), a), "B": np.full(len(te), 0.6 * a), "Z": np.zeros(len(te))}
            ms = {k: m.mase(tr, te, p) for k, p in preds.items()}
            for k in ms:
                mase_vals[k].append(ms[k])
            c["mase_finite"] += all(np.isfinite(list(ms.values())))
            # 평가 구간에 0이 하나라도 있으면 정의대로의 MAPE는 계산 불가
            c["mape_undefined"] += bool((te == 0).any())
            ep = {k: m.mape(te, p, zero="eps") for k, p in preds.items()}
            # eps 정책에서 늘 0 예측(Z)이 이기는 횟수. 4장의 함정이 그대로 나타난다
            c["eps_Z_best"] += ep["Z"] < min(ep["A"], ep["B"])
            c["mase_A_best"] += ms["A"] < min(ms["B"], ms["Z"])
            c["mase_Z_best"] += ms["Z"] < min(ms["A"], ms["B"])
        out[str(pz)] = {"counts": {k: int(v) for k, v in c.items()},
                        "mase_median": {k: float(np.median(v)) for k, v in mase_vals.items()}}
    y = cases.case_intermittent(0.2, n=n, seed=5000)
    tr = y[:n_train]
    out["example_seed5000_p0.2"] = {"train_first10": tr[:10].tolist(), "zeros_in_train": int((tr == 0).sum()),
                                     "denominator": m.naive_mae_in_sample(tr)}
    return out


def e5_denominator_limits():
    try:
        m.naive_mae_in_sample(np.full(10, 7.0))
        const = "no error"
    except ValueError as e:
        const = f"ValueError: {e}"
    rows = {}
    # jump가 작을수록 분모(jump/99)가 작아 같은 오차 1에서 MASE = 99/jump로 커진다
    for jump in (1.0, 10.0, 50.0):
        tr, te, p = cases.case_nearly_flat(jump=jump)
        rows[str(jump)] = {"denominator": m.naive_mae_in_sample(tr), "test_mae": m.mae(te, p), "mase": m.mase(tr, te, p)}
    return {"constant_train": const, "nearly_flat_n100_err1": rows}


def e6_seasonal():
    tr, te = cases.case_weekly(seed=0)
    h = len(te)
    # 요일별 학습 평균 7개(인덱스 i, i+7, i+14, ...의 평균)
    wk = np.array([tr[i::7].mean() for i in range(7)])
    preds = {
        "overall_mean": np.full(h, tr.mean()),
        "naive_rolling_lag1": m.naive_rolling(tr, te, lag=1),
        "seasonal_naive_rolling_lag7": m.naive_rolling(tr, te, lag=7),
        # 평가 구간 h = 28일이므로 4주 반복(h가 7의 배수라는 가정)
        "weekday_mean": np.tile(wk, h // 7),
    }
    d1, d7 = m.naive_mae_in_sample(tr, 1), m.naive_mae_in_sample(tr, 7)
    rows = {k: {"mae": m.mae(te, p), "mase_lag1": m.mase(tr, te, p, 1), "mase_lag7": m.mase(tr, te, p, 7)} for k, p in preds.items()}
    many = {k: {"lag1": [], "lag7": []} for k in preds}
    for s in range(SEEDS):
        tr2, te2 = cases.case_weekly(seed=6000 + s)
        wk2 = np.array([tr2[i::7].mean() for i in range(7)])
        ps = {"overall_mean": np.full(h, tr2.mean()), "naive_rolling_lag1": m.naive_rolling(tr2, te2, 1),
              "seasonal_naive_rolling_lag7": m.naive_rolling(tr2, te2, 7), "weekday_mean": np.tile(wk2, h // 7)}
        for k, p in ps.items():
            many[k]["lag1"].append(m.mase(tr2, te2, p, 1)); many[k]["lag7"].append(m.mase(tr2, te2, p, 7))
    return {"seed0": {"denominator_lag1": d1, "denominator_lag7": d7, "models": rows},
            "seeds": SEEDS, "summary": {k: {"lag1": q(v["lag1"]), "lag7": q(v["lag7"])} for k, v in many.items()}}


def e7_fixture_torch():
    tr, te, p = cases.fixture_small()
    # fixture 값이 정수라 Fraction으로 분모·MASE를 정확한 분수로 구해 부동소수 결과와 대조
    exact_den = sum(Fraction(abs(int(a) - int(b))) for a, b in zip(tr[1:], tr[:-1])) / (len(tr) - 1)
    exact = sum(Fraction(abs(int(a) - int(b))) for a, b in zip(te, p)) / len(te) / exact_den
    rng = np.random.default_rng(7)
    tr2 = 50 + np.cumsum(rng.normal(0, 2, 257)); te2 = tr2[-1] + np.cumsum(rng.normal(0, 2, 64)); p2 = te2 + rng.normal(0, 3, 64)
    diffs = {}
    for lag in (1, 7):
        a = m.mase(tr2, te2, p2, lag)
        b = tm.mase(torch.tensor(tr2), torch.tensor(te2), torch.tensor(p2), lag)
        diffs[str(lag)] = abs(a - b)
    return {"fixture": {"train": tr.tolist(), "test": te.tolist(), "pred": p.tolist(),
                        "exact_denominator": str(exact_den), "exact_mase": str(exact), "numpy_mase": m.mase(tr, te, p)},
            "numpy_torch_abs_diff": diffs}


def main():
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "E0": e0_first_screen(), "E1": e1_random_walk(), "E2": e2_horizon(), "E3": e3_panel(),
        "E4": e4_intermittent(), "E5": e5_denominator_limits(), "E6": e6_seasonal(), "E7": e7_fixture_torch(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
