"""모델 평가 4장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch04_mape/experiments.py` 로 실행한다.

E0 첫 화면 예시: 세 가게(정답 100, 10, 1)를 모두 2개씩 틀린 예측
E1 실패 (a) 0 근처 (b) 0이 있을 때, 정책별 값과 sklearn 문서 예제 재현 (c) 비대칭 두 관점과 상한
E2 비교 측정 1: 간헐 수요(0 비율 0, 0.05, 0.2) 시드 100개에서 MAPE 순위와 MAE 순위, 0 처리 정책별
E3 비교 측정 2: 수준 이동(정답과 예측에 같은 c를 더함)과 기온 단위(섭씨·화씨·켈빈)
E4 MAPE를 가장 작게 만드는 상수 예측: 1/y 가중 중앙값, 중앙값·평균과 비교
E5 NumPy-PyTorch 대조

결과는 results/ch04.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
"""

import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch04_cases as cases  # noqa: E402
import eval_ch04_metrics as m  # noqa: E402
import eval_ch04_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch04.json"


def _try(f, *a, **k):
    try:
        return f(*a, **k)
    except ValueError as e:
        return f"ValueError: {e}"


def e0_first_screen():
    y, p = cases.first_screen()
    return {
        "y": y.tolist(), "pred": p.tolist(), "abs_err": np.abs(y - p).tolist(),
        "ape": m.ape(y, p).tolist(), "mape": m.mape(y, p), "mae": m.mae(y, p),
        "mape_without_store3": m.mape(y[:2], p[:2]),
        "with_zero_store": _try(m.mape, np.append(y, 0.0), np.append(p, 2.0)),
    }


def e1_failures():
    y, p = cases.case_near_zero()
    a = {"y": y.tolist(), "pred": p.tolist(), "ape": m.ape(y, p).tolist(), "mape": m.mape(y, p), "mae": m.mae(y, p)}
    yz, pz = cases.case_with_zero()
    b = {
        "y": yz.tolist(), "pred": pz.tolist(), "ape": [str(v) for v in m.ape(yz, pz)],
        "raise": _try(m.mape, yz, pz, zero="raise"),
        "drop_pct": m.mape(yz, pz, zero="drop"),
        "eps_pct": m.mape(yz, pz, zero="eps"),
        "eps_ratio_like_sklearn": m.mape(yz, pz, zero="eps") / 100.0,
        "one_over_2eps": 1.0 / (2.0 * m.EPS),
        "eps": m.EPS,
        "zero_forecast_on_zero_day_ape_eps": float(100.0 * abs(0.0 - 0.0) / max(0.0, m.EPS)),
    }
    yd, pd = cases.case_sklearn_doc_example()
    b["sklearn_doc_example"] = {
        "y": yd.tolist(), "pred": pd.tolist(),
        "ours_eps_ratio": m.mape(yd, pd, zero="eps") / 100.0,
        "sklearn_doc_output": 112589990684262.48,
        "drop_pct": m.mape(yd, pd, zero="drop"),
    }
    ya, pa = cases.case_asymmetry_fixed_actual()
    yf, pf = cases.case_asymmetry_fixed_forecast()
    c = {
        "fixed_actual": {"y": ya.tolist(), "pred": pa.tolist(), "ape": m.ape(ya, pa).tolist()},
        "fixed_forecast": {"y": yf.tolist(), "pred": pf.tolist(), "ape": m.ape(yf, pf).tolist()},
        "bounds_y100": m.ape_bounds_demo(),
    }
    return {"a_near_zero": a, "b_zero": b, "c_asymmetry": c}


def e2_intermittent(seeds=100, n=60, lam=4.0):
    out = {}
    for pz in (0.0, 0.05, 0.2):
        models = cases.intermittent_models(pz, lam)
        res = {"forecasts": models, "n": n, "seeds": seeds}
        undefined = 0
        zeros_per_series = []
        mae_B_wins = 0
        pol = {"drop": {"B_wins": 0, "disagree_with_mae": 0, "Z_best_of_3": 0, "mape_A": [], "mape_B": [], "mape_Z": []},
               "eps": {"B_wins": 0, "disagree_with_mae": 0, "Z_best_of_3": 0, "mape_A": [], "mape_B": [], "mape_Z": []}}
        mae_vals = {"A": [], "B": [], "Z": []}
        for s in range(seeds):
            y = cases.case_intermittent_demand(pz, n, lam, seed=10_000 + s)
            nz = int(np.sum(y == 0))
            zeros_per_series.append(nz)
            undefined += nz > 0
            preds = {k: np.full(n, v) for k, v in models.items()}
            maes = {k: m.mae(y, p) for k, p in preds.items()}
            for k in maes:
                mae_vals[k].append(maes[k])
            b_mae = maes["B"] < maes["A"]
            mae_B_wins += b_mae
            for name in ("drop", "eps"):
                v = {k: m.mape(y, p, zero=name) for k, p in preds.items()}
                d = pol[name]
                for k in v:
                    d["mape_" + k].append(v[k])
                b = v["B"] < v["A"]
                d["B_wins"] += b
                d["disagree_with_mae"] += b != b_mae
                d["Z_best_of_3"] += v["Z"] < min(v["A"], v["B"])
        res["series_with_zero_raise_undefined"] = int(undefined)
        res["zeros_per_series_mean"] = float(np.mean(zeros_per_series))
        res["mae_B_wins"] = int(mae_B_wins)
        res["mae_median"] = {k: float(np.median(v)) for k, v in mae_vals.items()}
        res["mae_Z_best_of_3"] = int(sum(1 for i in range(seeds) if mae_vals["Z"][i] < min(mae_vals["A"][i], mae_vals["B"][i])))
        for name, d in pol.items():
            for k in ("A", "B", "Z"):
                arr = np.array(d.pop("mape_" + k))
                d["mape_median_" + k] = float(np.median(arr))
            d["B_wins"] = int(d["B_wins"]); d["disagree_with_mae"] = int(d["disagree_with_mae"]); d["Z_best_of_3"] = int(d["Z_best_of_3"])
        res["policies"] = pol
        out[str(pz)] = res
    return out


def e3_level_and_units():
    y, p = cases.case_level_shift()
    shift = {}
    for c in (0.0, 10.0, 100.0, 1000.0):
        shift[str(c)] = {"mean_y": float((y + c).mean()), "mae": m.mae(y + c, p + c), "mape": m.mape(y + c, p + c)}
    tc, fc = cases.case_temperature()
    units = {}
    for name, f in (("celsius", lambda v: np.asarray(v, float)), ("fahrenheit", cases.to_fahrenheit), ("kelvin", cases.to_kelvin)):
        a, b = f(tc), f(fc)
        units[name] = {"y": [round(v, 2) for v in a.tolist()], "pred": [round(v, 2) for v in b.tolist()],
                       "ape": m.ape(a, b).tolist(), "mape": m.mape(a, b), "mae": m.mae(a, b)}
    return {"level_shift": {"n": len(y), "seed": 0, "by_c": shift}, "temperature": units}


def e4_best_constant():
    card = np.array([50.0, 150.0])
    out = {"card_50_150": {"best": m.best_constant_mape(card),
                           "mape_at": {str(c): m.mape(card, np.full(2, c)) for c in (50.0, 100.0, 150.0)},
                           "mae_at": {str(c): m.mae(card, np.full(2, c)) for c in (50.0, 100.0, 150.0)}}}
    rows = {}
    series = {"1+Poisson(4)": cases.case_intermittent_demand(0.0, n=60, lam=4.0, seed=0),
              "로그정규(2, 0.8)": cases.case_skewed_sales(n=60, seed=0)}
    for lam, y in series.items():
        cands = {"MAPE 최적": m.best_constant_mape(y), "중앙값": float(np.median(y)), "평균": float(y.mean())}
        rows[str(lam)] = {
            "y_min_max": [float(y.min()), float(y.max())],
            "constants": cands,
            "mape": {k: m.mape(y, np.full(len(y), c)) for k, c in cands.items()},
            "mae": {k: m.mae(y, np.full(len(y), c)) for k, c in cands.items()},
            "bias_mean_y_minus_c": {k: float(np.mean(y - c)) for k, c in cands.items()},
        }
    out["by_series"] = rows
    return out


def e5_torch():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.5, 20.0, 257)
    p = y + rng.normal(0.0, 1.0, 257)
    yz = y.copy(); yz[::10] = 0.0
    yt, pt, yzt = torch.tensor(y), torch.tensor(p), torch.tensor(yz)
    yd, pd = cases.case_sklearn_doc_example()
    return {
        "raise_abs_diff": abs(m.mape(y, p) - tm.mape(yt, pt)),
        "drop_abs_diff": abs(m.mape(yz, p, zero="drop") - tm.mape(yzt, pt, zero="drop")),
        "eps_rel_diff": abs(m.mape(yz, p, zero="eps") - tm.mape(yzt, pt, zero="eps")) / m.mape(yz, p, zero="eps"),
        "mae_abs_diff": abs(m.mae(y, p) - tm.mae(yt, pt)),
        "torch_sklearn_doc_example_ratio": tm.mape(torch.tensor(yd), torch.tensor(pd), zero="eps") / 100.0,
    }


def main():
    res = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__.split("+")[0],
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "E0": e0_first_screen(),
        "E1": e1_failures(),
        "E2": e2_intermittent(),
        "E3": e3_level_and_units(),
        "E4": e4_best_constant(),
        "E5": e5_torch(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
