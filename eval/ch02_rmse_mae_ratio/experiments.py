"""모델 평가 2장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch02_rmse_mae_ratio/experiments.py` 로 실행한다.

E0 첫 화면 예시: 오차 5개짜리 세 모델(1장의 A·B + 중간 모델 C)의 MAE·RMSE·비율
E1 실패 최소 예제: n=100, 크기가 모두 같은 오차 / 하나만 큰 오차 / 정규 오차. MAE는 모두 1 근처
E2 기준값: 균등·정규·라플라스·t(3) 오차 n=10^6 표본 비율 vs 해석적 값
E3 표본 크기: 정규·라플라스·t(3)·t(2) x n=10..10^4 x 시드 100개, 비율의 평균과 5~95% 구간
E4 순위 안정성: MAE가 거의 같은 두 모델(정규 vs t(3)), n=1000, 시드 100개에서 MAE 순위와 비율 순위가 몇 번 뒤집히나
E5 같은 비율, 다른 모양: 63.7%만 크기 1이고 나머지 0인 오차가 정규 오차와 같은 비율을 낸다
E6 분해: 비율^2 = 1 + Var(|e|)/MAE^2, 그리고 Chai 등(2009)의 비율 1.63~2.29가 뜻하는 퍼짐

결과는 results/ch02.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
"""

import json
import math
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch02_cases as cases  # noqa: E402
import eval_ch02_metrics as m  # noqa: E402
import eval_ch02_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch02.json"


def _stats(e):
    z = np.zeros_like(e)
    return {"mae": m.mae(z, e), "rmse": m.rmse(z, e), "ratio": m.rmse_mae_ratio(z, e)}


def e0_first_screen():
    models = {"A": [2.0, 2, 2, 2, 2], "C": [1.0, 1, 2, 3, 3], "B": [0.0, 0, 0, 0, 9]}
    out = {k: {"errors": v, "sum_abs": float(np.sum(np.abs(v))), "sum_sq": float(np.sum(np.square(v))), **_stats(np.array(v))} for k, v in models.items()}
    out["upper_bound_n5"] = m.ratio_bounds(5)[1]
    return out


def e1_failure(n=100):
    _, uni = cases.case_uniform_errors(n, seed=0)
    _, spike = cases.case_single_spike(n, seed=0)
    gauss = cases.sample_errors("normal", n, seed=0)
    return {
        "n": n,
        "bounds": list(m.ratio_bounds(n)),
        "uniform_size": _stats(uni),
        "single_spike": _stats(spike),
        "normal_unit_mae": _stats(gauss),
        "normal_reference": m.ratio_reference("normal"),
    }


def e2_reference(n=1_000_000):
    rows = {}
    for dist, df in (("uniform", None), ("normal", None), ("laplace", None), ("t", 3)):
        e = cases.sample_errors(dist, n, seed=0, df=df)
        key = dist if df is None else f"t{df}"
        rows[key] = {"sample_ratio": _stats(e)["ratio"], "reference": m.ratio_reference(dist, df=df)}
    return {"n": n, "rows": rows}


def e3_sample_size(seeds=100):
    out = {}
    for dist, df in (("normal", None), ("laplace", None), ("t", 3), ("t", 2)):
        key = dist if df is None else f"t{df}"
        out[key] = {"reference": m.ratio_reference(dist, df=df) if not (dist == "t" and df <= 2) else None}
        for n in (10, 100, 1000, 10000):
            r = np.array([_stats(cases.sample_errors(dist, n, seed=s, df=df))["ratio"] for s in range(seeds)])
            out[key][str(n)] = {
                "mean": float(r.mean()),
                "p5": float(np.percentile(r, 5)),
                "p95": float(np.percentile(r, 95)),
                "max_possible": math.sqrt(n),
            }
    return {"seeds": seeds, "by_dist": out}


def e4_rank_stability(n=1000, seeds=100):
    rows = []
    for s in range(seeds):
        a = cases.sample_errors("normal", n, seed=10_000 + s)
        b = cases.sample_errors("t", n, seed=20_000 + s, df=3)
        rows.append((_stats(a), _stats(b)))
    mae_a = np.array([r[0]["mae"] for r in rows])
    mae_b = np.array([r[1]["mae"] for r in rows])
    rat_a = np.array([r[0]["ratio"] for r in rows])
    rat_b = np.array([r[1]["ratio"] for r in rows])
    return {
        "n": n,
        "seeds": seeds,
        "A": "normal, E|e|=1",
        "B": "t(3), E|e|=1",
        "mean_mae_A": float(mae_a.mean()),
        "mean_mae_B": float(mae_b.mean()),
        "mean_ratio_A": float(rat_a.mean()),
        "mean_ratio_B": float(rat_b.mean()),
        "ratio_B_p5_p95": [float(np.percentile(rat_b, 5)), float(np.percentile(rat_b, 95))],
        "ratio_A_p5_p95": [float(np.percentile(rat_a, 5)), float(np.percentile(rat_a, 95))],
        "seeds_B_lower_mae": int(np.sum(mae_b < mae_a)),
        "seeds_B_higher_ratio": int(np.sum(rat_b > rat_a)),
        "reference_A": m.ratio_reference("normal"),
        "reference_B": m.ratio_reference("t", df=3),
    }


def e5_same_ratio_different_shape(n=1000):
    frac = 2 / math.pi  # 비율 = 1/sqrt(frac) = sqrt(pi/2)
    _, two = cases.case_two_level(n, frac, seed=0)
    g = cases.sample_errors("normal", n, seed=0)
    def shape(e):
        a = np.abs(e)
        return {**_stats(e), "share_exact_zero": float(np.mean(a == 0)), "max_abs": float(a.max()), "median_abs": float(np.median(a))}
    return {
        "n": n,
        "frac_nonzero": frac,
        "k_nonzero": int(round(frac * n)),
        "two_level": shape(two),
        "normal": shape(g),
        "normal_reference": m.ratio_reference("normal"),
    }


def e6_decomposition():
    rng = np.random.default_rng(5)
    e = rng.standard_t(3, 500)
    implied = {}
    for r in (1.63, 2.29):
        v = r**2 - 1  # Var(|e|)/MAE^2
        implied[str(r)] = {"var_over_mae2": v, "sd_abs_over_mae": math.sqrt(v)}
    return {
        "check_ratio": _stats(e)["ratio"],
        "check_via_variance": m.ratio_via_variance(e),
        "torch_ratio": tm.rmse_mae_ratio(torch.zeros(500, dtype=torch.float64), torch.tensor(e)),
        "chai2009_range_implied": implied,
        "first_screen_B": {"var_abs": float(np.var(np.abs([0.0, 0, 0, 0, 9]))), "mae": 1.8},
    }


def main():
    res = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
        "E0": e0_first_screen(),
        "E1": e1_failure(),
        "E2": e2_reference(),
        "E3": e3_sample_size(),
        "E4": e4_rank_stability(),
        "E5": e5_same_ratio_different_shape(),
        "E6": e6_decomposition(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
