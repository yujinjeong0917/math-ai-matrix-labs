"""모델 평가 1장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch01_mae_rmse/experiments.py` 로 실행한다.

E0 첫 화면 예시: 오차 5개짜리 두 모델에서 MAE와 RMSE가 서로 다른 모델을 고른다
E1 실패 최소 예제: n=1000, 정규 오차 모델 A vs 드문 큰 실수 모델 B
E2 경계: B의 큰 실수 개수 k를 0..60으로 바꾸며 두 지표의 승자가 바뀌는 곳
E3 순위 안정성: 큰 실수 개수까지 무작위인 B와 A를 시드 100개로 다시 뽑아 승자가 몇 번 바뀌나
E4 표본 크기: 정규 오차에서 n=4..10^6, 시드 5개의 MAE·RMSE (Chai·Draxler 2014 Table 1과 같은 설정)
E5 상수 예측: MAE를 최소화하는 상수는 중앙값, MSE는 평균 (NumPy 격자 탐색 + PyTorch 경사하강)
E6 오차 분포와 추정의 흔들림: 정규 vs 라플라스 잡음에서 평균과 중앙값 중 누가 덜 흔들리나

결과는 results/ch01.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
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
import eval_ch01_cases as cases  # noqa: E402
import eval_ch01_metrics as m  # noqa: E402
import eval_ch01_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch01.json"
SQRT_2_OVER_PI = float(np.sqrt(2 / np.pi))


def e0_first_screen():
    a = np.array([2.0, 2, 2, 2, 2])
    b = np.array([0.0, 0, 0, 0, 9])
    zero = np.zeros(5)
    y_const = np.array([1.0, 2, 3, 4, 10])
    return {
        "errors_A": a.tolist(),
        "errors_B": b.tolist(),
        "A": {"mae": m.mae(zero, a), "rmse": m.rmse(zero, a)},
        "B": {"mae": m.mae(zero, b), "rmse": m.rmse(zero, b), "mse": m.mse(zero, b)},
        "constant_example_y": y_const.tolist(),
        "constant_example_median": float(np.median(y_const)),
        "constant_example_mean": float(np.mean(y_const)),
        "constant_example_scores": {
            str(c): {"mae": m.mae(y_const, np.full(5, c)), "rmse": m.rmse(y_const, np.full(5, c))} for c in (3.0, 4.0)
        },
    }


def e1_failure(n=1000, k=10):
    ya, pa = cases.case_gaussian_errors(n, seed=0)
    yb, pb = cases.case_rare_large_errors(n, k, seed=0)
    return {
        "n": n,
        "k_big": k,
        "A_sample": {"mae": m.mae(ya, pa), "rmse": m.rmse(ya, pa)},
        "A_expected": {"mae": SQRT_2_OVER_PI, "rmse": 1.0},
        "B": {"mae": m.mae(yb, pb), "rmse": m.rmse(yb, pb)},
        "B_hand": {
            "mae": (990 * 0.5 + 10 * 20) / 1000,
            "rmse": float(np.sqrt((990 * 0.25 + 10 * 400) / 1000)),
        },
        "torch_check": {"A_mae": tm.mae(torch.tensor(ya), torch.tensor(pa)), "B_rmse": tm.rmse(torch.tensor(yb), torch.tensor(pb))},
    }


def e2_boundary(n=1000):
    ya, pa = cases.case_gaussian_errors(n, seed=0)
    a_mae, a_rmse = m.mae(ya, pa), m.rmse(ya, pa)
    rows = []
    for k in range(0, 61):
        yb, pb = cases.case_rare_large_errors(n, k, seed=0)
        rows.append({"k": k, "mae_B": m.mae(yb, pb), "rmse_B": m.rmse(yb, pb)})
    last_rmse_prefers_B = max(r["k"] for r in rows if r["rmse_B"] < a_rmse)
    last_mae_prefers_B = max(r["k"] for r in rows if r["mae_B"] < a_mae)
    # 기대값 기준 해석적 경계: RMSE  ((n-k)0.25 + 400k)/n < 1,  MAE ((n-k)0.5 + 20k)/n < sqrt(2/pi)
    k_rmse = 0.75 * n / (400 - 0.25)
    k_mae = (SQRT_2_OVER_PI - 0.5) * n / (20 - 0.5)
    show = {0, 1, 2, 5, 10, 15, 16, 20, 50}
    return {
        "n": n,
        "A_sample": {"mae": a_mae, "rmse": a_rmse},
        "rows": [r for r in rows if r["k"] in show],
        "last_k_rmse_prefers_B": last_rmse_prefers_B,
        "last_k_mae_prefers_B": last_mae_prefers_B,
        "analytic_k_rmse_threshold": k_rmse,
        "analytic_k_mae_threshold": k_mae,
        "ratio_rmse_over_mae": {str(r["k"]): r["rmse_B"] / r["mae_B"] for r in rows if r["k"] in show},
        "ratio_peak_k_upto_1000": int(max(range(0, 1001), key=lambda k: np.sqrt(((n - k) * 0.25 + 400 * k) / n) / (((n - k) * 0.5 + 20 * k) / n))),
        "mse_B_k10": [r for r in rows if r["k"] == 10][0]["rmse_B"] ** 2,
        "disagree_ks": [r["k"] for r in rows if (r["mae_B"] < a_mae) != (r["rmse_B"] < a_rmse)],
    }


def e3_stability(seeds=100, p_big=0.01):
    out = {}
    for n in (100, 1000):
        mae_B_wins = rmse_B_wins = disagree = 0
        ks = []
        for s in range(seeds):
            ya, pa = cases.case_gaussian_errors(n, seed=10_000 + s)
            yb, pb = cases.case_random_rare_large_errors(n, p_big=p_big, seed=20_000 + s)
            ks.append(int(np.sum(np.abs(pb) > 1)))
            mb = m.mae(yb, pb) < m.mae(ya, pa)
            rb = m.rmse(yb, pb) < m.rmse(ya, pa)
            mae_B_wins += mb
            rmse_B_wins += rb
            disagree += mb != rb
        ks = np.array(ks)
        out[str(n)] = {
            "seeds": seeds,
            "mae_prefers_B": int(mae_B_wins),
            "rmse_prefers_B": int(rmse_B_wins),
            "metrics_disagree": int(disagree),
            "k_big_min": int(ks.min()),
            "k_big_max": int(ks.max()),
            "k_big_zero_or_one": int(np.sum(ks <= 1)),
            "k_big_zero": int(np.sum(ks == 0)),
        }
    return {"p_big": p_big, "by_n": out}


def e4_sample_size():
    rows = []
    for n in (4, 10, 100, 1000, 10_000, 100_000, 1_000_000):
        maes, rmses = [], []
        for s in range(5):
            y, p = cases.case_gaussian_errors(n, seed=s)
            maes.append(m.mae(y, p))
            rmses.append(m.rmse(y, p))
        rows.append({"n": n, "mae": maes, "rmse": rmses, "mae_range": [min(maes), max(maes)], "rmse_range": [min(rmses), max(rmses)]})
    return {"rows": rows, "mae_limit": SQRT_2_OVER_PI}


def e5_best_constant():
    res = {}
    small = np.array([1.0, 2, 3, 4, 10])
    rng = np.random.default_rng(0)
    skewed = rng.lognormal(0.0, 1.0, 1001)  # 오른쪽 꼬리가 긴 데이터(홀수 개라 중앙값이 하나로 정해진다)
    for name, y in (("small_1_2_3_4_10", small), ("lognormal_1001", skewed)):
        c_mae, step = m.best_constant(y, "mae")
        c_mse, _ = m.best_constant(y, "mse")
        res[name] = {
            "median": float(np.median(y)),
            "mean": float(np.mean(y)),
            "grid_argmin_mae": c_mae,
            "grid_argmin_mse": c_mse,
            "grid_step": step,
            "torch_fit_l1": tm.fit_constant(y, "l1"),
            "torch_fit_l2": tm.fit_constant(y, "l2"),
        }
    return res


def e6_noise_distribution(n=101, seeds=2000):
    """같은 '참값 0' 주변에 잡음을 얹은 표본에서 평균·중앙값으로 참값을 추정한다.

    평균은 제곱오차 합을, 중앙값은 절대오차 합을 최소화하는 상수다(E5).
    두 잡음 모두 분산 1로 맞췄다(라플라스 척도 b = 1/sqrt(2)).
    """
    rng = np.random.default_rng(0)
    out = {}
    for name in ("gaussian", "laplace"):
        means, meds = [], []
        for _ in range(seeds):
            z = rng.normal(0, 1, n) if name == "gaussian" else rng.laplace(0, 1 / np.sqrt(2), n)
            means.append(z.mean())
            meds.append(np.median(z))
        means, meds = np.array(means), np.array(meds)
        out[name] = {
            "sd_mean_estimate": float(means.std()),
            "sd_median_estimate": float(meds.std()),
            "var_ratio_median_over_mean": float(meds.var() / means.var()),
        }
    out["n"], out["seeds"] = n, seeds
    out["asymptotic_ratio"] = {"gaussian": float(np.pi / 2), "laplace": 0.5}
    return out


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
        "E1": e1_failure(),
        "E2": e2_boundary(),
        "E3": e3_stability(),
        "E4": e4_sample_size(),
        "E5": e5_best_constant(),
        "E6": e6_noise_distribution(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
