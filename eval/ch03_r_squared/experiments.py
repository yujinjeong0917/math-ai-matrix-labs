"""모델 평가 3장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch03_r_squared/experiments.py` 로 실행한다.

E0 첫 화면 예시: 학습 x=0..4, y=x^2에 직선을 맞추고 x=5..8에서 잰다(손 계산 가능)
E1 실패 (a) Anscombe 네 쌍: 같은 R^2와 회귀선, 다른 잔차. 3번은 관측 3을 빼고 다시 맞춘다
E2 실패 (b) 오프셋·배율 모델: r^2 = 1인데 R^2는 음수 또는 0
E3 실패 (c) 외삽: [0,5]에서 y=x^2+잡음에 직선, [5,10]에서 테스트 (시드 0)
E4 비교 측정 1: 테스트 구간 위치를 바꾸며 시드 50개에서 학습 R^2 - 테스트 R^2
E5 비교 측정 2: 같은 테스트 세트 부트스트랩 1000번에서 두 모델의 R^2 순위와 RMSE 순위
E6 정의가 갈리는 곳: 원점을 지나는 직선에서 R^2 계산법 세 가지, NumPy-PyTorch 대조

결과는 results/ch03.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
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
import eval_ch03_cases as cases  # noqa: E402
import eval_ch03_metrics as m  # noqa: E402
import eval_ch03_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch03.json"


def _pct(a, q):
    return float(np.percentile(a, q))


def e0_first_screen():
    xtr, ytr, xte, yte = cases.first_screen()
    coef = m.fit_line(xtr, ytr)
    ptr, pte = m.predict_line(coef, xtr), m.predict_line(coef, xte)
    return {
        "x_train": xtr.tolist(), "y_train": ytr.tolist(), "coef_b0_b1": list(coef),
        "pred_train": ptr.tolist(), "resid_train": (ytr - ptr).tolist(),
        "sse_train": m.sse(ytr, ptr), "sst_train": m.sst(ytr), "r2_train": m.r2(ytr, ptr),
        "x_test": xte.tolist(), "y_test": yte.tolist(), "pred_test": pte.tolist(), "resid_test": (yte - pte).tolist(),
        "mean_y_test": float(yte.mean()),
        "sse_test": m.sse(yte, pte), "sst_test": m.sst(yte), "r2_test": m.r2(yte, pte),
        "r2corr_test": m.squared_corr(yte, pte),
    }


def _leverage(x):
    x = np.asarray(x, float)
    return 1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)


def e1_anscombe():
    out = {}
    for k, (x, y) in cases.anscombe().items():
        coef = m.fit_line(x, y)
        p = m.predict_line(coef, x)
        d = m.residual_diagnostics(x, y, p, bins=3)
        out[k] = {
            "mean_x": float(x.mean()), "mean_y": float(y.mean()),
            "b0": coef[0], "b1": coef[1],
            "reg_ss": float(np.sum((p - y.mean()) ** 2)), "resid_ss": m.sse(y, p), "sst": m.sst(y),
            "r2": m.r2(y, p), "r2corr": m.squared_corr(y, p),
            "torch_r2": tm.r2(torch.tensor(y), torch.tensor(p)),
            **d,
            "argmax_abs_resid_obs": int(np.argmax(np.abs(y - p))) + 1,
            "max_leverage": float(_leverage(x).max()),
            "argmax_leverage_obs": int(np.argmax(_leverage(x))) + 1,
            "resid_sorted_by_x": [float(v) for v in (y - p)[np.argsort(x, kind="stable")]],
        }
    x, y = cases.anscombe()["3"]
    keep = np.arange(11) != 2  # 관측 3 (x=13, y=12.74)
    c3 = m.fit_line(x[keep], y[keep])
    out["set3_without_obs3"] = {"b0": c3[0], "b1": c3[1], "r2": m.r2(y[keep], m.predict_line(c3, x[keep]))}
    x, y = cases.anscombe()["2"]
    c2 = m.fit_poly(x, y, 2)
    out["set2_quadratic"] = {"coef": c2.tolist(), "r2": m.r2(y, m.predict_poly(c2, x))}
    return out


def e2_offset(n=200, seed=0):
    y, p = cases.case_offset(n, 5.0, seed)
    sweep = {}
    for s in (0.5, 1.0, 2.0, 5.0):
        _, ps = cases.case_offset(n, s, seed)
        sweep[str(s)] = {"r2": m.r2(y, ps), "r2corr": m.squared_corr(y, ps), "formula": 1 - n * s**2 / m.sst(y)}
    ys, ps2 = cases.case_scaled(n, 2.0, seed)
    return {
        "n": n, "seed": seed, "shift": 5.0,
        "sst": m.sst(y), "sse": m.sse(y, p), "var_y": float(y.var()),
        "r2": m.r2(y, p), "r2corr": m.squared_corr(y, p),
        "resid_mean": float(np.mean(y - p)), "rmse": m.rmse(y, p),
        "formula_1_minus_n_shift2_over_sst": 1 - n * 25.0 / m.sst(y),
        "shift_sweep": sweep,
        "scaled2": {"r2": m.r2(ys, ps2), "r2corr": m.squared_corr(ys, ps2), "rmse": m.rmse(ys, ps2)},
    }


def e3_extrapolation(seed=0):
    xtr, ytr, xte, yte = cases.case_extrapolation(seed)
    lin = m.fit_line(xtr, ytr)
    quad = m.fit_poly(xtr, ytr, 2)
    ptr, pte = m.predict_line(lin, xtr), m.predict_line(lin, xte)
    qte = m.predict_poly(quad, xte)
    return {
        "seed": seed, "n_train": len(xtr), "n_test": len(xte), "noise_sd": 1.0,
        "line_b0_b1": list(lin),
        "train": {"r2": m.r2(ytr, ptr), "r2corr": m.squared_corr(ytr, ptr), **m.residual_diagnostics(xtr, ytr, ptr, bins=5)},
        "test": {"r2": m.r2(yte, pte), "r2corr": m.squared_corr(yte, pte), "rmse": m.rmse(yte, pte),
                  "sst_over_n": m.sst(yte) / len(yte), **m.residual_diagnostics(xte, yte, pte, bins=5)},
        "quadratic_test": {"r2": m.r2(yte, qte), "rmse": m.rmse(yte, qte)},
    }


def e4_window_sweep(seeds=50):
    windows = {"[0,5] 같은 범위": (0.0, 5.0), "[0,1] 같은 범위, 좁은 왼쪽 끝": (0.0, 1.0), "[4,5] 같은 범위, 좁은 오른쪽 끝": (4.0, 5.0), "[2.5,7.5] 반쯤 밖": (2.5, 7.5),
               "[5,10] 밖": (5.0, 10.0), "[7.5,12.5] 더 밖": (7.5, 12.5)}
    out = {}
    for name, w in windows.items():
        tr, te, gap, tec, qte = [], [], [], [], []
        for s in range(seeds):
            xtr, ytr, xte, yte = cases.case_extrapolation(seed=1000 + s, test=w)
            lin = m.fit_line(xtr, ytr)
            a = m.r2(ytr, m.predict_line(lin, xtr))
            b = m.r2(yte, m.predict_line(lin, xte))
            tr.append(a); te.append(b); gap.append(a - b)
            tec.append(m.squared_corr(yte, m.predict_line(lin, xte)))
            qte.append(m.r2(yte, m.predict_poly(m.fit_poly(xtr, ytr, 2), xte)))
        tr, te, gap, tec, qte = map(np.array, (tr, te, gap, tec, qte))
        out[name] = {
            "window": list(w),
            "train_r2_mean": float(tr.mean()), "train_r2_p5_p95": [_pct(tr, 5), _pct(tr, 95)],
            "test_r2_median": float(np.median(te)), "test_r2_p5_p95": [_pct(te, 5), _pct(te, 95)],
            "gap_median": float(np.median(gap)), "gap_p5_p95": [_pct(gap, 5), _pct(gap, 95)],
            "test_r2corr_median": float(np.median(tec)),
            "seeds_test_r2_negative": int(np.sum(te < 0)),
            "quadratic_test_r2_median": float(np.median(qte)),
        }
    return {"seeds": seeds, "n_train": 50, "n_test": 50, "train_window": [0.0, 5.0], "by_window": out}


def e5_bootstrap_rank(B=1000, n_test=50):
    rng = np.random.default_rng(7)
    xte, yte = cases.quad_sample(0.0, 5.0, n_test, rng)
    models = {}
    for name, s in (("A", 101), ("B", 202)):
        r = np.random.default_rng(s)
        xtr, ytr = cases.quad_sample(0.0, 5.0, 12, r)
        models[name] = m.predict_poly(m.fit_poly(xtr, ytr, 2), xte)
    full = {k: {"r2": m.r2(yte, p), "rmse": m.rmse(yte, p)} for k, p in models.items()}
    br = np.random.default_rng(99)
    agree = b_rmse = b_r2 = 0
    r2a, r2b, ra, rb = [], [], [], []
    for _ in range(B):
        idx = br.integers(0, n_test, n_test)
        y = yte[idx]
        sa, sb = m.r2(y, models["A"][idx]), m.r2(y, models["B"][idx])
        ea, eb = m.rmse(y, models["A"][idx]), m.rmse(y, models["B"][idx])
        r2a.append(sa); r2b.append(sb); ra.append(ea); rb.append(eb)
        b_r2 += sb > sa
        b_rmse += eb < ea
        agree += (sb > sa) == (eb < ea)
    r2a, r2b, ra, rb = map(np.array, (r2a, r2b, ra, rb))
    return {
        "B": B, "n_test": n_test, "n_train_each": 12, "model": "2차 다항식, 학습 표본만 다름",
        "full_test": full,
        "boot_B_better_by_r2": int(b_r2), "boot_B_better_by_rmse": int(b_rmse), "boot_rank_agree": int(agree),
        "r2_A_p5_p95": [_pct(r2a, 5), _pct(r2a, 95)], "r2_B_p5_p95": [_pct(r2b, 5), _pct(r2b, 95)],
        "rmse_A_p5_p95": [_pct(ra, 5), _pct(ra, 95)], "rmse_B_p5_p95": [_pct(rb, 5), _pct(rb, 95)],
    }


def e6_definitions():
    # 개념 카드 eval-r-squared의 반례: x = 1, 2, 3, y = 10, 11, 10을 원점을 지나는 직선으로 맞춘다.
    x, y = np.array([1.0, 2, 3]), np.array([10.0, 11, 10])
    c = m.fit_line(x, y, intercept=False)
    p = m.predict_line(c, x)
    xi, yi = np.array([1.0, 2, 3]), np.array([10.0, 11, 10])
    ci = m.fit_line(xi, yi)
    pi = m.predict_line(ci, xi)
    rng = np.random.default_rng(3)
    xa, ya = rng.normal(size=300), rng.normal(size=300)
    return {
        "no_intercept": {
            "x": x.tolist(), "y": y.tolist(), "slope": c[1], "pred": p.tolist(), "sse": m.sse(y, p), "sst": m.sst(y),
            "r2_centered": m.r2(y, p), "r2corr": m.squared_corr(y, p),
            "r2_uncentered": 1 - m.sse(y, p) / float(np.sum(y**2)),
        },
        "with_intercept": {"b0_b1": list(ci), "r2_centered": m.r2(yi, pi)},  # 기울기 0, 평균 수평선이라 r^2는 정의되지 않는다
        "torch_vs_numpy": {
            "r2_abs_diff": abs(m.r2(ya, xa + ya) - tm.r2(torch.tensor(ya), torch.tensor(xa + ya))),
            "r2corr_abs_diff": abs(m.squared_corr(ya, xa + ya) - tm.squared_corr(torch.tensor(ya), torch.tensor(xa + ya))),
        },
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
        "E1": e1_anscombe(),
        "E2": e2_offset(),
        "E3": e3_extrapolation(),
        "E4": e4_window_sweep(),
        "E5": e5_bootstrap_rank(),
        "E6": e6_definitions(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
