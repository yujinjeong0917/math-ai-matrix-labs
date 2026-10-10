"""모델 평가 8장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch08_classification/experiments.py` 로 실행한다.

E0 첫 화면: 검사 100건 중 양성 1건. '모두 음성' 모델과 '1건 잡고 4건 헛의심' 모델
E1 (a) 양성 1%, n = 10000, '항상 음성' 분류기
E2 (b) 양성 90%, n = 1000, '항상 양성' 분류기
E3 (c) 같은 점수 분포에서 양성 비율 50% -> 1%: ROC-AUC와 AP(PR-AUC). 음성을 그대로 복제하는 정확한 확인도 함께
E4 측정 1: ROC-AUC가 같은 두 분류기 A, B. 양성 비율 5단계 x 시드 100개에서 두 지표가 고르는 쪽
E5 측정 2: 부트스트랩 95% 구간 폭. 양성 10개 대 1000개(양성 비율 1%, 같은 임계값)
E6 대조: 원 논문·문서의 숫자, F = 1 - E, NumPy-PyTorch, 사다리꼴 AUC = 쌍 비교 AUC
E7 상자 글: 브라이어 점수. 정직한 확률이 기대 점수를 가장 낮추는지, 불균형에서 '늘 0'의 점수
결과는 results/ch08.json 에 저장하고, 웹 챕터는 이 파일의 숫자를 그대로 옮긴다.
"""

import json
import os
import platform
import sys
import time
from math import erf, sqrt
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_ch08_cases as cases  # noqa: E402
import eval_ch08_metrics as m  # noqa: E402
import eval_ch08_torch as tm  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent / "results" / "ch08.json"
PREVS = (0.5, 0.2, 0.1, 0.05, 0.01)


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def nan_to_none(d):
    if isinstance(d, dict):
        return {k: nan_to_none(v) for k, v in d.items()}
    if isinstance(d, list):
        return [nan_to_none(v) for v in d]
    if isinstance(d, float) and d != d:
        return None  # JSON에는 null. 본문에는 '정의 안 됨'
    return d


def scores_cm(cm):
    tp, fn, fp, tn = m.cells(cm)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, **m.all_scores(cm)}


def e0_first_screen():
    y, lazy, useful = cases.first_screen()
    return {"n": len(y), "positives": int(y.sum()),
            "lazy": scores_cm(m.confusion(y, lazy)), "useful": scores_cm(m.confusion(y, useful))}


def e1_imbalanced_constant():
    y, yhat = cases.case_imbalanced_constant(0.01, 10_000, seed=0)
    return {"n": len(y), "positives": int(y.sum()), **scores_cm(m.confusion(y, yhat))}


def e2_majority_positive():
    y, yhat = cases.case_majority_positive(0.9, 1000, seed=0)
    out = {"n": len(y), "positives": int(y.sum()), **scores_cm(m.confusion(y, yhat))}
    # 비교: 음성 100개 중 90개를 맞게 골라내는 분류기(양성은 그대로 다 맞힘)
    yhat2 = yhat.copy()
    neg = np.nonzero(y == 0)[0]
    yhat2[neg[:90]] = 0
    out["catches_90_negatives"] = scores_cm(m.confusion(y, yhat2))
    return out


def e3_prevalence_sweep(seeds=20):
    out = {"theory_auc": phi(1 / sqrt(2)), "n_pos": 1000, "seeds": seeds, "rows": []}
    for prev in PREVS:
        aucs, aps, traps = [], [], []
        for s in range(seeds):
            y, sc = cases.case_prevalence(prev, 1000, seed=1000 * s + int(prev * 1000))
            aucs.append(m.roc_auc_np(y, sc)); aps.append(m.average_precision_np(y, sc)); traps.append(m.pr_auc_trapezoid(y, sc))
        y0, s0 = cases.case_prevalence(prev, 1000, seed=int(prev * 1000))
        out["rows"].append({"prev": prev, "n_neg": int((y0 == 0).sum()), "auc_mean": float(np.mean(aucs)),
                            "auc_p5": float(np.percentile(aucs, 5)), "auc_p95": float(np.percentile(aucs, 95)),
                            "ap_mean": float(np.mean(aps)), "ap_p5": float(np.percentile(aps, 5)),
                            "ap_p95": float(np.percentile(aps, 95)), "pr_trapezoid_mean": float(np.mean(traps)),
                            "ap_random_baseline": prev})
    # 정확한 확인: 양성 비율 50% 데이터의 음성을 99번씩 복제 -> 양성 비율 1000/(1000+99000) ~ 1%
    y, sc = cases.case_prevalence(0.5, 1000, seed=0)
    neg = y == 0
    y_rep = np.r_[y[~neg], np.zeros(99 * neg.sum(), dtype=int)]
    s_rep = np.r_[sc[~neg], np.tile(sc[neg], 99)]
    out["replicate_negatives_x99"] = {
        "auc_before": m.roc_auc_np(y, sc), "auc_after": m.roc_auc_np(y_rep, s_rep),
        "ap_before": m.average_precision_np(y, sc), "ap_after": m.average_precision_np(y_rep, s_rep),
        "prev_after": float(y_rep.mean())}
    # 임계값 하나에서: 점수 1.5 이상을 양성이라 하면
    thr = []
    for prev in PREVS:
        y, sc = cases.case_prevalence(prev, 1000, seed=int(prev * 1000))
        cm = m.confusion(y, sc >= 1.5)
        thr.append({"prev": prev, **scores_cm(cm), "tpr": m.recall(cm),
                    "fpr": m.cells(cm)[2] / (m.cells(cm)[2] + m.cells(cm)[3])})
    out["threshold_1_5"] = thr
    return out


def e4_pair_flips(seeds=100, n_pos=200):
    out = {"pair": {k: {"mu_pos": v[0], "sd_pos": v[1]} for k, v in cases.PAIR.items()},
           "theory_auc": {k: phi(v[0] / sqrt(1 + v[1] ** 2)) for k, v in cases.PAIR.items()},
           "n_pos": n_pos, "seeds": seeds, "rows": []}
    for prev in PREVS:
        c = {"roc_picks_B": 0, "ap_picks_B": 0, "disagree": 0}
        auc_d, ap_d = [], []
        for s in range(seeds):
            y, sa, sb = cases.case_pair(prev, n_pos, seed=50_000 + 1000 * s + int(prev * 1000))
            aa, ab = m.roc_auc_np(y, sa), m.roc_auc_np(y, sb)
            pa, pb = m.average_precision_np(y, sa), m.average_precision_np(y, sb)
            c["roc_picks_B"] += ab > aa; c["ap_picks_B"] += pb > pa; c["disagree"] += (ab > aa) != (pb > pa)
            auc_d.append(ab - aa); ap_d.append(pb - pa)
        # 큰 표본 기준값(양성 20000개, 시드 하나)
        y, sa, sb = cases.case_pair(prev, 20_000, seed=7)
        ref = {"auc_A": m.roc_auc_np(y, sa), "auc_B": m.roc_auc_np(y, sb),
               "ap_A": m.average_precision_np(y, sa), "ap_B": m.average_precision_np(y, sb)}
        out["rows"].append({"prev": prev, **{k: int(v) for k, v in c.items()},
                            "auc_diff_mean": float(np.mean(auc_d)), "auc_diff_p5": float(np.percentile(auc_d, 5)),
                            "auc_diff_p95": float(np.percentile(auc_d, 95)),
                            "ap_diff_mean": float(np.mean(ap_d)), "ap_diff_p5": float(np.percentile(ap_d, 5)),
                            "ap_diff_p95": float(np.percentile(ap_d, 95)), "reference_n_pos_20000": ref})
    return out


def _bootstrap_cm(cm, reps, rng):
    """행을 복원 추출하는 부트스트랩은 혼동행렬 기준으로 '네 칸 비율로 n번 뽑는 다항분포'와 같다."""
    cm = np.asarray(cm); n = int(cm.sum())
    draws = rng.multinomial(n, cm.ravel() / n, size=reps)
    return [d.reshape(2, 2) for d in draws]


def e5_bootstrap_width(reps=1000):
    out = {"threshold": 1.5, "prev": 0.01, "reps": reps, "rows": []}
    for n_pos in (10, 1000):
        y, sc = cases.case_prevalence(0.01, n_pos, seed=3)
        cm = m.confusion(y, sc >= 1.5)
        rng = np.random.default_rng(11)
        boot = _bootstrap_cm(cm, reps, rng)
        row = {"n_pos": n_pos, "n": len(y), "point": scores_cm(cm), "ci": {}}
        for name, fn in (("recall", m.recall), ("precision", m.precision), ("f1", m.f1),
                         ("mcc", lambda c: m.mcc(c, "nan")), ("accuracy", m.accuracy)):
            vals = np.array([fn(c) for c in boot])
            ok = vals[~np.isnan(vals)]
            lo, hi = np.percentile(ok, [2.5, 97.5])
            row["ci"][name] = {"lo": float(lo), "hi": float(hi), "width": float(hi - lo), "undefined": int(np.isnan(vals).sum())}
        out["rows"].append(row)
    return out


def e6_checks():
    a1 = cases.chicco_use_case_a1()
    bal, imb = cases.saito_er_minus()
    yd, sd = cases.sklearn_ap_doc_example()
    grid = []
    for p in (0.1, 0.3, 0.5, 0.9):
        for r in (0.2, 0.5, 0.8):
            for beta in (0.5, 1.0, 2.0):
                fb = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
                grid.append(abs((1 - m.van_rijsbergen_e(p, r, 1 / (beta ** 2 + 1))) - fb))
    rng = np.random.default_rng(5)
    diffs = {"confusion": [], "f1": [], "f2": [], "mcc": [], "auc": [], "ap": [], "brier": [], "auc_trap_vs_pairs": []}
    for _ in range(20):
        n = int(rng.integers(20, 300))
        y = (rng.random(n) < rng.uniform(0.1, 0.6)).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.standard_normal(n) + y, 1)  # 반올림해서 동점을 일부러 만든다
        yhat = (s > 0.5).astype(int)
        cm = m.confusion(y, yhat)
        diffs["confusion"].append(float(np.abs(cm - tm.confusion(y, yhat).numpy()).max()))
        diffs["f1"].append(abs(m.f1(cm) - tm.f_beta(cm, 1.0)))
        diffs["f2"].append(abs(m.f_beta(cm, 2.0) - tm.f_beta(cm, 2.0)))
        diffs["mcc"].append(abs(m.mcc(cm) - tm.mcc(cm)))
        diffs["auc"].append(abs(m.roc_auc_np(y, s) - tm.roc_auc_pairs(y, s)))
        diffs["ap"].append(abs(m.average_precision_np(y, s) - tm.average_precision(y, s)))
        prob = 1 / (1 + np.exp(-s))
        diffs["brier"].append(abs(m.brier(y, prob) - tm.brier(y, prob)))
        diffs["auc_trap_vs_pairs"].append(abs(m.roc_auc_np(y, s) - m.auc_pairwise(y, s)))
    # AP와 사다리꼴 PR 넓이가 다른 예
    yx, sx = cases.case_prevalence(0.01, 50, seed=0)
    return {
        "chicco_a1": {"accuracy": m.accuracy(a1), "f1": m.f1(a1), "mcc": m.mcc(a1), "paper": {"accuracy": 0.90, "f1": 0.95, "mcc": -0.03}},
        "saito_er_minus": {"precision_balanced": m.precision(bal), "precision_imbalanced": m.precision(imb),
                           "tpr": m.recall(bal), "fpr": 160 / 1000},
        "sklearn_ap_doc": {"numpy": m.average_precision_np(yd, sd), "torch": tm.average_precision(yd, sd), "doc": 0.83,
                           "trapezoid": m.pr_auc_trapezoid(yd, sd)},
        "f_equals_1_minus_e_max_abs_diff": float(max(grid)),
        "numpy_torch_max_abs_diff": {k: float(max(v)) for k, v in diffs.items()}, "numpy_torch_cases": len(diffs["f1"]),
        "ap_vs_trapezoid_example": {"n_pos": int(yx.sum()), "ap": m.average_precision_np(yx, sx), "trapezoid": m.pr_auc_trapezoid(yx, sx)},
    }


def e7_brier():
    p_true = 0.01
    qs = np.round(np.arange(0, 0.051, 0.005), 3)
    exp_score = {f"{q:.3f}": float(p_true * (1 - q) ** 2 + (1 - p_true) * q ** 2) for q in qs}
    best = min(exp_score, key=exp_score.get)
    y, sc = cases.case_prevalence(0.01, 1000, seed=0)
    prev = y.mean()
    # 점수 분포를 아는 '정직한' 확률: P(양성 | 점수) (양성 N(1,1), 음성 N(0,1), 사전 비율 prev)
    lr = np.exp(sc - 0.5)
    post = prev * lr / (prev * lr + (1 - prev))
    return {"expected_by_forecast": exp_score, "argmin": best, "p_true": p_true,
            "data": {"n": len(y), "prev": float(prev), "always_0": m.brier(y, np.zeros(len(y))),
                     "always_prev": m.brier(y, np.full(len(y), prev)), "posterior": m.brier(y, post),
                     "always_0_5": m.brier(y, np.full(len(y), 0.5))}}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "E0": e0_first_screen(), "E1": e1_imbalanced_constant(), "E2": e2_majority_positive(),
        "E3": e3_prevalence_sweep(), "E4": e4_pair_flips(), "E5": e5_bootstrap_width(), "E6": e6_checks(),
        "E7": e7_brier(),
    }
    res["env"]["total_seconds"] = time.perf_counter() - t0
    res = nan_to_none(res)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
