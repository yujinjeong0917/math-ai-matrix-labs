"""강화학습 7장 실험. `cd math-ai-matrix-labs && uv run python rl/ch07_nstep_td_lambda/experiments.py`

E0 손계산: 칸 A~E, C B C D E -> 오른쪽 끝(+1). 모든 값 0, alpha 0.5, gamma 1.
   1스텝·2스텝·끝까지(n스텝 TD)와 λ 0.5 흔적(누적·대체)이 한 판 뒤 무엇을 고치나
E1 실패 최소 예제: 13칸 무작위 보행(왼쪽 끝 -1, 오른쪽 끝 +1, V0 = 0, gamma 1), 시드 0~99, 판 30개씩
   n 1, 2, 4, 8, 16, 32, 끝까지 x alpha 0.01~0.04(0.01 간격)와 0.05 ~ 1.0(0.05 간격) 24개. 10판 뒤 RMSE(13칸)의 시드 100개 평균.
   n마다 가장 좋은 alpha를 고르고, 그 alpha를 새 시드 100~199에서 다시 돌려 95% 구간을 낸다.
E2 TD(λ): λ 0, 0.3, 0.6, 0.8, 0.9, 0.95, 1 x 같은 alpha 격자. 온라인 누적 흔적, 온라인 대체 흔적,
   판마다 합산(= 판 동안 V를 고정한 누적 흔적, Sutton 1988 식 (1)), 오프라인 λ-리턴(판 끝에 차례로, Sutton·Barto 식 12.4). 발산 기준(실험 전에 정함): 판이 끝났을 때
   |V|가 10을 넘거나 유한하지 않으면 발산으로 보고 그 행을 멈춘다. "처음보다 나빠짐" = 발산 또는 10판 뒤 RMSE > 처음 RMSE.
E3 같은 판에서 맞대기: n스텝·TD(λ) 각자 가장 좋은 설정을 새 시드 100~199에서 시드별로 비교
E4 6장 외길(결정 칸 9개, gamma 0.9, alpha 1): 모든 칸이 정답이 되기까지의 판 수
E5 계산량: 걸음당 시간(n스텝 버퍼 vs 흔적 갱신), 상태 수에 따른 흔적 갱신 시간
E6 흔적 로그: 시드 0 첫 판, λ 0.8, 누적·대체 흔적을 logs/ch07_traces.jsonl에 남긴다

결과는 results/ch07.json.
"""

import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch07_np as T  # noqa: E402
import rl_ch07_walk as W  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch07.json"
LOG = HERE / "logs" / "ch07_traces.jsonl"
NS = W.N_WALK + 2
ALPHAS = [0.01, 0.02, 0.03, 0.04] + [round(0.05 * i, 2) for i in range(1, 21)]
NS_LIST = [1, 2, 4, 8, 16, 32, None]
LAMS = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0]
K = 30            # 판 수(기록은 3, 10, 30판에서 읽는다)
SEEDS_SEL = list(range(0, 100))
SEEDS_EVAL = list(range(100, 200))
V0_RMSE = W.rmse(np.zeros(NS))


def r6(x):
    return None if x is None or not math.isfinite(float(x)) else float(round(float(x), 6))


def nkey(n):
    return "inf" if n is None else str(n)


def ci(a):
    """6장 experiments.ci에서 복사."""
    a = np.asarray(a, dtype=float)
    m, sd = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
    h = 1.96 * sd / np.sqrt(len(a))
    return {"mean": r6(m), "lo": r6(m - h), "hi": r6(m + h), "sd": r6(sd), "median": r6(np.median(a)),
            "min": r6(a.min()), "max": r6(a.max())}


def rmse_rows(V):
    """V: [rows, NS] -> 행별 RMSE (NaN 그대로)."""
    truth = W.walk_true_values()[1:W.N_WALK + 1]
    with np.errstate(invalid="ignore", over="ignore"):
        return np.sqrt(np.mean((V[:, 1:W.N_WALK + 1] - truth) ** 2, axis=1))


EPS_SEL = [W.walk_episodes(s, K) for s in SEEDS_SEL]
EPS_EVAL = [W.walk_episodes(s, K) for s in SEEDS_EVAL]


# ---------------- E0 ----------------

def e0_hand():
    ep = W.hand_episode()
    names = W.HAND_NAMES
    out = {"path": "C B C D E -> 오른쪽 끝(+1)", "alpha": 0.5, "gamma": 1.0, "v0": 0.0, "values_after": {}}
    for label, n in (("1step", 1), ("2step", 2), ("4step", 4), ("end", None)):
        V = T.n_step_td([ep], 7, n, 0.5)
        out["values_after"][label] = {names[i]: r6(V[i]) for i in range(1, 6)}
    for tr in ("accumulating", "replacing"):
        log = []
        V, _ = T.td_lambda([ep], 7, 0.5, 0.5, trace=tr, trace_log=log)
        out["values_after"][f"lambda0.5_{tr}"] = {names[i]: r6(V[i]) for i in range(1, 6)}
        out[f"trace_last_step_{tr}"] = {names[i]: r6(log[-1][5][i]) for i in range(1, 6)}
    out["td_error_last_step"] = 1.0
    out["note"] = "보상이 마지막 +1뿐이고 처음 값이 모두 0이라, TD 오차가 0이 아닌 건 마지막 걸음 하나다."
    return out


# ---------------- E1 / E2 공통 ----------------

def summarize_grid(hist, n_seeds, diverged=None):
    """hist [K, n_seeds*A, NS] -> alpha별 평균 RMSE(3, 10, 30판 뒤와 1~10판 평균)."""
    A = len(ALPHAS)
    res = {}
    for kk in (3, 10, 30):
        r = rmse_rows(hist[kk - 1]).reshape(n_seeds, A)
        res[f"after{kk}"] = r
    curve = np.stack([rmse_rows(hist[k]).reshape(n_seeds, A) for k in range(10)])   # [10, seeds, A]
    res["avg1to10"] = curve.mean(axis=0)
    if diverged is not None:
        res["diverged"] = diverged.reshape(n_seeds, A)
    return res


def best_alpha(mat):
    """mat [seeds, A]: 그 시점의 RMSE가 유한하지 않은 시드가 있는 alpha만 뺀다.
    (그 시점 뒤에 발산하는 alpha는 빼지 않는다. 발산 여부는 grid의 diverged_seeds로 따로 본다.)"""
    m = np.array([np.mean(mat[:, a]) if np.all(np.isfinite(mat[:, a])) else np.inf for a in range(mat.shape[1])])
    a = int(np.argmin(m))
    return a, float(m[a])


def e1_nstep():
    out = {"n_list": [nkey(n) for n in NS_LIST], "alphas": ALPHAS, "rmse_v0": r6(V0_RMSE), "grid": {}, "best": {}}
    for n in NS_LIST:
        hist = T.n_step_td_batch(EPS_SEL, NS, n, ALPHAS)
        s = summarize_grid(hist, len(SEEDS_SEL))
        out["grid"][nkey(n)] = {"after10_mean": [r6(x) for x in s["after10"].mean(0)],
                                "avg1to10_mean": [r6(x) for x in s["avg1to10"].mean(0)]}
        b = {}
        for kk in ("after3", "after10", "after30", "avg1to10"):
            a, m = best_alpha(s[kk])
            b[kk] = {"alpha": ALPHAS[a], "rmse": r6(m)}
        out["best"][nkey(n)] = b
    # 가장 좋은 n (시점별)
    for kk in ("after3", "after10", "after30", "avg1to10"):
        kbest = min(out["best"], key=lambda k: out["best"][k][kk]["rmse"])
        out[f"best_n_{kk}"] = {"n": kbest, **out["best"][kbest][kk]}
    return out


VARIANTS = {"online_acc": dict(trace="accumulating", online=True),
            "online_rep": dict(trace="replacing", online=True),
            "offline_summed": dict(trace="accumulating", online=False),
            "offline_lambda_return": None}


def run_lambda(vname, eps, lam, alphas):
    kw = VARIANTS[vname]
    if kw is None:
        return T.offline_lambda_return_batch(eps, NS, lam, alphas)
    return T.td_lambda_batch(eps, NS, lam, alphas, **kw)


def e2_lambda():
    out = {"lams": LAMS, "alphas": ALPHAS, "blowup": 10.0, "variants": {}}
    for vname in VARIANTS:
        vo = {"grid": {}, "best": {}, "unstable": {}}
        for lam in LAMS:
            hist, div, div_at = run_lambda(vname, EPS_SEL, lam, ALPHAS)
            s = summarize_grid(hist, len(SEEDS_SEL), div)
            d = s["diverged"]
            after10 = s["after10"]
            worse = d | ~np.isfinite(after10) | (np.nan_to_num(after10, nan=np.inf) > V0_RMSE)
            vo["grid"][str(lam)] = {
                "after10_mean": [r6(np.mean(after10[:, a])) if np.all(np.isfinite(after10[:, a])) else None
                                 for a in range(len(ALPHAS))],
                "diverged_seeds": [int(d[:, a].sum()) for a in range(len(ALPHAS))],
                "worse_than_start_seeds": [int(worse[:, a].sum()) for a in range(len(ALPHAS))],
            }
            first_div = [ALPHAS[a] for a in range(len(ALPHAS)) if d[:, a].any()]
            first_worse = [ALPHAS[a] for a in range(len(ALPHAS)) if worse[:, a].any()]
            vo["unstable"][str(lam)] = {"smallest_alpha_any_diverged": first_div[0] if first_div else None,
                                        "smallest_alpha_any_worse": first_worse[0] if first_worse else None,
                                        "alphas_with_divergence": len(first_div)}
            b = {}
            for kk in ("after3", "after10", "after30", "avg1to10"):
                a, m = best_alpha(s[kk])
                b[kk] = {"alpha": ALPHAS[a], "rmse": r6(m)}
            vo["best"][str(lam)] = b
        for kk in ("after3", "after10", "after30", "avg1to10"):
            lb = min(vo["best"], key=lambda k: vo["best"][k][kk]["rmse"])
            vo[f"best_lam_{kk}"] = {"lam": float(lb), **vo["best"][lb][kk]}
        out["variants"][vname] = vo
    return out


# ---------------- E3 ----------------

def eval_nstep(n, alpha, eps):
    hist = T.n_step_td_batch(eps, NS, n, [alpha])
    return rmse_rows(hist[9])


def eval_lambda(vname, lam, alpha, eps):
    hist, div, _ = run_lambda(vname, eps, lam, [alpha])
    return rmse_rows(hist[9]), div


def e3_compare(e1, e2):
    out = {"seeds": "100~199 (alpha·n·λ는 시드 0~99에서 고름)", "metric": "10판 뒤 RMSE", "rows": {}}
    per_seed = {}
    for n in NS_LIST:
        a = e1["best"][nkey(n)]["after10"]["alpha"]
        r = eval_nstep(n, a, EPS_EVAL)
        per_seed[f"n={nkey(n)}"] = r
        out["rows"][f"n={nkey(n)}"] = {"alpha": a, **ci(r)}
    for vname in VARIANTS:
        for lam in LAMS:
            a = e2["variants"][vname]["best"][str(lam)]["after10"]["alpha"]
            r, div = eval_lambda(vname, lam, a, EPS_EVAL)
            per_seed[f"{vname} lam={lam}"] = r
            out["rows"][f"{vname} lam={lam}"] = {"alpha": a, "diverged": int(div.sum()), **ci(r)}
    nb = e1["best_n_after10"]["n"]
    lb = e2["variants"]["online_acc"]["best_lam_after10"]["lam"]
    A, B = per_seed[f"n={nb}"], per_seed[f"online_acc lam={lb}"]
    one, inf = per_seed["n=1"], per_seed["n=inf"]
    out["paired"] = {
        "best_n": nb, "best_lam_online_acc": lb,
        "best_n_minus_best_lam": ci(A - B),
        "seeds_best_n_smaller_than_best_lam": int((A < B).sum()),
        "seeds_best_n_smaller_than_n1": int((A < one).sum()),
        "seeds_best_n_smaller_than_ninf": int((A < inf).sum()),
        "best_n_minus_n1": ci(A - one), "best_n_minus_ninf": ci(A - inf),
        "seeds_best_lam_smaller_than_lam0": int((B < per_seed["online_acc lam=0.0"]).sum()),
        "seeds_best_lam_smaller_than_lam1": int((B < per_seed["online_acc lam=1.0"]).sum()),
    }
    # 3판·30판 시점도 같은 절차로: 시드 0~99에서 고른 alpha를 새 시드에서
    hz = {}
    for kk in (3, 30):
        row = {"nstep": {}, "online_acc": {}}
        for n in NS_LIST:
            a = e1["best"][nkey(n)][f"after{kk}"]["alpha"]
            hist = T.n_step_td_batch(EPS_EVAL, NS, n, [a])
            row["nstep"][nkey(n)] = {"alpha": a, **ci(rmse_rows(hist[kk - 1]))}
        for lam in LAMS:
            a = e2["variants"]["online_acc"]["best"][str(lam)][f"after{kk}"]["alpha"]
            hist, div, _ = run_lambda("online_acc", EPS_EVAL, lam, [a])
            row["online_acc"][str(lam)] = {"alpha": a, "diverged": int(div.sum()), **ci(rmse_rows(hist[kk - 1]))}
        row["best_n"] = min(row["nstep"], key=lambda k: row["nstep"][k]["mean"])
        row["best_lam"] = min(row["online_acc"], key=lambda k: row["online_acc"][k]["mean"])
        hz[f"after{kk}"] = row
    out["fresh_by_horizon"] = hz
    return out


# ---------------- E4 ----------------

def e4_chain():
    ep = W.chain_episode(10)
    truth = W.chain_true_values(10, 0.9)
    out = {"cells": 9, "gamma": 0.9, "alpha": 1.0, "episodes_to_exact": {}, "td_lambda_after_1": {}}
    for n in (1, 2, 3, 4, 8, None):
        rec = []
        T.n_step_td([ep] * 12, 10, n, 1.0, gamma=0.9, record=rec)
        out["episodes_to_exact"][nkey(n)] = next(k + 1 for k, V in enumerate(rec) if np.allclose(V, truth, atol=1e-12))
    for lam in (0.0, 0.5, 0.9, 1.0):
        V, _ = T.td_lambda([ep], 10, lam, 1.0, gamma=0.9, trace="accumulating")
        out["td_lambda_after_1"][str(lam)] = {"max_abs_error": r6(np.abs(V - truth).max()),
                                              "cells_nonzero": int((np.abs(V[:9]) > 0).sum()),
                                              "V0": r6(V[0]), "V0_true": r6(truth[0])}
    return out


# ---------------- E5 ----------------

def e5_compute():
    rng = np.random.default_rng(0)
    steps = 20_000
    out = {"steps": steps, "note": "파이썬 반복, 스레드 2개. 실행마다 조금씩 달라진다."}
    # 흔적 갱신: 상태 수에 따라
    tr = {}
    for nS in (15, 1_001, 100_001):
        s = rng.integers(0, nS, size=steps)
        V = np.zeros(nS)
        e = np.zeros(nS)
        t0 = time.perf_counter()
        for t in range(steps - 1):
            e *= 0.8
            e[s[t]] += 1.0
            delta = 0.0 + V[s[t + 1]] - V[s[t]]
            V += 0.1 * delta * e
        tr[str(nS)] = r6((time.perf_counter() - t0) / (steps - 1) * 1e6)
    out["trace_update_us_per_step"] = tr
    # n스텝: 길이 n 버퍼 하나로 한 걸음마다 한 칸 갱신(상태 수와 무관)
    ns = {}
    eps = W.walk_episodes(0, 400)
    n_tr = sum(len(e) for e in eps)
    for n in (1, 4, 32):
        t0 = time.perf_counter()
        T.n_step_td(eps, NS, n, 0.1)
        ns[str(n)] = r6((time.perf_counter() - t0) / n_tr * 1e6)
    out["n_step_us_per_step_reference_impl"] = ns
    out["n_step_transitions"] = n_tr
    out["memory"] = {"n_step": "최근 n개 (상태, 보상)", "td_lambda": "상태 수만큼의 흔적 배열 하나",
                     "offline_lambda_return": "판 하나 전체"}
    lens = [len(e) for eps_s in EPS_SEL for e in eps_s[:10]]
    out["walk_episode_length_first10"] = {"mean": r6(np.mean(lens)), "max": int(np.max(lens)),
                                         "min": int(np.min(lens))}
    return out


# ---------------- E6 ----------------

def e6_log():
    ep = EPS_SEL[0][0]
    logs = {}
    for tr in ("accumulating", "replacing"):
        lg = []
        T.td_lambda([ep], NS, 0.8, 0.1, trace=tr, trace_log=lg)
        logs[tr] = lg
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        for (t, s, r, s2, done, ea), (_, _, _, _, _, er) in zip(logs["accumulating"], logs["replacing"]):
            f.write(json.dumps({"seed": 0, "episode": 0, "t": t, "s": int(s), "a": "right" if s2 > s else "left",
                                "r": r, "s_next": int(s2), "done": bool(done),
                                "info": {"lam": 0.8, "trace_accumulating": [round(float(x), 4) for x in ea[1:NS - 1]],
                                         "trace_replacing": [round(float(x), 4) for x in er[1:NS - 1]]}},
                               ensure_ascii=False) + "\n")
    last_a, last_r = logs["accumulating"][-1][5], logs["replacing"][-1][5]
    return {"episode_length": len(ep), "max_trace_accumulating": r6(last_a.max()),
            "max_trace_accumulating_any_step": r6(max(x[5].max() for x in logs["accumulating"])),
            "max_trace_replacing_any_step": r6(max(x[5].max() for x in logs["replacing"])),
            "visits_per_state": {str(i): sum(1 for x in ep if x[0] == i) for i in range(1, W.N_WALK + 1)}}


def main():
    t0 = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "machine": platform.machine(), "system": platform.system(), "threads": 2},
           "walk": {"n_states": W.N_WALK, "start": (W.N_WALK + 1) // 2, "rewards": "왼쪽 끝 -1, 오른쪽 끝 +1",
                    "true_values": [r6(x) for x in W.walk_true_values()[1:W.N_WALK + 1]], "v0": 0.0,
                    "rmse_v0": r6(V0_RMSE)}}
    res["E0_hand"] = e0_hand()
    res["E1_nstep"] = e1_nstep()
    print("E1 done", round(time.perf_counter() - t0, 1))
    res["E2_lambda"] = e2_lambda()
    print("E2 done", round(time.perf_counter() - t0, 1))
    res["E3_compare"] = e3_compare(res["E1_nstep"], res["E2_lambda"])
    res["E4_chain"] = e4_chain()
    res["E5_compute"] = e5_compute()
    res["E6_log"] = e6_log()
    res["runtime_s"] = r6(time.perf_counter() - t0)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, allow_nan=False))
    print("saved", OUT, res["runtime_s"])


if __name__ == "__main__":
    main()
