"""강화학습 8장 실험. `cd math-ai-matrix-labs && uv run python rl/ch08_function_approx/experiments.py`

문제는 모두 Baird(1995) 6상태 별 문제. 모든 보상이 0이라 참값은 모든 상태에서 0이다.
값 오차 = 상태 값 6개의 최대 절댓값. 처음 가중치(기본) w = (1,1,1,1,1,1,10): V(1..5) = 3, V(6) = 12.

E0 손계산: gamma 0.99, alpha 0.01, 한 스윕(전이 6개의 고칠 양을 모아 한 번에). 선형(가중치 7개)과 표(상태 6칸) 비교.
E1 실패 최소 예제: 같은 설정으로 10,000 스윕. 스윕마다 w0, w6의 부호를 세고, 체크포인트에서 노름·값 오차를 남긴다.
   같은 설정을 gamma 0.9로도 돌린다(설계도의 원래 설정).
E2 고윳값: 기대 갱신 행렬 A = Phi^T D (I - gamma P) Phi (D = 1/6씩)의 고윳값.
   가장 작은 실수부가 0이 되는 gamma를 이분법으로 찾고, 트레이스로 구한 식 7/6 - 5 gamma / 4와 대조한다.
E3 죽음의 삼각형 한 다리씩 빼기: gamma 0.99, alpha 0.01, 10,000 스윕(궤적은 60,000걸음), 처음 가중치 시드 10개
   (w ~ U(0.5, 1.5), 시드 0~9). 발산 기준(실험 전에 정함): 끝났을 때 값 오차가 처음 값 오차보다 크거나
   가중치 노름이 1e8을 넘거나 유한하지 않으면 발산. 수렴 기준: 값 오차 < 0.01.
E4 alpha x gamma 지도: alpha {0.001, 0.01, 0.1} x gamma {0.5, 0.9, 0.95, 0.99}, 기본 처음 가중치, 10,000 스윕.
   이론 판정(반복 행렬 I - 6 alpha A의 스펙트럼 반경 > 1이면 발산)과 실측을 나란히 둔다.
E5 잔차 경사의 값: Baird 그림 2의 복도(6칸 표)와 별 문제에서 직접 TD와 잔차 경사가 값 오차 0.01 아래로
   내려가는 스윕 수. alpha는 격자에서 시드 10개 중앙값이 가장 작은 값을 고른다. 잔차 알고리즘의 섞는 비율
   phi의 안정 경계도 고윳값으로 찾는다.
E6 계산량: 스윕 하나·궤적 한 걸음의 시간, 고윳값 계산 시간(d = 7, 70, 700).
E7 PyTorch 대조: detach 있음(반경사)·없음(잔차 경사) 300 스윕을 NumPy와 비교.
E8 로그: 오프폴리시 궤적(시드 0, gamma 0.99) 처음 600걸음을 logs/ch08_offpolicy.jsonl에.

결과는 results/ch08.json.
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
import rl_ch08_np as F  # noqa: E402
import rl_ch08_torch as FT  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch08.json"
LOG = HERE / "logs" / "ch08_offpolicy.jsonl"

PHI, NXT, R = F.baird_star()
P_T = F.transition_matrix(NXT)          # 목표 정책: 모두 상태 6으로
P_B = F.behavior_chain()                # 행동 정책: 모든 칸 1/6
D_U = np.full(6, 1 / 6)                 # 균등 갱신 분포
W_BAIRD = np.array([1, 1, 1, 1, 1, 1, 10.0])
SEEDS = list(range(10))
SWEEPS = 10_000


def r6(x):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return float(f"{x:.6g}")


def init_w(seed):
    return np.random.default_rng(seed).uniform(0.5, 1.5, size=7)


def vmax(w, Phi=PHI):
    return F.value_norm(w, Phi)


# ---------------- E0 ----------------

def e0_hand():
    g, a = 0.99, 0.01
    w = W_BAIRD.copy()
    v = PHI @ w
    delta = R + g * v[NXT] - v
    w1, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, w, a, g, 1)
    v_tab = v.copy()
    v_tab1, _, _ = F.semi_gradient_td_sweeps(np.eye(6), NXT, R, v_tab, a, g, 1)
    return {
        "gamma": g, "alpha": a, "w_before": W_BAIRD.tolist(),
        "values_before": [r6(x) for x in v],
        "targets": [r6(x) for x in R + g * v[NXT]],
        "td_errors": [r6(x) for x in delta],
        "dw": [r6(x) for x in (w1 - w)],
        "dw0_parts": {"from_states_1_5": r6(a * 5 * delta[0] * 1.0), "from_state_6": r6(a * delta[5] * 2.0)},
        "w_after": [r6(x) for x in w1],
        "values_after_linear": [r6(x) for x in PHI @ w1],
        "values_after_table": [r6(x) for x in v_tab1],
        "note": "V(6)의 TD 오차는 음수(-0.12)라 혼자라면 내려가야 하지만, w0을 1~5번 상태가 다섯 번 올리고 6번이 한 번(특징 2배) 내려 V(6)이 올라간다.",
    }


# ---------------- E1 ----------------

def e1_fail(gamma):
    a = 0.01
    w = W_BAIRD.copy()
    checkpoints = {0, 10, 100, 1000, 2000, 5000, 10000}
    rows = []
    sign_changes = {"w0": 0, "w6": 0}
    prev = np.sign(w[[0, 6]])
    first_over = {"1e3": None, "1e6": None}
    for k in range(0, SWEEPS + 1):
        if k > 0:
            w, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, w, a, gamma, 1, blowup=1e300)
            s = np.sign(w[[0, 6]])
            sign_changes["w0"] += int(s[0] != prev[0])
            sign_changes["w6"] += int(s[1] != prev[1])
            prev = s
        n = np.linalg.norm(w)
        for key, thr in (("1e3", 1e3), ("1e6", 1e6)):
            if first_over[key] is None and n > thr:
                first_over[key] = k
        if k in checkpoints:
            rows.append({"sweep": k, "w_norm": r6(n), "value_max": r6(vmax(w)), "w0": r6(w[0]), "w6": r6(w[6]),
                         "V6": r6(PHI[5] @ w), "bellman_error": r6(F.bellman_error(w, PHI, P_T, R, gamma, D_U))})
    w_inc, _, div_inc = F.semi_gradient_td_sweeps(PHI, NXT, R, W_BAIRD, a, gamma, SWEEPS, epoch=False, blowup=1e300)
    v_tab, _, _ = F.semi_gradient_td_sweeps(np.eye(6), NXT, R, PHI @ W_BAIRD, a, gamma, SWEEPS)
    return {"gamma": gamma, "alpha": a, "sweeps": SWEEPS, "checkpoints": rows,
            "sign_changes": sign_changes, "first_sweep_norm_over": first_over,
            "incremental_final": {"w_norm": r6(np.linalg.norm(w_inc)), "value_max": r6(vmax(w_inc))},
            "table_final_value_max": r6(np.max(np.abs(v_tab))),
            "final_w": [r6(x) for x in w]}


# ---------------- E2 ----------------

def min_real(gamma, P=P_T, D=D_U, Phi=PHI):
    return float(np.linalg.eigvals(F.expected_update_matrix(Phi, P, D, gamma)).real.min())


def e2_eigen():
    rows = []
    for g in [0.5, 0.8, 0.9, 0.93, 14 / 15, 0.94, 0.95, 0.99]:
        ev = np.linalg.eigvals(F.expected_update_matrix(PHI, P_T, D_U, g))
        ev = ev[np.argsort(ev.real)]
        cpx = ev[np.abs(ev.imag) > 1e-9]
        rows.append({"gamma": r6(g), "min_real": r6(ev.real.min()),
                     "complex_pair": [r6(cpx[0].real), r6(abs(cpx[0].imag))] if len(cpx) else None,
                     "formula_7_6_minus_5g_4": r6(7 / 6 - 5 * g / 4),
                     "eigs_real_sorted": [r6(x) for x in ev.real]})
    lo, hi = 0.5, 0.999
    for _ in range(80):
        m = (lo + hi) / 2
        if min_real(m) < -1e-13:
            hi = m
        else:
            lo = m
    trace_check = {g: r6(np.trace(F.expected_update_matrix(PHI, P_T, D_U, g))) for g in (0.5, 0.9, 0.99)}
    onp = {}
    for g in (0.5, 0.9, 0.99):
        A_on = F.expected_update_matrix(PHI, P_B, F.stationary_distribution(P_B), g)
        sym = (A_on + A_on.T) / 2
        onp[str(g)] = {"min_real": r6(np.linalg.eigvals(A_on).real.min()),
                       "min_eig_symmetric_part": r6(np.linalg.eigvalsh(sym).min())}
    table = {str(g): r6(np.linalg.eigvals(F.expected_update_matrix(np.eye(6), P_T, D_U, g)).real.min())
             for g in (0.5, 0.9, 0.99)}
    return {"rows": rows, "threshold_gamma": r6(lo), "threshold_exact": "14/15", "trace": trace_check,
            "rank_phi": int(np.linalg.matrix_rank(PHI)), "n_weights": 7,
            "behavior_stationary": [r6(x) for x in F.stationary_distribution(P_B)],
            "on_policy_behavior_chain": onp, "table_min_real": table}


# ---------------- E3 ----------------

def time_to(hist, thr=1e-2, Phi=PHI):
    for k, w in hist:
        if vmax(w, Phi) < thr:
            return k
    return None


def e3_ablation():
    g, a = 0.99, 0.01
    conds = {}

    def summarize(name, finals, start_vals, conv_times, extra=None):
        div = sum(1 for f, s0 in zip(finals, start_vals) if (not math.isfinite(f)) or f > s0)
        conv = sum(1 for f in finals if math.isfinite(f) and f < 1e-2)
        d = {"value_max_final_median": r6(np.median(finals)), "value_max_final_min": r6(np.min(finals)),
             "value_max_final_max": r6(np.max(finals)),
             "diverged": div, "converged": conv, "n": len(finals),
             "sweeps_to_0.01_median": r6(np.median([c for c in conv_times if c is not None])) if any(c is not None for c in conv_times) else None}
        if extra:
            d.update(extra)
        conds[name] = d

    starts = [vmax(init_w(s)) for s in SEEDS]
    # a) 셋 다: 선형 + 부트스트랩 + 균등 갱신(목표 정책의 궤적이 아님)
    fin, ct = [], []
    for s in SEEDS:
        w, h, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, init_w(s), a, g, SWEEPS, record_every=10, blowup=1e300)
        fin.append(vmax(w)); ct.append(time_to(h))
    summarize("all_three", fin, starts, ct)
    # b) 함수 근사 빼기: 표
    fin, ct = [], []
    for s in SEEDS:
        v0 = PHI @ init_w(s)
        w, h, _ = F.semi_gradient_td_sweeps(np.eye(6), NXT, R, v0, a, g, SWEEPS, record_every=10)
        fin.append(vmax(w, np.eye(6))); ct.append(time_to(h, Phi=np.eye(6)))
    long_t = sweeps_to_converge(np.eye(6), NXT, [PHI @ init_w(s) for s in SEEDS], a, g, 0.0, 1_000_000)
    summarize("no_function_approx_table", fin, starts, ct,
              {"sweeps_to_0.01_long_median": r6(np.median(long_t)) if all(t is not None for t in long_t) else None,
               "sweeps_to_0.01_long_range": [min(long_t), max(long_t)] if all(t is not None for t in long_t) else None})
    # c) 부트스트랩 빼기: 참값(0)을 목표로 하는 회귀
    fin, ct = [], []
    for s in SEEDS:
        w, h, _ = F.regression_sweeps(PHI, np.zeros(6), init_w(s), a, SWEEPS, record_every=10)
        fin.append(vmax(w)); ct.append(time_to(h))
    summarize("no_bootstrap_regression", fin, starts, ct)
    # d) 오프폴리시 빼기: 행동 정책 자신의 값을 그 궤적으로(온폴리시). 6걸음 = 1스윕으로 환산
    fin, wn = [], []
    for s in SEEDS:
        w, dv, _ = F.td0_trajectory(PHI, P_B, P_T, R, init_w(s), a, g, 6 * SWEEPS, seed=1000 + s, off_policy=False)
        fin.append(vmax(w)); wn.append(float(np.linalg.norm(w)))
    summarize("no_off_policy_on_trajectory", fin, starts, [None] * len(SEEDS),
              {"w_norm_final_median": r6(np.median(wn)), "steps": 6 * SWEEPS})
    # e) 셋 다, 실제 궤적 + 중요도 비
    fin, dvs = [], []
    for s in SEEDS:
        w, dv, _ = F.td0_trajectory(PHI, P_B, P_T, R, init_w(s), a, g, 6 * SWEEPS, seed=1000 + s, off_policy=True)
        fin.append(vmax(w) if dv is None else float("inf")); dvs.append(dv)
    summarize("all_three_trajectory_importance_ratio", fin, starts, [None] * len(SEEDS),
              {"steps": 6 * SWEEPS, "blowup_step_median": r6(np.median([d for d in dvs if d])) if any(dvs) else None})
    # f) 잔차 경사(삼각형은 그대로, 갱신 규칙만 바꿈)
    fin, ct = [], []
    for s in SEEDS:
        w, h, _ = F.residual_sweeps(PHI, NXT, R, init_w(s), a, g, SWEEPS, mix=1.0, record_every=10)
        fin.append(vmax(w)); ct.append(time_to(h))
    long_t = sweeps_to_converge(PHI, NXT, [init_w(s) for s in SEEDS], a, g, 1.0, 1_000_000)
    be_drop = []
    for s in SEEDS:
        w, _, _ = F.residual_sweeps(PHI, NXT, R, init_w(s), a, g, SWEEPS, mix=1.0)
        be_drop.append(F.bellman_error(w, PHI, P_T, R, g, D_U) / F.bellman_error(init_w(s), PHI, P_T, R, g, D_U))
    summarize("residual_gradient", fin, starts, ct,
              {"sweeps_to_0.01_long_median": r6(np.median(long_t)) if all(t is not None for t in long_t) else None,
               "bellman_error_ratio_after_median": r6(np.median(be_drop))})
    # 같은 다리 빼기를 gamma 0.9에서도(삼각형이 다 있어도 수렴하는지)
    fin = []
    for s in SEEDS:
        w, _, _ = F.semi_gradient_td_sweeps(PHI, NXT, R, init_w(s), a, 0.9, SWEEPS)
        fin.append(vmax(w))
    g09 = {"value_max_final_median": r6(np.median(fin)), "converged": sum(f < 1e-2 for f in fin), "n": len(fin)}
    fin, dvs = [], []
    for s in SEEDS:
        w, dv, _ = F.td0_trajectory(PHI, P_B, P_T, R, init_w(s), a, 0.9, 6 * SWEEPS, seed=1000 + s, off_policy=True)
        fin.append(vmax(w) if dv is None else float("inf")); dvs.append(dv)
    g09_traj = {"value_max_final_median": r6(np.median(fin)), "converged": sum(f < 1e-2 for f in fin),
                "diverged": sum(1 for f in fin if not math.isfinite(f)), "n": len(fin)}
    return {"gamma": g, "alpha": a, "sweeps": SWEEPS, "start_value_max_median": r6(np.median(starts)),
            "conditions": conds, "gamma0.9_all_three_uniform": g09, "gamma0.9_all_three_trajectory": g09_traj}


# ---------------- E4 ----------------

def e4_grid():
    rows = []
    for g in [0.5, 0.9, 0.95, 0.99]:
        A = F.expected_update_matrix(PHI, P_T, D_U, g)
        for a in [0.001, 0.01, 0.1]:
            lam = np.linalg.eigvals(A)
            lam = lam[np.abs(lam) > 1e-9]        # Phi의 영공간 방향(고윳값 0)은 값에 영향이 없어 뺀다
            rad = float(np.max(np.abs(1 - 6 * a * lam)))
            w, _, dv = F.semi_gradient_td_sweeps(PHI, NXT, R, W_BAIRD, a, g, SWEEPS, blowup=1e300)
            vm = vmax(w)
            rows.append({"gamma": g, "alpha": a, "spectral_radius": r6(rad), "theory": "발산" if rad > 1 + 1e-12 else "수렴",
                         "value_max_after": r6(vm), "measured": "발산" if (not math.isfinite(vm) or vm > 12) else ("수렴" if vm < 1e-2 else "덜 줄어듦")})
    return {"sweeps": SWEEPS, "start_value_max": 12.0, "rows": rows}


# ---------------- E5 ----------------

def sweeps_to_converge(Phi, nxt, inits, a, g, mix, cap, thr=1e-2):
    """시드 여러 개를 한꺼번에(행 = 시드) 스윕한다. 시드마다 값 오차가 thr 아래로 처음 내려간 스윕 수.
    한 시드라도 발산하면(노름 > 1e12) None들을 돌려준다. 보상 0이라 한 스윕은 w <- T w (T = I - a M)."""
    M = (Phi - mix * g * Phi[nxt]).T @ (Phi - g * Phi[nxt])
    T = np.eye(Phi.shape[1]) - a * M
    W = np.array(inits, float)
    done = [None] * len(W)
    for k in range(1, cap + 1):
        W = W @ T.T
        if not np.all(np.isfinite(W)) or np.abs(W).max() > 1e12:
            return [None] * len(W)
        vm = np.abs(W @ Phi.T).max(axis=1)
        for i in np.nonzero(vm < thr)[0]:
            if done[i] is None:
                done[i] = k
        if all(d is not None for d in done):
            break
    return done


def e5_residual():
    out = {}
    cap = 1_000_000
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    for prob in ("hall", "star"):
        if prob == "hall":
            Phi, nxt, _ = F.hall()
            inits = [np.random.default_rng(s).uniform(0, 10, 6) for s in SEEDS]
        else:
            Phi, nxt = PHI, NXT
            inits = [init_w(s) for s in SEEDS]
        for g in (0.9, 0.99):
            for name, mix in (("direct", 0.0), ("residual_gradient", 1.0)):
                best = None
                for a in alphas:
                    ts = sweeps_to_converge(Phi, nxt, inits, a, g, mix, cap)
                    if all(t is not None for t in ts):
                        med = float(np.median(ts))
                        if best is None or med < best["median"]:
                            best = {"alpha": a, "median": med, "min": min(ts), "max": max(ts)}
                out[f"{prob}_g{g}_{name}"] = best if best else {"alpha": None, "median": None,
                                                                "note": f"격자의 어떤 alpha로도 {cap} 스윕 안에 시드 10개 모두 0.01 아래로 못 내려감"}
    # 잔차 알고리즘 섞는 비율 phi의 안정 경계(별 문제, D 균등)
    mix_thr = {}
    for g in (0.95, 0.99):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = (lo + hi) / 2
            if np.linalg.eigvals(F.residual_update_matrix(PHI, P_T, D_U, g, m)).real.min() < -1e-13:
                lo = m
            else:
                hi = m
        mix_thr[str(g)] = r6(hi)
    out["residual_mix_threshold"] = mix_thr
    spec = {}
    for prob, (Phi, nxt) in (("hall", F.hall()[:2]), ("star", (PHI, NXT))):
        for g in (0.9, 0.99):
            M = (Phi - g * Phi[nxt]).T @ (Phi - g * Phi[nxt])
            ev = np.sort(np.linalg.eigvalsh(M))
            ev = ev[ev > 1e-9]
            spec[f"{prob}_g{g}"] = {"min_nonzero": r6(ev[0]), "max": r6(ev[-1]), "ratio": r6(ev[-1] / ev[0])}
    out["residual_gradient_spectrum"] = spec
    out["cap"] = cap
    out["alphas"] = alphas
    return out


# ---------------- E6 ----------------

def e6_timing():
    w = W_BAIRD.copy()
    reps = 20000
    t0 = time.perf_counter()
    for _ in range(reps):
        v = PHI @ w
        delta = R + 0.9 * v[NXT] - v
        w = w + 0.001 * PHI.T @ delta
    sweep_us = (time.perf_counter() - t0) / reps * 1e6
    t0 = time.perf_counter()
    F.td0_trajectory(PHI, P_B, P_T, R, W_BAIRD, 0.001, 0.9, 20000, seed=0)
    step_us = (time.perf_counter() - t0) / 20000 * 1e6
    eig = {}
    rng = np.random.default_rng(0)
    for d in (7, 70, 700):
        M = rng.standard_normal((d, d))
        ts = []
        for _ in range(3 if d == 700 else 7):
            t0 = time.perf_counter()
            np.linalg.eigvals(M)
            ts.append(time.perf_counter() - t0)
        eig[str(d)] = r6(np.median(ts) * 1e3)
    return {"sweep_us": r6(sweep_us), "trajectory_step_us": r6(step_us), "eig_ms": eig}


# ---------------- E7 ----------------

def e7_torch():
    out = {}
    for name, mix, det in (("semi_gradient", 0.0, True), ("residual_gradient", 1.0, False)):
        w = W_BAIRD.copy()
        ws = [w.copy()]
        for _ in range(300):
            w, _, _ = F.residual_sweeps(PHI, NXT, R, w, 0.01, 0.99, 1, mix=mix)
            ws.append(w.copy())
        ws = np.array(ws)
        wt = FT.run_sweeps(PHI, NXT, R, W_BAIRD, 0.01, 0.99, 300, detach_target=det).numpy()
        rel = np.max(np.abs(ws - wt) / np.maximum(np.abs(ws), 1e-12))
        out[name] = {"max_rel_diff": r6(rel), "allclose_rtol_1e-9": bool(np.allclose(ws, wt, rtol=1e-9, atol=1e-12))}
    return out


# ---------------- E8 ----------------

def e8_log():
    log = []
    w, dv, _ = F.td0_trajectory(PHI, P_B, P_T, R, W_BAIRD, 0.01, 0.99, 600, seed=0, off_policy=True, log=log, log_steps=600)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        for row in log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rhos = [r["info"]["rho"] for r in log]
    return {"steps": len(log), "rho_values": sorted(set(rhos)), "share_rho_nonzero": r6(np.mean([x > 0 for x in rhos])),
            "w_norm_start": r6(np.linalg.norm(W_BAIRD)), "w_norm_end": r6(log[-1]["info"]["w_norm"])}


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                "machine": platform.machine(), "system": platform.system(), "threads": 2},
        "problem": {"name": "Baird(1995) 6상태 별 문제", "features": PHI.tolist(), "next_state(1-based)": (NXT + 1).tolist(),
                    "rewards": 0, "true_values": [0] * 6, "w_baird_init": W_BAIRD.tolist(),
                    "behavior": "모든 상태에서 6개 상태로 1/6씩(Baird Q-러닝 판의 행동 정책: 1/6로 상태 6, 5/6로 1~5 균등)",
                    "target": "모든 상태에서 상태 6으로"},
        "E0_hand": e0_hand(),
        "E1_fail": {"gamma0.99": e1_fail(0.99), "gamma0.9": e1_fail(0.9)},
        "E2_eigen": e2_eigen(),
        "E3_ablation": e3_ablation(),
        "E4_grid": e4_grid(),
        "E5_residual": e5_residual(),
        "E6_timing": e6_timing(),
        "E7_torch": e7_torch(),
        "E8_log": e8_log(),
    }
    res["runtime_s"] = r6(time.perf_counter() - t0)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E0_hand", "E2_eigen", "E3_ablation", "E4_grid", "E5_residual", "E6_timing", "E7_torch", "E8_log", "runtime_s")},
                     ensure_ascii=False, indent=1)[:12000])


if __name__ == "__main__":
    main()
