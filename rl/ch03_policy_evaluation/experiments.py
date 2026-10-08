"""강화학습 3장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch03_policy_evaluation/experiments.py`

E0 손계산 확인: 한 줄 복도 [A][B][도착], gamma 0.9. 연립방정식의 해와 반복 평가의 처음 몇 스윕
E1 실패 재현: n x n 격자(n = 10, 20, 40, 60, 70), 균등 무작위 정책, gamma 0.99.
   직접 풀이(조밀 LU) vs 반복 평가(조밀·희소)의 실측 시간, 메모리, 스윕 수, 이론 연산량
E2 차원 축: 축마다 10칸인 d차원 격자(d = 1..6)의 상태 수와 조밀 P^pi 메모리 (계산만, 실행하지 않음)
E3 할인율: 20 x 20에서 gamma = 0.9, 0.99, 0.999. 스윕 수 vs 예측 log(eps/e0)/log(gamma),
   그리고 종료 칸 때문에 실제 비율이 gamma * rho 로 줄어드는 효과. 앞뒤 차이로 멈췄을 때의 실제 오차
E4 제자리 반복(가우스-자이델식) vs 두 배열 반복(야코비식), 고치는 순서 두 가지
E5 구현 대조: NumPy vs PyTorch, policy_matrix vs 2장식 P[S,A,S]
E6 그림용: 10 x 10에서 스윕마다 가치가 퍼지는 모습, 20 x 20 오차 곡선
E7 특이행렬: gamma = 1에서 solve가 어떻게 되나 (종료 칸 있음 / 없음)
E8 몬테카를로 확인: 10 x 10에서 실제로 걸어 본 평균 할인 리턴 vs 방정식 값. 로그는 logs/ch03_rollouts.jsonl

결과는 results/ch03.json.
"""

import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch03_eval_np as ev  # noqa: E402
import rl_ch03_eval_torch as evt  # noqa: E402
import rl_ch03_grid as g  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch03.json"
LOG = HERE / "logs" / "ch03_rollouts.jsonl"
EPS = 1e-6
GAMMA = 0.99


def timed(fn, reps):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def e0_corridor(gamma=0.9):
    c = g.corridor()
    P, r = c["P_pi"], c["r_pi"]
    V = ev.evaluate_direct(P, r, gamma)
    rows, Vk = [], np.zeros(3)
    rows.append([float(x) for x in Vk])
    for _ in range(8):
        Vk = r + gamma * P @ Vk
        rows.append([float(x) for x in Vk])
    _, sweeps, _ = ev.evaluate_iterative(P, r, gamma, EPS, V_true=V)
    a = 0.5 * gamma  # A: 0.5 확률로 제자리, 0.5 확률로 B
    return {
        "gamma": gamma,
        "equations": "V(A) = 0.45 V(A) + 0.45 V(B);  V(B) = 0.45 V(A) + 0.5",
        "VA_over_VB": a / (1 - a),
        "V_direct": [float(x) for x in V],
        "sweeps_table": rows,
        "errors_A_B": [[float(V[0] - x[0]), float(V[1] - x[1])] for x in rows],
        "sweeps_to_1e-6": sweeps,
    }


def e1_sizes(sizes=(10, 20, 40, 60, 70)):
    out = {}
    for n in sizes:
        m = g.chapter_grid(n)
        S = n * n
        P, r = m["P_pi"], m["r_pi"]
        V, t_direct = timed(lambda: ev.evaluate_direct(P, r, GAMMA), 5)
        (Vd, kd, _), t_dense = timed(lambda: ev.evaluate_iterative(P, r, GAMMA, EPS, V_true=V), 3)
        (Vs, ks, _), t_sparse = timed(lambda: ev.evaluate_iterative_sparse(m["nbr"], m["prob"], r, GAMMA, EPS, V_true=V), 5)
        out[str(n)] = {
            "states": S,
            "dense_matrix_bytes": 8 * S * S,
            "direct_seconds": t_direct,
            "dense_iter_seconds": t_dense,
            "sparse_iter_seconds": t_sparse,
            "sweeps": kd,
            "sweeps_sparse": ks,
            "flops_direct_lu": 2 / 3 * S**3 + 2 * S**2,
            "flops_dense_iter": 2 * S * S * kd,
            "flops_sparse_iter": 8 * S * ks,
            "max_diff_dense_vs_direct": float(np.max(np.abs(Vd - V))),
            "max_diff_sparse_vs_direct": float(np.max(np.abs(Vs - V))),
            "V_start": float(V[0]),
            "V_max": float(V.max()),
        }
        print(f"n={n}: direct {t_direct:.4f}s, dense {t_dense:.4f}s, sparse {t_sparse:.4f}s, sweeps {kd}")
    a, b = out["40"], out["70"]
    slope = np.log(b["direct_seconds"] / a["direct_seconds"]) / np.log(b["states"] / a["states"])
    rate = b["flops_direct_lu"] / b["direct_seconds"]
    return {"gamma": GAMMA, "eps": EPS, "goal": "오른쪽 아래 구석 +1", "by_n": out,
            "direct_time_exponent_40_to_70": float(slope), "direct_flops_per_second_n70": float(rate)}


def e2_dims(rate):
    rows = []
    for d in range(1, 7):
        S = 10**d
        lu = 2 / 3 * S**3
        rows.append({
            "d": d, "states": S,
            "dense_bytes": 8 * S * S,
            "sparse_bytes": S * (2 * d) * (8 + 8) + 8 * S,  # 이웃 2d개의 번호·확률 + r^pi
            "flops_lu": lu,
            "lu_seconds_at_measured_rate": lu / rate,
        })
    # 조밀 행렬이 80GB(80e9 바이트)를 넘는 상태 수
    s80 = (80e9 / 8) ** 0.5
    return {"axis_cells": 10, "rows": rows, "states_for_80GB_dense": s80, "d_for_80GB_dense": float(np.log10(s80))}


def e3_gamma(n=20, gammas=(0.9, 0.99, 0.999)):
    m = g.chapter_grid(n)
    P, r = m["P_pi"], m["r_pi"]
    rho = ev.spectral_radius_nonterminal(P, m["term"])
    out = {}
    for gm in gammas:
        V = ev.evaluate_direct(P, r, gm)
        e0 = float(np.max(np.abs(V)))
        _, k_true, errs = ev.evaluate_iterative_sparse(m["nbr"], m["prob"], r, gm, EPS, V_true=V)
        V_diff, k_diff, _ = ev.evaluate_iterative_sparse(m["nbr"], m["prob"], r, gm, EPS, V_true=None)
        # 마지막 몇십 스윕에서 본 실제 감소 비율
        tail = np.array(errs[-50:])
        obs_rate = float(np.exp(np.mean(np.diff(np.log(tail))))) if len(tail) > 2 else None
        out[str(gm)] = {
            "V_start": float(V[0]), "e0": e0,
            "sweeps_true_error": k_true,
            "pred_log_eps_over_log_gamma": ev.predicted_sweeps(EPS, gm, 1.0),
            "pred_with_e0": ev.predicted_sweeps(EPS, gm, e0),
            "pred_with_gamma_rho": ev.predicted_sweeps(EPS, gm, e0, rate=gm * rho),
            "gamma_rho": gm * rho,
            "observed_rate_tail": obs_rate,
            "sweeps_diff_stop": k_diff,
            "true_error_at_diff_stop": float(np.max(np.abs(V_diff - V))),
            "worst_case_error_bound_at_diff_stop": gm / (1 - gm) * EPS,
        }
        print(f"gamma={gm}: sweeps {k_true}, pred {out[str(gm)]['pred_with_e0']}, pred(gamma*rho) {out[str(gm)]['pred_with_gamma_rho']}")
    return {"n": n, "rho_nonterminal": rho, "eps": EPS, "by_gamma": out}


def e4_inplace(sizes=(10, 20, 40), gamma=GAMMA):
    out = {}
    for n in sizes:
        m = g.chapter_grid(n, dense=(n <= 40))
        S = n * n
        V = ev.evaluate_direct(m["P_pi"], m["r_pi"], gamma)
        (_, kj, _), tj = timed(lambda: ev.evaluate_iterative_sparse(m["nbr"], m["prob"], m["r_pi"], gamma, EPS, V_true=V), 3)
        (_, kf, _), tf = timed(lambda: ev.evaluate_inplace(m["nbr"], m["prob"], m["r_pi"], gamma, EPS, V_true=V), 1)
        rev = np.arange(S)[::-1]  # 도착 칸(마지막 번호)에서 가까운 칸부터
        (_, kr, _), tr = timed(lambda: ev.evaluate_inplace(m["nbr"], m["prob"], m["r_pi"], gamma, EPS, V_true=V, order=rev), 1)
        out[str(n)] = {"jacobi_sweeps": kj, "inplace_forward_sweeps": kf, "inplace_reverse_sweeps": kr,
                       "jacobi_seconds": tj, "inplace_forward_seconds": tf, "inplace_reverse_seconds": tr}
        print(f"n={n}: jacobi {kj}, inplace fwd {kf}, rev {kr}")
    return {"gamma": gamma, "eps": EPS, "order_forward": "0, 1, ..., S-1 (도착 칸이 마지막)",
            "order_reverse": "S-1, ..., 0 (도착 칸 쪽부터)", "by_n": out}


def e5_match():
    m = g.chapter_grid(20)
    P, r = m["P_pi"], m["r_pi"]
    Vn = ev.evaluate_direct(P, r, GAMMA)
    Vt = evt.evaluate_direct(torch.tensor(P), torch.tensor(r), GAMMA).numpy()
    Vni, kn, _ = ev.evaluate_iterative(P, r, GAMMA, EPS, V_true=Vn)
    Vti, kt = evt.evaluate_iterative(torch.tensor(P), torch.tensor(r), GAMMA, EPS, V_true=torch.tensor(Vn))
    # policy_matrix: 2장식 P[S,A,S]에서 정책을 곱한 것과 바로 만든 P^pi가 같은지 (6x6, slip 0)
    Pf, Rf = g.make_grid_full(6, 6, {(5, 5): 1.0}, ((5, 5),))
    pi = np.full((36, 4), 0.25)
    m6 = g.chapter_grid(6)
    keep = ~m["term"]
    Q = P[np.ix_(keep, keep)]
    return {
        "nonterminal_block_symmetric_max_diff": float(np.max(np.abs(Q - Q.T))),
        "direct_np_vs_torch": float(np.max(np.abs(Vn - Vt))),
        "iter_np_vs_torch": float(np.max(np.abs(Vni - Vti.numpy()))),
        "iter_sweeps_np_torch": [kn, kt],
        "policy_matrix_vs_direct_build": float(np.max(np.abs(g.policy_matrix(Pf, pi) - m6["P_pi"]))),
        "policy_reward_vs_direct_build": float(np.max(np.abs(g.policy_reward(Pf, Rf, pi) - m6["r_pi"]))),
    }


def e6_frames():
    m = g.chapter_grid(10)
    P, r = m["P_pi"], m["r_pi"]
    V = ev.evaluate_direct(P, r, GAMMA)
    keep = (1, 2, 5, 10, 20, 50, 100, 300)
    frames, Vk = {}, np.zeros(100)
    for k in range(1, max(keep) + 1):
        Vk = r + GAMMA * P @ Vk
        if k in keep:
            frames[str(k)] = Vk.reshape(10, 10).round(4).tolist()
    nonzero = {str(k): int((np.array(v) > 0).sum()) for k, v in frames.items()}
    m20 = g.chapter_grid(20)
    V20 = ev.evaluate_direct(m20["P_pi"], m20["r_pi"], GAMMA)
    _, k20, errs = ev.evaluate_iterative_sparse(m20["nbr"], m20["prob"], m20["r_pi"], GAMMA, EPS, V_true=V20)
    pick = [1, 10, 50, 100, 200, 400, 600, 800, k20]
    return {"n": 10, "frames": frames, "true": V.reshape(10, 10).round(4).tolist(), "nonzero_cells": nonzero,
            "error_curve_20": {str(k): errs[k - 1] for k in pick}, "error_bound_20": {str(k): float(GAMMA**k * np.max(V20)) for k in pick}}


def e7_singular():
    out = {}
    m = g.chapter_grid(10)
    S = 100
    A = np.eye(S) - m["P_pi"]
    try:
        np.linalg.solve(A, m["r_pi"])
        out["with_terminal"] = {"raised": False}
    except np.linalg.LinAlgError as ex:
        out["with_terminal"] = {"raised": True, "error": f"LinAlgError: {ex}"}
    # 종료 칸을 없애고(도착 칸도 그냥 걷는 칸) 보상만 남긴다: 행 합이 모두 1인 확률 표
    nbr, _ = g.neighbors(10, 10, ())
    P = np.zeros((S, S))
    np.add.at(P, (np.repeat(np.arange(S), 4), nbr.ravel()), 0.25)
    r = np.zeros(S)
    r[98] = r[89] = 0.25  # 도착 칸 이웃이 그쪽으로 걸을 때 +1
    A2 = np.eye(S) - P
    try:
        V = np.linalg.solve(A2, r)
        out["no_terminal"] = {"raised": False, "max_abs_V": float(np.max(np.abs(V))),
                              "residual": float(np.max(np.abs(A2 @ V - r))), "rank": int(np.linalg.matrix_rank(A2)),
                              "cond": float(np.linalg.cond(A2))}
    except np.linalg.LinAlgError as ex:
        out["no_terminal"] = {"raised": True, "error": str(ex)}
    return out


def e8_montecarlo(seeds=range(10), episodes=200, max_steps=20000):
    m = g.chapter_grid(10, dense=True)
    V = ev.evaluate_direct(m["P_pi"], m["r_pi"], GAMMA)
    nbr, rew, term = m["nbr"], m["rew"], m["term"]
    returns, steps, log = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for ep in range(episodes):
            s, G, disc = 0, 0.0, 1.0
            acts = rng.integers(0, 4, size=max_steps)
            for t in range(max_steps):
                a = int(acts[t])
                s2, rr = int(nbr[s, a]), float(rew[s, a])
                G += disc * rr
                disc *= GAMMA
                done = bool(term[s2])
                if seed == 0 and ep == 0 and t < 40:
                    log.append({"seed": seed, "episode": ep, "t": t, "s": s, "a": a, "r": rr, "s_next": s2,
                                "done": done, "info": {}})
                s = s2
                if done:
                    break
            returns.append(G)
            steps.append(t + 1)
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log))
    R = np.array(returns)
    return {"episodes": len(R), "mean_return": float(R.mean()), "stderr": float(R.std(ddof=1) / np.sqrt(len(R))),
            "V_start": float(V[0]), "mean_steps": float(np.mean(steps)), "median_steps": float(np.median(steps)),
            "log_lines": len(log)}


def main():
    t0 = time.perf_counter()
    cfg = np.show_config(mode="dicts")
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
                   "blas": cfg["Build Dependencies"]["blas"]["name"], "lapack": cfg["Build Dependencies"]["lapack"]["name"],
                   "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}}
    res["E0"] = e0_corridor()
    res["E1"] = e1_sizes()
    res["E2"] = e2_dims(res["E1"]["direct_flops_per_second_n70"])
    res["E3"] = e3_gamma()
    res["E4"] = e4_inplace()
    res["E5"] = e5_match()
    res["E6"] = e6_frames()
    res["E7"] = e7_singular()
    res["E8"] = e8_montecarlo()
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"saved {OUT} in {res['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
