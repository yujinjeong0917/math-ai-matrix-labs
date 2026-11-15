"""강화학습 4장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch04_policy_value_iteration/experiments.py`

E0 손계산 확인: 한 줄 복도 [A][B][도착], gamma 0.9, 처음 정책은 모두 "왼쪽". 정책 반복과 가치 반복의 매 단계
E1 실패 1(동률): 6x6, 미끄러짐 없음, gamma 0.95. 동률을 무작위로 깨면 정책 반복이 끝나지 않는다(최대 1,000번)
E2 실패 2(gamma=1): 종료 칸이 없는 3x3, (0,0)에 들어가면 +1. 가치 반복의 ||V_k||가 k에 비례해 커진다
E3 비교 측정: 20x20, 미끄러짐 0.1, gamma 0.99. 정책 반복·가치 반복·수정 정책 반복(k=1,5,20)
   반복 수, 백업 수, 직접 풀이 FLOP, 실측 시간(5회 중앙값), 최종 오차, 탐욕 정책이 최적이 된 시점
E4 할인율: 10x10, 미끄러짐 0.1, gamma = 0.9, 0.99, 0.999. 가치 반복 스윕 수 vs 이론값 log(eps(1-gamma))/log(gamma)
E5 정책 개선 정리 확인: E3 격자에서 정책 반복의 V^{pi_k}가 모든 칸에서 줄지 않는지
E6 축약 부등식: 무작위 V, U 100쌍에서 ||TV - TU|| / ||V - U|| <= gamma
E7 구현 대조: NumPy vs PyTorch
E8 롤아웃: E3 격자에서 최적 정책으로 실제로 걸어 본 평균 할인 리턴 vs V*(시작). 로그는 logs/ch04_rollouts.jsonl
E9 그림용: 6x6에서 정책 반복이 바꾼 화살표(반복별)

결과는 results/ch04.json.
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
import rl_ch04_grid as g  # noqa: E402
import rl_ch04_pivi_np as m  # noqa: E402
import rl_ch04_pivi_torch as mt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch04.json"
LOG = HERE / "logs" / "ch04_rollouts.jsonl"
THETA = 1e-10  # 수렴 허용오차 theta: 가치 반복·수정 정책 반복이 앞뒤 차이 < THETA 에서 멈춘다


def timed(fn, reps=5):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def arrows(pi, term, w):
    """정책 pi[S]를 화살표 문자열 행으로 바꾼다(종료 칸은 ◎). w: 격자 너비. 칸 번호 = 행 * w + 열."""
    rows = []
    for r in range(len(pi) // w):
        rows.append("".join("◎" if term[r * w + c] else g.ARROWS[pi[r * w + c]] for c in range(w)))
    return rows


def optimal(P, R, gamma):
    # 기준값 V*, pi*: 가치 반복을 theta=1e-13 으로 촘촘히 돌린 뒤 탐욕 정책을 뽑고
    V, _ = m.value_iteration(P, R, gamma, theta=1e-13)
    pi = m.greedy_policy(V, P, R, gamma, tie="first")
    # 그 정책을 직접 풀이로 다시 평가해 반복 오차가 섞이지 않은 V^{pi*}를 쓴다
    V = m.evaluate_exact(pi, P, R, gamma)
    return V, pi


def e0_corridor(gamma=0.9):
    P, R, term = g.corridor()
    pi0 = np.full(3, 3)  # 모두 왼쪽
    pi, V, hist, ok = m.policy_iteration(P, R, gamma, pi0=pi0, tie="keep")
    steps, cur = [], pi0.copy()
    for h in hist:
        Q = m.q_from_v(h["V"], P, R, gamma)
        new = m.greedy_policy(h["V"], P, R, gamma, tie="keep", old=cur)
        steps.append({"policy": [g.ARROWS[a] for a in cur[:2]], "V_AB": [round(float(x), 6) for x in h["V"][:2]],
                      "Q_right_AB": [round(float(Q[s, 1]), 6) for s in (0, 1)],
                      "Q_left_AB": [round(float(Q[s, 3]), 6) for s in (0, 1)],
                      "next_policy": [g.ARROWS[a] for a in new[:2]], "changed": h["changed"]})
        cur = new
    Vk, vi_rows = np.zeros(3), [[0.0, 0.0]]
    for _ in range(4):
        # 식: V_{k+1} = T V_k  (가치 반복의 손계산 표)
        Vk = m.bellman_T(Vk, P, R, gamma)
        vi_rows.append([round(float(x), 6) for x in Vk[:2]])
    return {"gamma": gamma, "pi_iterations": len(hist), "converged": ok, "steps": steps,
            "final_policy_AB": [g.ARROWS[a] for a in pi[:2]], "vi_rows": vi_rows}


def e1_ties(gamma=0.95, max_iter=1000, seeds=range(10)):
    P, R, term = g.tie_grid(6)
    S = P.shape[0]
    V_star, pi_star = optimal(P, R, gamma)
    Q = m.q_from_v(V_star, P, R, gamma)
    # 동률 후보: 식 Q(s,a) >= max_a Q(s,a) - TIE_TOL. 비종료 칸마다 최적 행동 개수가 n_best이고, 그 곱이 최적 결정론 정책 수
    best = Q >= Q.max(axis=1, keepdims=True) - m.TIE_TOL
    n_best = best.sum(axis=1)[~term]
    out = {"gamma": gamma, "n_states": S, "nonterminal": int((~term).sum()),
           "n_det_policies_log10": float((~term).sum() * np.log10(4)),
           "tie_cells": int((n_best > 1).sum()),
           "tie_cells_exact_equal": int((((Q == Q.max(axis=1, keepdims=True)).sum(axis=1) > 1) & ~term).sum()),
           "n_optimal_det_policies": int(np.prod(n_best.astype(np.int64))),
           "V_star_start": float(V_star[0]), "rules": {}}
    pi0 = np.zeros(S, dtype=np.int64)  # 모두 위
    for tie in ("first", "keep", "raw"):
        pi, V, hist, ok = m.policy_iteration(P, R, gamma, pi0=pi0, tie=tie, max_iter=max_iter)
        out["rules"][tie] = {"converged": ok, "iterations": len(hist), "changed": [h["changed"] for h in hist],
                             "final_err": float(np.max(np.abs(V - V_star)))}
    runs = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        pi, V, hist, ok = m.policy_iteration(P, R, gamma, pi0=pi0, tie="random", rng=rng, max_iter=max_iter)
        errs = [float(np.max(np.abs(h["V"] - V_star))) for h in hist]
        # 정책의 가치가 V*에 1e-12 안으로 처음 닿은 반복(1부터 셈). 이후에도 정책이 계속 바뀌는지가 실패 1의 핵심
        first_opt = next(i + 1 for i, e in enumerate(errs) if e < 1e-12)
        ch = np.array([h["changed"] for h in hist])
        runs.append({"seed": sd, "converged": ok, "iterations": len(hist), "first_optimal_iter": first_opt,
                     "changed_after_optimal_mean": float(ch[first_opt:].mean()),
                     "changed_after_optimal_min": int(ch[first_opt:].min()),
                     "changed_first12": ch[:12].tolist(),
                     "max_err_after_optimal": float(max(errs[first_opt - 1:]))})
    out["random"] = runs
    # 고친 규칙(keep)은 무작위로 시작한 정책에서도 끝나는가
    keep_runs = []
    for sd in seeds:
        pi0r = np.random.default_rng(100 + sd).integers(0, 4, S)
        pi, V, hist, ok = m.policy_iteration(P, R, gamma, pi0=pi0r, tie="keep", max_iter=max_iter)
        keep_runs.append({"seed": 100 + sd, "converged": ok, "iterations": len(hist),
                          "final_err": float(np.max(np.abs(V - V_star)))})
    out["keep_random_start"] = keep_runs
    return out


def e2_gamma_one(K=50):
    P, R, term = g.loop_grid()
    rows = {}
    for gamma in (1.0, 0.9):
        V, norms, diffs = np.zeros(9), [], []
        for _ in range(K):
            # 식: V_{k+1} = T V_k. gamma=1 이면 norm이 k에 비례해 커지고 diff가 0으로 가지 않는다
            V2 = m.bellman_T(V, P, R, gamma)
            diffs.append(float(np.max(np.abs(V2 - V))))
            V = V2
            norms.append(float(np.max(np.abs(V))))
        rows[str(gamma)] = {"norm": norms, "diff": diffs, "V_final_grid": np.round(V.reshape(3, 3), 4).tolist()}
    # gamma=1이면 직접 풀이도 망가진다: 최적 정책(구석에 머물기)의 I - P^pi
    V_s, pi_s = optimal(P, R, 0.9)
    Ppi, rpi = m.policy_mats(pi_s, P, R)
    A = np.eye(9) - 1.0 * Ppi  # 식: I - gamma P^pi 에 gamma = 1 을 대입. 행 합이 1인 P^pi라 특이행렬이 된다
    try:
        np.linalg.solve(A, rpi)
        solve = "ok"
    except np.linalg.LinAlgError as ex:
        solve = f"LinAlgError: {ex}"
    rows["solve_gamma1"] = solve
    rows["rank_I_minus_Ppi"] = int(np.linalg.matrix_rank(A))
    rows["policy_gamma09"] = arrows(pi_s, np.zeros(9, bool), 3)
    return rows


def e3_compare(n=20, gamma=0.99, reps=5):
    (P, R, term), pits = g.compare_grid(n)
    S, A = P.shape[0], P.shape[1]
    V_star, pi_star = optimal(P, R, gamma)
    res = {"n": n, "S": S, "A": A, "gamma": gamma, "slip": 0.1, "pits": pits, "theta": THETA,
           "V_star_start": float(V_star[0]), "methods": {}}

    (pi, V, hist, ok), t = timed(lambda: m.policy_iteration(P, R, gamma, tie="first"), reps)
    k = len(hist)
    # 정책 반복의 비용: 반복마다 개선 S*A 백업 + 직접 풀이 (2/3)S^3 FLOP (평가 백업은 따로 세지 않는다)
    res["methods"]["PI"] = {"iterations": k, "backups": k * S * A, "solve_flops": int(k * (2 / 3) * S ** 3),
                            "time_s": t, "err": float(np.max(np.abs(V - V_star))),
                            "policy_diff_cells": int((pi != pi_star).sum()), "changed": [h["changed"] for h in hist]}

    def vi():
        return m.value_iteration(P, R, gamma, theta=THETA, V_star=V_star)
    (V, hist), t = timed(vi, reps)
    pi = m.greedy_policy(V, P, R, gamma)
    # 탐욕 정책이 처음으로 최적이 된(그 뒤로 계속 최적) 스윕
    Vk, opt_at, last_bad = np.zeros(S), None, 0
    for kk in range(1, len(hist) + 1):
        Vk = m.bellman_T(Vk, P, R, gamma)  # 식: V_k = T V_{k-1}
        pk = m.greedy_policy(Vk, P, R, gamma)
        # 탐욕 정책 pk의 진짜 가치 V^{pk}가 V*에서 1e-9보다 멀면 "아직 최적 아님". 마지막으로 아니었던 스윕을 기록
        if np.max(np.abs(m.evaluate_exact(pk, P, R, gamma) - V_star)) > 1e-9:
            last_bad = kk
    opt_at = last_bad + 1  # 그다음 스윕부터 계속 최적(허용오차 1e-9 기준)
    err_at_opt = hist[opt_at - 1][1]
    res["methods"]["VI"] = {"iterations": len(hist), "backups": len(hist) * S * A, "time_s": t,
                            "err": float(np.max(np.abs(V - V_star))), "policy_diff_cells": int((pi != pi_star).sum()),
                            "greedy_optimal_from_sweep": opt_at, "value_err_at_that_sweep": err_at_opt,
                            "sweeps_to_err_1e-6": next(i + 1 for i, (_, e) in enumerate(hist) if e < 1e-6)}
    for kk in (0, 1, 5, 20):
        (pi, V, it, b), t = timed(lambda: m.modified_policy_iteration(P, R, gamma, kk, theta=THETA), reps)
        res["methods"][f"MPI{kk}"] = {"iterations": it, "backups": b, "time_s": t,
                                      "err": float(np.max(np.abs(V - V_star))),
                                      "policy_diff_cells": int((pi != pi_star).sum())}
    # 같은 단위로 맞추기: 이 구현은 백업 하나가 조밀한 다음 칸 합(곱셈·덧셈 2S번)이다. 정책 반복은 직접 풀이 FLOP를 더한다.
    for v in res["methods"].values():
        # 백업 1번 = 조밀한 다음 칸 합 S개의 곱·덧셈 = 2S FLOP (이 구현 기준. 희소 구현이면 달라진다)
        v["dense_flops"] = int(v["backups"] * 2 * S + v.get("solve_flops", 0))
    return res, (P, R, term, V_star, pi_star)


def e4_gammas(n=10, gammas=(0.9, 0.99, 0.999), eps=1e-6):
    (P, R, term), _ = g.compare_grid(n)
    S = P.shape[0]
    rows = []
    for gamma in gammas:
        V_star, pi_star = optimal(P, R, gamma)
        V, hist = m.value_iteration(P, R, gamma, theta=1e-14, V_star=V_star)
        # 정답 오차 < eps 가 처음 되는 스윕 수(이 hist는 theta=1e-14로 돌려 eps 도달 전에 멈추지 않는다고 가정)
        k_err = next(i + 1 for i, (_, e) in enumerate(hist) if e < eps)
        Vk, last_bad = np.zeros(S), 0
        for kk in range(1, k_err + 1):
            Vk = m.bellman_T(Vk, P, R, gamma)  # 식: V_k = T V_{k-1}
            pk = m.greedy_policy(Vk, P, R, gamma)
            if np.max(np.abs(m.evaluate_exact(pk, P, R, gamma) - V_star)) > 1e-9:
                last_bad = kk
        pi, Vp, hist_pi, ok = m.policy_iteration(P, R, gamma)
        # 실제 오차가 줄어드는 비율: 마지막 구간의 기하평균
        es = [e for _, e in hist[:k_err]]
        rate = float((es[-1] / es[k_err // 2 - 1]) ** (1 / (k_err - k_err // 2))) if k_err > 4 else None
        keep = ~term
        Ppi, _ = m.policy_mats(pi_star, P, R)
        # 최적 정책의 비종료 칸 부분 행렬의 스펙트럼 반경 rho. 실제 오차 감소율은 대략 gamma * rho (3장과 같은 논리)
        rho = float(np.max(np.abs(np.linalg.eigvals(Ppi[np.ix_(keep, keep)]))))
        rows.append({"gamma": gamma, "rho_nonterminal_pistar": rho, "gamma_rho": gamma * rho, "vi_sweeps_to_eps": k_err, "theory_sweeps": m.theory_vi_sweeps(eps, gamma),
                     "theory_with_Vstar_norm": int(np.ceil(np.log(eps / np.max(np.abs(V_star))) / np.log(gamma))),
                     "Vstar_norm": float(np.max(np.abs(V_star))), "observed_rate": rate,
                     "greedy_optimal_from_sweep": last_bad + 1, "pi_iterations": len(hist_pi)})
    # 종료 칸이 없으면(E2의 3x3, 구석 +1) 오차가 정확히 gamma배씩 준다: 상한이 거의 그대로 맞는다
    P2, R2, _ = g.loop_grid()
    loop = []
    for gamma in gammas:
        V_star, _ = optimal(P2, R2, gamma)
        V, hist = m.value_iteration(P2, R2, gamma, theta=1e-14, V_star=V_star)
        k_err = next(i + 1 for i, (_, e) in enumerate(hist) if e < eps)
        es = [e for _, e in hist]
        loop.append({"gamma": gamma, "vi_sweeps_to_eps": k_err, "theory_sweeps": m.theory_vi_sweeps(eps, gamma),
                     "Vstar_norm": float(np.max(np.abs(V_star))), "observed_rate": float(es[k_err - 1] / es[k_err - 2])})
    return {"n": n, "eps": eps, "rows": rows, "loop_rows": loop}


def e5_monotone(P, R, gamma=0.99):
    S = P.shape[0]
    pi, V, hist, ok = m.policy_iteration(P, R, gamma)
    Vs = [h["V"] for h in hist]
    # 식: V^{pi_{k+1}} >= V^{pi_k} (정책 개선 정리). 칸별 증가량의 최솟값이 음수가 아니어야 한다(동률 허용폭 1e-9 안의 잡음 제외)
    mins = [float(np.min(Vs[i + 1] - Vs[i])) for i in range(len(Vs) - 1)]
    return {"V_start_per_iter": [round(float(v[0]), 6) for v in Vs], "min_increase_per_iter": mins,
            "mean_V_per_iter": [round(float(v.mean()), 6) for v in Vs]}


def e6_contraction(P, R, gamma=0.99, pairs=100):
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(pairs):
        V, U = rng.normal(0, 5, P.shape[0]), rng.normal(0, 5, P.shape[0])
        TV, TU = m.bellman_T(V, P, R, gamma), m.bellman_T(U, P, R, gamma)
        # 식: ||T V - T U||_inf / ||V - U||_inf <= gamma  (축약 비율. max_ratio가 gamma를 넘으면 안 된다)
        ratios.append(float(np.max(np.abs(TV - TU)) / np.max(np.abs(V - U))))
    return {"pairs": pairs, "gamma": gamma, "max_ratio": max(ratios), "min_ratio": min(ratios)}


def e7_match(P, R, gamma=0.99):
    Pt, Rt = torch.tensor(P), torch.tensor(R)
    pi, V, hist, _ = m.policy_iteration(P, R, gamma)
    pit, Vt, it = mt.policy_iteration(Pt, Rt, gamma)
    Vv, _ = m.value_iteration(P, R, gamma, theta=THETA)
    Vvt, kt = mt.value_iteration(Pt, Rt, gamma, theta=THETA)
    _, Vm, _, _ = m.modified_policy_iteration(P, R, gamma, 5, theta=THETA)
    _, Vmt, _ = mt.modified_policy_iteration(Pt, Rt, gamma, 5, theta=THETA)
    return {"PI_same_policy": bool(np.array_equal(pi, pit.numpy())), "PI_iters": [len(hist), it],
            "PI_max_abs_diff": float(np.max(np.abs(V - Vt.numpy()))),
            "VI_max_abs_diff": float(np.max(np.abs(Vv - Vvt.numpy()))),
            "MPI5_max_abs_diff": float(np.max(np.abs(Vm - Vmt.numpy())))}


def e8_rollouts(P, R, term, V_star, pi_star, gamma=0.99, seeds=range(10), episodes=100, max_steps=2000):
    log, rets, goals, lens = [], [], [], []
    goal = P.shape[0] - 1
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for ep in range(episodes):
            s, G, disc = 0, 0.0, 1.0
            for t in range(max_steps):
                a = int(pi_star[s])
                s2, r = g.step(s, a, P, R, rng)
                done = bool(term[s2])
                if sd == 0 and ep == 0:
                    log.append({"seed": sd, "episode": ep, "t": t, "s": s, "a": a, "r": r, "s_next": s2, "done": done,
                                "info": {}})
                # 식: G = sum_t gamma^t r_t  (할인 리턴. disc = gamma^t 를 매 걸음 곱해 간다). 평균 G가 V*(시작)와 맞는지 비교
                G += disc * r
                disc *= gamma
                s = s2
                if done:
                    break
            rets.append(G)
            goals.append(s == goal)
            lens.append(t + 1)
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log))
    rets = np.array(rets)
    return {"episodes": len(rets), "mean_return": float(rets.mean()), "stderr": float(rets.std(ddof=1) / np.sqrt(len(rets))),
            "V_star_start": float(V_star[0]), "goal_rate": float(np.mean(goals)), "mean_len": float(np.mean(lens)),
            "logged_steps": len(log)}


def e9_frames(gamma=0.95):
    P, R, term = g.tie_grid(6)
    pi = np.zeros(36, dtype=np.int64)
    frames = [arrows(pi, term, 6)]
    for it in range(1, 50):
        V = m.evaluate_exact(pi, P, R, gamma)  # 평가: 식 (I - gamma P^pi) V = r^pi
        new = m.greedy_policy(V, P, R, gamma, tie="first")  # 개선: 식 pi'(s) = argmax_a Q^pi(s,a)
        frames.append(arrows(new, term, 6))  # 바뀌지 않은 마지막 반복의 프레임도 저장된다
        if np.array_equal(new, pi):
            break
        pi = new
    V_star, _ = optimal(P, R, gamma)
    return {"frames": frames, "V_star": np.round(V_star.reshape(6, 6), 4).tolist()}


def main():
    t0 = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__.split("+")[0],
                   "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
                   "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}}
    res["E0"] = e0_corridor()
    res["E1"] = e1_ties()
    res["E2"] = e2_gamma_one()
    res["E3"], (P, R, term, V_star, pi_star) = e3_compare()
    res["E4"] = e4_gammas()
    res["E5"] = e5_monotone(P, R)
    res["E6"] = e6_contraction(P, R)
    res["E7"] = e7_match(P, R)
    res["E8"] = e8_rollouts(P, R, term, V_star, pi_star)
    res["E9"] = e9_frames()
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("E1", "E2", "E9")}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
