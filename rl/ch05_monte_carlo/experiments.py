"""강화학습 5장 실험. `cd math-ai-matrix-labs && uv run python rl/ch05_monte_carlo/experiments.py`

E0 손계산: 3장의 복도 [A][B][도착], 반반 정책, gamma 0.9. A에서 출발한 판 몇 개의 리턴과 N판 평균
E1 실패 1(기다림): 판이 끝나지 않으면 MC는 아무것도 못 고친다
   a) 2장 격자(5x5) 균등 무작위 정책의 판 길이   b) 무작위 결정론적 정책 1,000개, 상한 10,000걸음
   c) Q=0에서 시작한 탐욕(epsilon=0) MC 제어     d) 같은 정책 1,000개를 epsilon=0.1로 섞었을 때
   e) 20x20 격자 균등 무작위 정책의 판 길이(긴 꼬리)
E2 첫 방문 vs 모든 방문: 4x4 격자, 균등 무작위, gamma 0.9. RMSE(판 수 함수, 시드 10개)와 판 하나일 때의 편향
E3 MC 제어(epsilon-soft, epsilon 0.1): 2장 격자, gamma 0.9, 시드 10개 x 2,000판
E4 실패 2(분산): 상태 하나짜리 문제. 행동 정책 b("한 번 더" 0.5), 목표 정책 pi("한 번 더" p = 0.6, 0.7, 0.9)
   일반 IS vs 가중 IS, 판 수 1e2~1e5, 시드 100개. 복도에서의 오프폴리시 확인도 함께
E5 구현 대조: NumPy vs PyTorch
E6 계산량: 판당 O(T) 시간, 메모리(한 판 전체 저장)

결과는 results/ch05.json, 로그는 logs/ch05_episodes.jsonl.
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
import rl_ch05_grid as g  # noqa: E402
import rl_ch05_mc_np as m  # noqa: E402
import rl_ch05_mc_torch as mt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch05.json"
LOG = HERE / "logs" / "ch05_episodes.jsonl"
SEEDS = list(range(10))
CAP = 10_000


def r6(x):
    return float(round(float(x), 6))


def stats(a):
    a = np.asarray(a, dtype=float)
    return {
        "mean": r6(a.mean()), "median": r6(np.median(a)), "p90": r6(np.percentile(a, 90)),
        "p99": r6(np.percentile(a, 99)), "max": r6(a.max()), "min": r6(a.min()),
    }


def log_episode(f, ep, seed, episode, env_name, gamma, keep=None):
    """공통 스키마 {"seed","episode","t","s","a","r","s_next","done","info"}로 한 판을 적는다.

    keep=(앞 n, 뒤 m)이면 긴 판의 앞뒤만 적는다(잘린 판 기록용).
    """
    T = len(ep.states)
    ts = range(T)
    if keep is not None and T > sum(keep):
        ts = list(range(keep[0])) + list(range(T - keep[1], T))
    for t in ts:
        s_next = ep.states[t + 1] if t + 1 < T else ep.final
        last = t == T - 1
        info = {"env": env_name, "gamma": gamma}
        if last and not ep.done:
            info["truncated"] = True
            info["note"] = f"상한 {T}걸음에서 잘림, 리턴을 모름"
        rec = {"seed": seed, "episode": episode, "t": t, "s": ep.states[t], "a": ep.actions[t],
               "r": ep.rewards[t], "s_next": s_next, "done": bool(last and ep.done), "info": info}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def path_letters(ep):
    names = "AB"
    return "".join(names[s] for s in ep.states) + "◎"


# ---------------- E0 ----------------
def e0_corridor(logf):
    P, R, term, pi = g.corridor()
    env = m.make_env(P, R, term)
    gamma = 0.9
    V = g.exact_v(P, R, pi, gamma, term)
    rng = np.random.default_rng(0)
    first = []
    for i in range(4):
        ep = m.generate_episode(env, pi, 0, rng)
        G0 = float(m.returns(ep.rewards, gamma)[0])
        first.append({"path": path_letters(ep), "steps": len(ep.states), "G0": r6(G0),
                      "G0_formula": f"0.9^{len(ep.states) - 1}"})
        log_episode(logf, ep, 0, i, "corridor", gamma)
    hand_avg = np.mean([f["G0"] for f in first])
    Ns = [10, 100, 1000, 10000]
    per_N = {}
    for N in Ns:
        ests = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            uni = m._Uniforms(rng)
            pc = m.policy_cdf(pi)
            tot = 0.0
            for _ in range(N):
                ep = m.generate_episode(env, pi, 0, rng, uni=uni, pcdf=pc)
                tot += gamma ** (len(ep.states) - 1)
            ests.append(tot / N)
        a = np.array(ests)
        per_N[str(N)] = {"mean": r6(a.mean()), "min": r6(a.min()), "max": r6(a.max()),
                         "sd_across_seeds": r6(a.std(ddof=1)),
                         "max_abs_err": r6(np.max(np.abs(a - V[0])))}
    # 판 하나의 리턴이 얼마나 들쭉날쭉한지(시드 123, 10만 판)
    rng = np.random.default_rng(123)
    uni = m._Uniforms(rng)
    gs = np.array([gamma ** (len(m.generate_episode(env, pi, 0, rng, uni=uni).states) - 1) for _ in range(100_000)])
    return {
        "gamma": gamma, "V_exact": {"A": r6(V[0]), "B": r6(V[1])},
        "first_episodes_seed0": first, "hand_average_4": r6(hand_avg),
        "average_over_N": per_N,
        "single_return_sd_100k": r6(gs.std(ddof=1)), "single_return_mean_100k": r6(gs.mean()),
        "steps_mean_100k": r6(np.mean(np.log(gs) / np.log(gamma) + 1)),
    }


# ---------------- E1 ----------------
def nbr_table(P):
    """미끄러짐 없는 격자에서 (s, a) -> 다음 칸."""
    return P.argmax(axis=2)


def e1_waiting(logf):
    out = {"cap": CAP}
    P, R, term = g.ch02_grid()
    env = m.make_env(P, R, term)
    S = P.shape[0]
    gamma = 0.9
    # a) 균등 무작위 정책
    lens = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        uni = m._Uniforms(rng)
        pi = g.uniform_policy(S)
        pc = m.policy_cdf(pi)
        for _ in range(1000):
            lens.append(len(m.generate_episode(env, pi, g.CH02_START, rng, CAP, uni=uni, pcdf=pc).states))
    out["a_uniform_ch02"] = {"episodes": len(lens), **stats(lens), "truncated": int(np.sum(np.array(lens) >= CAP))}
    # b) 무작위 결정론적 정책 1,000개
    rng = np.random.default_rng(0)
    n_pol = 1000
    acts = rng.integers(0, 4, size=(n_pol, S))
    stuck, done_lens, logged = 0, [], False
    t0 = time.perf_counter()
    stuck_ids = []
    for i in range(n_pol):
        pi = np.zeros((S, 4))
        pi[np.arange(S), acts[i]] = 1.0
        ep = m.generate_episode(env, pi, g.CH02_START, rng, CAP)
        if ep.done:
            done_lens.append(len(ep.states))
        else:
            stuck += 1
            stuck_ids.append(i)
            if not logged:
                log_episode(logf, ep, 0, i, "ch02_grid_random_deterministic_policy", gamma, keep=(6, 1))
                logged = True
                seen = []
                for s in ep.states[:40]:
                    if s in seen:
                        break
                    seen.append(s)
                first_stuck = {"policy_index": i, "visited_before_repeat": [int(x) for x in seen],
                               "steps_logged": CAP}
    t_b = time.perf_counter() - t0
    # 같은 묶음에 첫 방문 MC를 걸면 몇 칸이 고쳐지나
    out["b_random_deterministic"] = {
        "policies": n_pol, "never_terminated": stuck, "share_never": r6(stuck / n_pol),
        "terminated_lengths": stats(done_lens) if done_lens else None,
        "first_stuck_example": first_stuck, "seconds": r6(t_b),
        "steps_spent_on_stuck": stuck * CAP,
    }
    # c) Q=0, 탐욕(epsilon=0) MC 제어: 동률이면 번호가 가장 작은 행동(위)
    rng = np.random.default_rng(0)
    Q, pi_c, lens_c, trunc_c = m.mc_control_eps_soft(env, g.CH02_START, gamma, 0.0, 5, rng, CAP)
    out["c_greedy_control_eps0"] = {"episodes": 5, "truncated": trunc_c, "lengths": lens_c,
                                    "Q_nonzero_entries": int(np.count_nonzero(Q)),
                                    "first_action_at_start": int(pi_c[g.CH02_START].argmax())}
    # d) 같은 1,000개 정책을 epsilon=0.1로 섞기
    rng = np.random.default_rng(1)
    uni = m._Uniforms(rng)
    lens_d, trunc_d = [], 0
    lens_d_stuck = []
    for i in range(n_pol):
        pi = np.full((S, 4), 0.1 / 4)
        pi[np.arange(S), acts[i]] += 0.9
        ep = m.generate_episode(env, pi, g.CH02_START, rng, CAP, uni=uni)
        lens_d.append(len(ep.states))
        if i in set(stuck_ids):
            lens_d_stuck.append(len(ep.states))
        trunc_d += int(not ep.done)
    out["d_eps_soft_0.1"] = {"policies": n_pol, "truncated": trunc_d, "lengths_all": stats(lens_d),
                             "lengths_for_previously_stuck": stats(lens_d_stuck)}
    # e) 20x20 균등 무작위
    P20, R20, term20 = g.corner_grid(20)
    nbr = nbr_table(P20)
    L, alive = [], []
    for sd in SEEDS:
        l, a = m.random_walk_lengths(200, np.random.default_rng(sd), nbr, term20, 0, CAP)
        L.append(l)
        alive.append(a)
    L, alive = np.concatenate(L), np.concatenate(alive)
    # 정확한 기대 걸음 수: 걸음마다 1을 더하는 보상, gamma=1
    steps_R = np.where(P20 > 0, 1.0, 0.0)
    exp_steps = g.exact_v(P20, steps_R, g.uniform_policy(400), 1.0, term20)[0]
    out["e_uniform_20x20"] = {"episodes": int(L.size), **stats(L), "truncated": int(alive.sum()),
                              "share_truncated": r6(alive.mean()), "expected_steps_exact": r6(exp_steps)}
    P5, R5, term5 = g.corner_grid(5)
    steps_R5 = np.where(P5 > 0, 1.0, 0.0)
    out["e_uniform_5x5_expected_steps"] = r6(g.exact_v(P5, steps_R5, g.uniform_policy(25), 1.0, term5)[0])
    return out


# ---------------- E2 ----------------
def e2_first_vs_every():
    n = 4
    P, R, term = g.corner_grid(n)
    env = m.make_env(P, R, term)
    S = P.shape[0]
    gamma = 0.9
    pi = g.uniform_policy(S)
    V = g.exact_v(P, R, pi, gamma, term)
    steps_R = np.where(P > 0, 1.0, 0.0)
    exp_len = g.exact_v(P, steps_R, pi, 1.0, term)[0]
    nonterm = ~term
    Ns = [10, 100, 1000, 10000]
    res = {str(N): {"fv": [], "ev": [], "fv_unvisited": [], "ev_start": [], "fv_start": []} for N in Ns}
    t_gen = t_fv = t_ev = 0.0
    total_steps = 0
    max_len = 0
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        uni = m._Uniforms(rng)
        pc = m.policy_cdf(pi)
        t = time.perf_counter()
        eps = [m.generate_episode(env, pi, 0, rng, CAP, uni=uni, pcdf=pc) for _ in range(Ns[-1])]
        t_gen += time.perf_counter() - t
        total_steps += sum(len(e.states) for e in eps)
        max_len = max(max_len, max(len(e.states) for e in eps))
        for N in Ns:
            t = time.perf_counter()
            Vf, Nf, _ = m.mc_first_visit(eps[:N], gamma, S)
            t1 = time.perf_counter()
            Ve, _, _ = m.mc_every_visit(eps[:N], gamma, S)
            t2 = time.perf_counter()
            if N == Ns[-1]:
                t_fv += t1 - t
                t_ev += t2 - t1
            d = res[str(N)]
            d["fv"].append(float(np.sqrt(np.mean((Vf[nonterm] - V[nonterm]) ** 2))))
            d["ev"].append(float(np.sqrt(np.mean((Ve[nonterm] - V[nonterm]) ** 2))))
            d["fv_unvisited"].append(int(np.sum(Nf[nonterm] == 0)))
            d["fv_start"].append(float(Vf[0]))
            d["ev_start"].append(float(Ve[0]))
    table = {}
    for N in Ns:
        d = res[str(N)]
        table[str(N)] = {
            "rmse_first_mean": r6(np.mean(d["fv"])), "rmse_first_sd": r6(np.std(d["fv"], ddof=1)),
            "rmse_every_mean": r6(np.mean(d["ev"])), "rmse_every_sd": r6(np.std(d["ev"], ddof=1)),
            "every_lower_in_seeds": int(np.sum(np.array(d["ev"]) < np.array(d["fv"]))),
            "unvisited_states_max": int(max(d["fv_unvisited"])),
        }
    # 판이 하나·열 개일 때의 편향: 시작 칸 추정값의 평균 - 참값 (2,000번 반복)
    bias = {}
    for N in (1, 10):
        fs, es = [], []
        rng = np.random.default_rng(1000 + N)
        uni = m._Uniforms(rng)
        pc = m.policy_cdf(pi)
        for _ in range(2000):
            eps = [m.generate_episode(env, pi, 0, rng, CAP, uni=uni, pcdf=pc) for _ in range(N)]
            fs.append(m.mc_first_visit(eps, gamma, S)[0][0])
            es.append(m.mc_every_visit(eps, gamma, S)[0][0])
        fs, es = np.array(fs), np.array(es)
        bias[str(N)] = {
            "first_mean_minus_true": r6(fs.mean() - V[0]), "first_se": r6(fs.std(ddof=1) / np.sqrt(fs.size)),
            "every_mean_minus_true": r6(es.mean() - V[0]), "every_se": r6(es.std(ddof=1) / np.sqrt(es.size)),
            "first_rmse": r6(np.sqrt(np.mean((fs - V[0]) ** 2))), "every_rmse": r6(np.sqrt(np.mean((es - V[0]) ** 2))),
        }
    return {
        "grid": "4x4, 오른쪽 아래 +1 종료, 균등 무작위, 시작 (0,0)", "gamma": gamma,
        "V_start_exact": r6(V[0]), "expected_length_exact": r6(exp_len), "rmse_over": "종료 칸을 뺀 15칸, 방문 못 한 칸은 0으로",
        "by_N": table, "start_state_bias": bias, "repeats_for_bias": 2000,
        "cost": {"episodes": len(SEEDS) * Ns[-1], "total_steps": int(total_steps), "max_episode_len": int(max_len),
                 "seconds_generate": r6(t_gen), "seconds_first_visit_10k_all_seeds": r6(t_fv),
                 "seconds_every_visit_10k_all_seeds": r6(t_ev),
                 "us_per_step_first_visit": r6(1e6 * t_fv / total_steps)},
    }


# ---------------- E3 ----------------
def greedy_return(P, R, term, Q, s0, gamma, max_steps=100):
    s, G, disc, path = s0, 0.0, 1.0, []
    for _ in range(max_steps):
        if term[s]:
            return G, len(path), True
        a = int(Q[s].argmax())
        s2 = int(P[s, a].argmax())
        G += disc * R[s, a, s2]
        disc *= gamma
        path.append(s)
        s = s2
    return G, len(path), False


def e3_control():
    """MC 제어 네 설정 x 시드 10개 x 2,000판. 끝 상태의 탐욕 경로가 +10에 닿는지(시작 칸 가치 5.9049) 본다."""
    P, R, term = g.ch02_grid()
    env = m.make_env(P, R, term)
    gamma, n_ep = 0.9, 2000
    opt = 10 * 0.9 ** 5
    configs = [("fixed_start_eps0.1", False, 0.1), ("fixed_start_eps0.5", False, 0.5),
               ("exploring_starts_eps0", True, 0.0), ("exploring_starts_eps0.1", True, 0.1),
               ("fixed_start_eps0.1_10000ep", False, 0.1)]
    out = {"grid": "2장 5x5", "gamma": gamma, "episodes": n_ep, "optimal_start_value": r6(opt), "configs": {}}
    for name, es, eps in configs:
        n_ep = 10_000 if name.endswith("10000ep") else 2000
        rows = []
        t = time.perf_counter()
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            Q, _, lens, trunc = m.mc_control_eps_soft(env, g.CH02_START, gamma, eps, n_ep, rng, CAP, exploring_starts=es)
            Gg, steps, ok = greedy_return(P, R, term, Q, g.CH02_START, gamma)
            rows.append({"seed": sd, "greedy_return": r6(Gg), "greedy_terminates": ok, "truncated": trunc,
                         "total_steps": int(sum(lens))})
        out["configs"][name] = {
            "exploring_starts": es, "epsilon": eps,
            "seeds_reaching_10": int(sum(abs(r["greedy_return"] - opt) < 1e-9 for r in rows)),
            "seeds_settling_1": int(sum(abs(r["greedy_return"] - 1.0) < 1e-9 for r in rows)),
            "seeds_greedy_not_terminating": int(sum(not r["greedy_terminates"] for r in rows)),
            "truncated_total": int(sum(r["truncated"] for r in rows)),
            "truncated_min": int(min(r["truncated"] for r in rows)), "truncated_max": int(max(r["truncated"] for r in rows)),
            "steps_total": int(sum(r["total_steps"] for r in rows)), "seconds": r6(time.perf_counter() - t),
            "episodes": n_ep, "seeds": rows,
        }
    return out


# ---------------- E4 ----------------
def e4_importance():
    Ns = [100, 1000, 10000, 100000]
    out = {"b_more": 0.5, "Ns": Ns, "seeds": 100, "cases": {}}
    for p in (0.6, 0.65, 0.7, 0.9):
        truth = m.one_state_truth(p)
        o_all, w_all = [], []
        max_rho = 0.0
        pooled = []
        for sd in range(100):
            k = m.sample_more_counts(Ns[-1], np.random.default_rng(10_000 + sd))
            o, w = m.is_estimates_prefix(k, p, Ns)
            o_all.append(o)
            w_all.append(w)
            rho = m.rho_of_k(k, p)
            max_rho = max(max_rho, float(rho.max()))
            if sd == 0:
                pooled = rho * k
        o_all, w_all = np.array(o_all), np.array(w_all)
        rows = {}
        for j, N in enumerate(Ns):
            o, w = o_all[:, j], w_all[:, j]
            rows[str(N)] = {
                "ordinary_mean": r6(o.mean()), "ordinary_sd": r6(o.std(ddof=1)), "ordinary_median": r6(np.median(o)),
                "ordinary_within10": int(np.sum(np.abs(o - truth) <= 0.1 * truth)),
                "ordinary_min": r6(o.min()), "ordinary_max": r6(o.max()),
                "weighted_mean": r6(w.mean()), "weighted_sd": r6(w.std(ddof=1)), "weighted_median": r6(np.median(w)),
                "weighted_within10": int(np.sum(np.abs(w - truth) <= 0.1 * truth)),
            }
        sm = m.is_second_moment(p)
        out["cases"][str(p)] = {
            "truth": r6(truth), "x_ratio_sq_over_b": r6(p * p / 0.5),
            "second_moment_exact": None if np.isinf(sm) else r6(sm),
            "variance_exact": None if np.isinf(sm) else r6(sm - truth**2),
            "variance_infinite": bool(np.isinf(sm)),
            "rho_formula": f"({p}/0.5)^k * ({1 - p:.2g}/0.5)",
            "rho_k0": r6(m.rho_of_k(0, p)), "rho_k10": r6(m.rho_of_k(10, p)), "rho_k20": r6(m.rho_of_k(20, p)),
            "max_rho_seen": r6(max_rho),
            "seed0_sample_var_rhoG_1e5": r6(np.var(pooled, ddof=1)),
            "by_N": rows,
        }
    # 첫 화면·2절 손계산용: 행동 정책에서 k = 0, 1, 2, 3 이 나올 확률과 p=0.9의 비율
    out["hand"] = {str(k): {"prob_b": float(0.5 ** (k + 1)), "rho_p0.9": r6(m.rho_of_k(k, 0.9)),
                            "rhoG_p0.9": r6(m.rho_of_k(k, 0.9) * k)} for k in (0, 1, 2, 3, 20)}
    # 행동 정책 그대로 평균 내면(IS 없이) b의 값 1에 모인다
    k = m.sample_more_counts(100000, np.random.default_rng(7))
    out["no_correction_mean_1e5"] = r6(k.mean())
    # 복도에서의 오프폴리시: b = 반반, pi = 오른쪽 0.8, A에서 출발, gamma 0.9
    P, R, term, b = g.corridor()
    pi = np.tile([0.0, 0.8, 0.0, 0.2], (3, 1))
    env = m.make_env(P, R, term)
    Vpi = g.exact_v(P, R, pi, 0.9, term)
    corr = {}
    for N in (100, 1000, 10000):
        o, w = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(500 + sd)
            uni = m._Uniforms(rng)
            pc = m.policy_cdf(b)
            eps = [m.generate_episode(env, b, 0, rng, CAP, uni=uni, pcdf=pc) for _ in range(N)]
            o.append(m.off_policy_is(eps, pi, b, 0.9, False))
            w.append(m.off_policy_is(eps, pi, b, 0.9, True))
        corr[str(N)] = {"ordinary_mean": r6(np.mean(o)), "ordinary_sd": r6(np.std(o, ddof=1)),
                        "weighted_mean": r6(np.mean(w)), "weighted_sd": r6(np.std(w, ddof=1))}
    out["corridor_offpolicy"] = {"V_pi_A_exact": r6(Vpi[0]), "V_b_A_exact": r6(g.exact_v(P, R, b, 0.9, term)[0]),
                                 "pi_right": 0.8, "by_N": corr}
    return out


# ---------------- E5 ----------------
def e5_torch():
    out = {}
    P, R, term = g.corner_grid(4)
    env = m.make_env(P, R, term)
    pi = g.uniform_policy(16)
    rng = np.random.default_rng(0)
    eps = [m.generate_episode(env, pi, 0, rng, CAP) for _ in range(500)]
    Vn, Nn, _ = m.mc_first_visit(eps, 0.9, 16)
    Vt, Nt = mt.mc_first_visit(eps, 0.9, 16)
    out["first_visit_4x4_500ep"] = {"max_abs_diff": float(np.max(np.abs(Vn - Vt.numpy()))),
                                    "N_equal": bool(np.array_equal(Nn, Nt.numpy())), "atol": 1e-12}
    Pc, Rc, termc, b = g.corridor()
    envc = m.make_env(Pc, Rc, termc)
    pic = np.tile([0.0, 0.8, 0.0, 0.2], (3, 1))
    rng = np.random.default_rng(1)
    epc = [m.generate_episode(envc, b, 0, rng, CAP) for _ in range(2000)]
    on, wn = m.off_policy_is(epc, pic, b, 0.9, False), m.off_policy_is(epc, pic, b, 0.9, True)
    ot, wt = mt.off_policy_is(epc, torch.tensor(pic), torch.tensor(b), 0.9)
    rho_n = np.array([m.importance_ratio(e, pic, b) for e in epc if e.done])
    rho_t = mt.importance_ratios(epc, torch.tensor(pic), torch.tensor(b)).numpy()
    out["is_corridor_2000ep"] = {"ordinary_rel_diff": float(abs(on - ot.item()) / abs(on)),
                                 "weighted_rel_diff": float(abs(wn - wt.item()) / abs(wn)),
                                 "rho_max_rel_diff": float(np.max(np.abs(rho_n - rho_t) / rho_n)), "rtol": 1e-12}
    k = m.sample_more_counts(100000, np.random.default_rng(10_000))
    Ns = [100, 1000, 10000, 100000]
    on, wn = m.is_estimates_prefix(k, 0.9, Ns)
    ot, wt = mt.is_estimates_prefix(k, 0.9, Ns)
    out["is_one_state_p0.9"] = {"ordinary_max_rel_diff": float(np.max(np.abs(on - ot.numpy()) / np.abs(on))),
                                "weighted_max_rel_diff": float(np.max(np.abs(wn - wt.numpy()) / np.abs(wn))),
                                "rtol": 1e-12}
    return out


def main():
    t0 = time.perf_counter()
    LOG.parent.mkdir(exist_ok=True)
    OUT.parent.mkdir(exist_ok=True)
    with LOG.open("w") as logf:
        res = {"E0_corridor": e0_corridor(logf), "E1_waiting": e1_waiting(logf)}
    res["E2_first_vs_every"] = e2_first_vs_every()
    res["E3_control"] = e3_control()
    res["E4_importance"] = e4_importance()
    res["E5_torch"] = e5_torch()
    res["env"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "machine": f"{platform.system()} {platform.machine()}", "threads": 2,
                  "total_seconds": r6(time.perf_counter() - t0)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
