"""강화학습 10장 실험. `cd math-ai-matrix-labs && uv run python rl/ch10_policy_gradient/experiments.py`

E0 손계산: 짧은 복도에서 p = 0.5일 때의 칸 가치, J(p) 표, 가장 좋은 p, 가장 짧은 판(오른쪽-왼쪽-오른쪽) 하나로 만든 기울기 추정
E1 실패(가치 기반): 칸을 못 보는 Q 두 개로는 오른쪽 확률을 eps/2 또는 1 - eps/2 둘 중 하나로만 고를 수 있다
   a) 결정론적 정책 두 개와 epsilon-탐욕 정책 두 개의 정확한 J   b) epsilon-탐욕 Sarsa를 시드 10개로 돌려 끝난 곳
E2 실패(흔들림): 같은 theta에서 REINFORCE 기울기 추정 1,000개 x 시드 10개. 베이스라인 없음 / 상수 / 칸 가치(이상적)
E3 복도 학습: theta0 = -2에서 REINFORCE(베이스라인 없음 vs 배운 숫자 하나), 보폭은 시드 100~104로 고르고 시드 0~9로 보고
E4 cart-pole: REINFORCE, REINFORCE + 가치 베이스라인, one-step actor-critic. 보폭 격자는 시드 100~102로 고르고 시드 0~9로 보고
E5 구현 대조: NumPy 손 기울기 vs PyTorch autograd, 정확한 기울기 vs 유한 차분
E6 계산량: 환경 걸음당 시간, 저장해야 하는 양

결과는 results/ch10.json, 로그는 logs/ch10_episodes.jsonl.
"""

import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch10_cartpole as cp  # noqa: E402
import rl_ch10_corridor as cor  # noqa: E402
import rl_ch10_pg_np as m  # noqa: E402
import rl_ch10_pg_torch as mt  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch10.json"
LOG = HERE / "logs" / "ch10_episodes.jsonl"
SEEDS = list(range(10))
SWEEP_SEEDS_COR = [100, 101, 102, 103, 104]
SWEEP_SEEDS_CP = [100, 101, 102]


def r6(x):
    x = float(x)
    return float(round(x, 6)) if np.isfinite(x) else None  # NaN, inf는 엄격한 JSON에서 읽히도록 null로


def log_rows(f, seed, episode, env, states, actions, rewards, done, info=None, s_names=None):
    """공통 스키마 {"seed","episode","t","s","a","r","s_next","done","info"}로 한 판을 적는다."""
    T = len(actions)
    for t in range(T):
        s = states[t]
        s_next = states[t + 1] if t + 1 < len(states) else None
        last = t == T - 1
        fmt = (lambda v: s_names[v] if s_names else [round(float(x), 4) for x in v])
        row = {
            "seed": seed, "episode": episode, "t": t,
            "s": fmt(s), "a": int(actions[t]), "r": float(rewards[t]),
            "s_next": ("도착" if (done and last and s_names) else (fmt(s_next) if s_next is not None else None)),
            "done": bool(done and last), "info": dict(info or {}, env=env),
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ E0

def e0_hand():
    V = cor.exact_v(0.5)
    ps = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]
    table = {str(p): r6(cor.exact_v(p)[0]) for p in ps}
    pb = cor.best_p()
    # 가장 짧은 판: A -오른쪽-> B -왼쪽-> C -오른쪽-> 도착 (3걸음). 확률 0.5^3
    ep = m.Episode(states=[0, 1, 2], actions=[1, 0, 1], rewards=[-1.0, -1.0, -1.0], done=True)
    G = m.returns(ep.rewards, 1.0).tolist()
    score = [a - 0.5 for a in ep.actions]
    est = {b: r6(m.corridor_grad_estimate(ep, 0.0, b)) for b in ("none", "const", "value")}
    return {
        "env": "Sutton-Barto 2판 예제 13.1의 짧은 복도 설정(보상 -1/걸음, 할인 없음)",
        "V_at_p05": {"A": r6(V[0]), "B": r6(V[1]), "C": r6(V[2])},
        "J_of_p": table,
        "best_p": r6(pb), "best_p_closed_form": "2 - sqrt(2)", "best_J": r6(cor.exact_v(pb)[0]),
        "exact_slope_dJ_dtheta_at_theta0": r6(cor.exact_grad(0.0)),
        "exact_slope_dJ_dp_at_p05": r6(cor.exact_dJ_dp(0.5)[0]),
        "shortest_episode": {
            "path": "A-오른쪽->B-왼쪽->C-오른쪽->도착", "prob": 0.125, "G": G, "score_a_minus_p": score,
            "estimate_none": est["none"], "estimate_const_b_minus12": est["const"], "estimate_value_b_V": est["value"],
        },
    }


# ------------------------------------------------------------------ E1

def e1_value_based():
    det = {"always_right_p1": "끝나지 않음(A와 B를 오감)", "always_left_p0": "끝나지 않음(A에서 벽)"}
    # 시뮬레이션으로도 확인: 상한 1,000걸음에서 잘린다
    rng = np.random.default_rng(0)
    sim = {"p1_truncated_at_1000": not m.corridor_episode(40.0, rng, 1000).done,
           "p0_truncated_at_1000": not m.corridor_episode(-40.0, rng, 1000).done}
    eps_table = {"eps0.1_mostly_right_p0.95": r6(cor.exact_v(0.95)[0]),
                 "eps0.1_mostly_left_p0.05": r6(cor.exact_v(0.05)[0])}
    runs = []
    for sd in SEEDS:
        Q, p_right, L = m.corridor_sarsa_aliased(2000, sd, eps=0.1, alpha=0.01)
        runs.append({"seed": sd, "Q_left": r6(Q[0]), "Q_right": r6(Q[1]), "p_right": p_right,
                     "J_final_policy": r6(cor.exact_v(p_right)[0]), "mean_len_last200": r6(L[-200:].mean())})
    n_right = sum(r["p_right"] > 0.5 for r in runs)
    return {"deterministic": det, "simulated": sim, "eps_greedy_exact": eps_table,
            "sarsa_aliased": {"eps": 0.1, "alpha": 0.01, "episodes": 2000, "runs": runs,
                              "n_seeds_ended_mostly_right": n_right, "n_seeds_ended_mostly_left": len(runs) - n_right}}


# ------------------------------------------------------------------ E2

def e2_variance():
    out = {}
    for th in (0.0, -1.0):
        per = {b: {"means": [], "sds": []} for b in ("none", "const", "value")}
        all_g = {b: [] for b in per}
        trunc = 0
        lens = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            eps = [m.corridor_episode(th, rng, cap=1000) for _ in range(1000)]
            trunc += sum(not e.done for e in eps)
            lens += [len(e.actions) for e in eps]
            for b in per:
                g = np.array([m.corridor_grad_estimate(e, th, b) for e in eps])
                per[b]["means"].append(g.mean())
                per[b]["sds"].append(g.std(ddof=1))
                all_g[b].append(g)
        exact = cor.exact_grad(th)
        res = {"theta": th, "p": r6(cor.sigmoid(th)), "J": r6(cor.exact_J(th)), "exact_grad": r6(exact),
               "samples_per_seed": 1000, "seeds": len(SEEDS), "truncated_episodes": trunc,
               "episode_len_mean": r6(np.mean(lens)), "episode_len_max": int(np.max(lens))}
        sd_none = np.mean(per["none"]["sds"])
        for b in per:
            g = np.concatenate(all_g[b])
            se = g.std(ddof=1) / np.sqrt(len(g))
            sds = np.array(per[b]["sds"])
            res[b] = {
                "pooled_mean": r6(g.mean()), "pooled_se": r6(se), "z_vs_exact": r6((g.mean() - exact) / se),
                "sd_mean_over_seeds": r6(sds.mean()), "sd_min": r6(sds.min()), "sd_max": r6(sds.max()),
                "seed_mean_min": r6(np.min(per[b]["means"])), "seed_mean_max": r6(np.max(per[b]["means"])),
                "variance_ratio_vs_none": r6((sds.mean() / sd_none) ** 2),
                "episodes_for_se_0.1": int(np.ceil((sds.mean() / 0.1) ** 2)),
            }
        out[f"theta_{th:g}"] = res
    return out


# ------------------------------------------------------------------ E3

def e3_corridor_learning():
    theta0, n_ep = -2.0, 1000
    alphas = [2.0 ** k for k in range(-16, -5)]
    sweep = {"none": {}, "learned": {}}
    for a in alphas:
        Js = [m.corridor_reinforce(theta0, a, n_ep, sd, "none")[1].mean() for sd in SWEEP_SEEDS_COR]
        sweep["none"][f"2^{int(np.log2(a))}"] = r6(np.mean(Js))
        Js = [m.corridor_reinforce(theta0, a, n_ep, sd, "learned", alpha_w=2.0 ** -6)[1].mean() for sd in SWEEP_SEEDS_COR]
        sweep["learned"][f"2^{int(np.log2(a))}"] = r6(np.mean(Js))
    best = {k: max(v, key=lambda q, v=v: v[q] if v[q] is not None else -np.inf) for k, v in sweep.items()}
    report = {}
    checkpoints = [10, 100, 300, 1000]
    for k in ("none", "learned"):
        a = 2.0 ** int(best[k].split("^")[1])
        curves, thetas = [], []
        for sd in SEEDS:
            th, J, _ = m.corridor_reinforce(theta0, a, n_ep, sd, k, alpha_w=2.0 ** -6)
            curves.append(J)
            thetas.append(th[-1])
        C = np.array(curves)
        report[k] = {"alpha": best[k], "alpha_w": "2^-6" if k == "learned" else None,
                     "J_at": {str(c): {"mean": r6(C[:, c - 1].mean()), "min": r6(C[:, c - 1].min()),
                                       "max": r6(C[:, c - 1].max())} for c in checkpoints},
                     "final_p": [r6(cor.sigmoid(t)) for t in thetas],
                     "mean_J_over_all_episodes": r6(C.mean())}
    # 보폭을 한 칸(두 배) 키운 2^-7, 베이스라인 없음: 어느 시드에서 무너졌고 돌아왔나
    collapse = []
    for sd in SWEEP_SEEDS_COR:
        th, J, _ = m.corridor_reinforce(theta0, 2.0 ** -7, n_ep, sd, "none")
        bad = np.nonzero(J < -1000)[0]
        collapse.append({"seed": sd, "episodes_J_below_-1000": int(len(bad)),
                         "first_bad_episode": int(bad[0]) + 1 if len(bad) else None,
                         "last_bad_episode": int(bad[-1]) + 1 if len(bad) else None,
                         "min_p": r6(cor.sigmoid(th.min())), "max_p": r6(cor.sigmoid(th.max())),
                         "final_J": r6(J[-1])})
    return {"theta0": theta0, "p0": r6(cor.sigmoid(theta0)), "J0": r6(cor.exact_J(theta0)), "episodes": n_ep,
            "collapse_none_alpha_2^-7": collapse,
            "best_J": r6(cor.exact_v(cor.best_p())[0]),
            "sweep_seeds": SWEEP_SEEDS_COR, "sweep_mean_J": sweep, "chosen": best, "report_seeds": SEEDS,
            "report": report}


# ------------------------------------------------------------------ E4

CP_EP_SWEEP, CP_EP, CP_CAP = 600, 1000, 500


def cp_run(method, seed, a, aw, n):
    if method == "reinforce":
        return m.cartpole_reinforce(seed, n, a, cap=CP_CAP)
    if method == "reinforce_baseline":
        return m.cartpole_reinforce(seed, n, a, baseline="value", alpha_w=aw, cap=CP_CAP)
    return m.cartpole_actor_critic(seed, n, a, alpha_w=aw, cap=CP_CAP)


def moving(x, k=20):
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def e4_cartpole():
    a_grid = [2.0 ** k for k in range(-14, -3, 2)]
    aw_grid = {"reinforce": [None], "reinforce_baseline": [2.0 ** k for k in range(-12, 1, 2)],
               "actor_critic": [2.0 ** k for k in range(-12, 1, 2)]}
    sweep, chosen = {}, {}
    t0 = time.time()
    for meth in ("reinforce", "reinforce_baseline", "actor_critic"):
        sweep[meth] = {}
        best_val, best_key = -1, None
        for aw in aw_grid[meth]:
            for a in a_grid:
                vals = [cp_run(meth, sd, a, aw, CP_EP_SWEEP)[0].mean() for sd in SWEEP_SEEDS_CP]
                key = f"a=2^{int(np.log2(a))}" + (f",aw=2^{int(np.log2(aw))}" if aw else "")
                sweep[meth][key] = r6(np.mean(vals))
                if np.mean(vals) > best_val:
                    best_val, best_key = np.mean(vals), (a, aw, key)
        chosen[meth] = best_key
    sweep_sec = time.time() - t0

    report = {}
    cps = [1, 50, 100, 200, 400, 600, 800, 1000]
    for meth in ("reinforce", "reinforce_baseline", "actor_critic"):
        a, aw, key = chosen[meth]
        lens, last100, reach, mov_all, steps, secs = [], [], [], [], 0, 0.0
        for sd in SEEDS:
            t = time.time()
            L, _ = cp_run(meth, sd, a, aw, CP_EP)
            secs += time.time() - t
            steps += int(L.sum())
            lens.append(L)
            last100.append(L[-100:].mean())
            mv = moving(L, 20)
            mov_all.append(mv)
            hit = np.nonzero(mv >= 475)[0]
            reach.append(int(hit[0]) + 1 if len(hit) else None)
        last100 = np.array(last100)
        M = np.array(mov_all)
        report[meth] = {
            "chosen": key, "mean_return_all_episodes": r6(np.mean([L.mean() for L in lens])),
            "last100_per_seed": [r6(x) for x in last100],
            "last100_mean": r6(last100.mean()), "last100_sd_across_seeds": r6(last100.std(ddof=1)),
            "last100_min": r6(last100.min()), "last100_max": r6(last100.max()),
            "seeds_last100_ge_475": int((last100 >= 475).sum()),
            "seeds_last100_lt_50": int((last100 < 50).sum()),
            "episode_reach_mov20_ge_475": reach,
            "fan_mov20": {str(c): {"min": r6(M[:, c - 1].min()), "median": r6(np.median(M[:, c - 1])),
                                   "max": r6(M[:, c - 1].max())} for c in cps},
            "env_steps_total": steps, "us_per_env_step": r6(1e6 * secs / steps),
        }
    push = {}
    for a in (0, 1):
        L = []
        for sd in range(20):
            env = cp.CartPole(sd, CP_CAP)
            env.reset()
            n = 0
            while True:
                _, _, fell, trunc = env.step(a)
                n += 1
                if fell or trunc:
                    break
            L.append(n)
        push["always_" + ("right" if a else "left")] = {"min": min(L), "max": max(L), "mean": r6(np.mean(L))}
    return {"cap": CP_CAP, "gamma": 0.99, "constant_push_20_starts": push, "episodes": CP_EP, "episodes_sweep": CP_EP_SWEEP,
            "sweep_seeds": SWEEP_SEEDS_CP, "report_seeds": SEEDS, "sweep_mean_return": sweep,
            "sweep_seconds": r6(sweep_sec), "report": report,
            "policy_features": "x/2.4, x_dot/3, angle/0.21, angle_dot/3, 1",
            "value_features": "정책 특징 5개 + 네 상태값 제곱 4개",
            "notes": "REINFORCE는 gamma^t 곱을 빼고 G_t를 그대로 씀. 마찰 0, g=+9.8, 실패 |x|>2.4 또는 |각도|>12도"}, chosen


# ------------------------------------------------------------------ E5

def e5_impl_check():
    # 복도: 판 50개를 이어 붙여 손 기울기와 autograd 비교(특징은 상수 1 하나)
    rng = np.random.default_rng(7)
    th = 0.3
    eps = [m.corridor_episode(th, rng) for _ in range(50)]
    max_rel = 0.0
    for b in ("none", "const", "value"):
        for e in eps:
            g_np = m.corridor_grad_estimate(e, th, b)
            G = m.returns(e.rewards, 1.0)
            if b == "none":
                bb = np.zeros_like(G)
            elif b == "const":
                bb = np.full_like(G, cor.exact_J(th))
            else:
                bb = cor.exact_v(cor.sigmoid(th))[np.asarray(e.states)]
            g_t = mt.pg_grad([th], np.ones((len(G), 1)), e.actions, G - bb)[0]
            max_rel = max(max_rel, abs(g_np - g_t) / max(1e-12, abs(g_np)))
    # cart-pole: 무작위 theta로 판 하나
    env = cp.CartPole(3, 500)
    theta = np.array([0.3, -0.2, 1.5, 0.4, 0.1])
    S, A, R, _ = m.run_cartpole_episode(env, theta, np.random.default_rng(3))
    G = m.returns(R, 0.99)
    Phi = np.array([m.phi_policy(s) for s in S])
    g_np = ((G * (A - m.sigmoid(Phi @ theta))) @ Phi)
    g_t = mt.pg_grad(theta, Phi, A, G)
    rel_cp = float(np.max(np.abs(g_np - g_t) / np.maximum(1e-12, np.abs(g_np))))
    G_t = mt.returns(R, 0.99).numpy()
    # 정확한 기울기 vs 유한 차분
    h = 1e-6
    fd = {str(t): r6(abs(cor.exact_grad(t) - (cor.exact_J(t + h) - cor.exact_J(t - h)) / (2 * h))) for t in (-1.0, 0.0, 0.5)}
    return {"corridor_max_rel_err_np_vs_torch": float(max_rel), "cartpole_max_rel_err_np_vs_torch": rel_cp,
            "cartpole_episode_len": int(len(A)), "returns_max_abs_err": float(np.max(np.abs(G - G_t))),
            "rtol": 1e-9, "exact_grad_vs_finite_diff_abs_err": fd}


# ------------------------------------------------------------------ main

def main():
    t_all = time.time()
    res = {"meta": {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "machine": f"{platform.system()} {platform.machine()}", "torch_threads": torch.get_num_threads(),
    }}
    res["E0_hand"] = e0_hand()
    res["E1_value_based"] = e1_value_based()
    res["E2_variance"] = e2_variance()
    res["E3_corridor_learning"] = e3_corridor_learning()
    res["E4_cartpole"], chosen = e4_cartpole()
    res["E5_impl"] = e5_impl_check()
    res["E6_cost"] = {
        "reinforce_memory": "판 하나의 (상태, 행동, 보상)을 끝날 때까지 모두 저장. 길이 T에 비례",
        "actor_critic_memory": "직전 걸음 하나만. 판 길이와 무관",
        "reinforce_updates_per_episode": 1, "actor_critic_updates_per_episode": "걸음 수만큼",
        "us_per_env_step": {k: v["us_per_env_step"] for k, v in res["E4_cartpole"]["report"].items()},
    }

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        rng = np.random.default_rng(0)
        for i in range(3):
            e = m.corridor_episode(0.0, rng)
            log_rows(f, 0, i, "corridor", e.states, e.actions, e.rewards, e.done, {"theta": 0.0, "p_right": 0.5},
                     s_names=cor.NAMES)
        for tag, theta in (("untrained", np.zeros(5)),):
            env = cp.CartPole(0, CP_CAP)
            S, A, R, fell = m.run_cartpole_episode(env, theta, np.random.default_rng(0))
            log_rows(f, 0, 0, "cartpole", list(S), A, R, fell, {"policy": tag})
        a, aw, _ = chosen["actor_critic"]
        _, theta = m.cartpole_actor_critic(0, CP_EP, a, alpha_w=aw, cap=CP_CAP)
        env = cp.CartPole(0, CP_CAP)
        S, A, R, fell = m.run_cartpole_episode(env, theta, np.random.default_rng(1))
        res["E4_cartpole"]["trained_ac_seed0_eval_len"] = int(len(A))
        res["E4_cartpole"]["trained_ac_seed0_theta"] = [r6(x) for x in theta]
        log_rows(f, 0, 1, "cartpole", list(S), A, R, fell, {"policy": "actor_critic_seed0_after_1000"})

    res["meta"]["seconds"] = r6(time.time() - t_all)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E0_hand", "E1_value_based")}, ensure_ascii=False, indent=1)[:3000])
    print("E2", json.dumps(res["E2_variance"], ensure_ascii=False)[:2500])
    print("E3", json.dumps(res["E3_corridor_learning"]["report"], ensure_ascii=False)[:2000])
    for k, v in res["E4_cartpole"]["report"].items():
        print("E4", k, v["chosen"], v["last100_mean"], v["last100_sd_across_seeds"], v["seeds_last100_ge_475"],
              v["episode_reach_mov20_ge_475"], v["us_per_env_step"])
    print("E5", res["E5_impl"])
    print("seconds", res["meta"]["seconds"])


if __name__ == "__main__":
    main()
