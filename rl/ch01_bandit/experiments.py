"""강화학습 1장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch01_bandit/experiments.py`

E0 손계산 확인: 손잡이 두 개(0.45, 0.55)에서 "좋은 쪽 첫 결과 0, 나쁜 쪽 첫 결과 1"의 확률,
   UCB1 보너스 sqrt(2 ln 10 / 1), sqrt(2 ln 10 / 9)와 지수 비교
E1 실패 재현(2팔 축소판): 욕심쟁이 규칙, 2,000 반복 x 1,000번, 갇힘 비율과 로그 한 줄
E2 10팔 비교: p = (0.10, 0.15, ..., 0.55), 네 규칙(욕심쟁이, eps 0.1, UCB1, Thompson),
   2,000 반복(시드 0~1999) x 1,000번, 누적 후회 평균·분위수, 좋은 손잡이 선택 비율 곡선, 갇힘 비율,
   Lai-Robbins 곡선, 실측 시간
E3 eps 민감도: eps in {0.01, 0.1, 0.3}
E4 간격 민감도: 2팔 (0.5, 0.5 + Delta), Delta in {0.02, 0.05, 0.1, 0.2}
E5 긴 구간: 10팔, 10,000번, 200 반복(시드 0~199), 후회와 Lai-Robbins 곡선·Auer 정리 1 상한
E6 구현 대조: UCB1·Thompson 선택 열 NumPy vs PyTorch, 20 반복 x 1,000번

결과는 results/ch01.json, 상호작용 로그 일부는 logs/ch01_two_arm.jsonl 에 저장한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch01_bandit_np as bd  # noqa: E402
import rl_ch01_bandit_torch as bdt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch01.json"
LOG = HERE / "logs" / "ch01_two_arm.jsonl"

P10 = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
P2 = [0.45, 0.55]
N_STEPS = 1000
SEEDS = list(range(2000))
POLICY_SEED = 12345
CHECKPOINTS = [10, 20, 50, 100, 200, 500, 1000]
STUCK_SHARE = 0.10


def policies():
    return [bd.greedy(), bd.eps_greedy(0.1), bd.ucb1(), bd.thompson_beta()]


def summarize(actions, env, checkpoints=CHECKPOINTS):
    n = actions.shape[1]
    reg = bd.pseudo_regret(actions, env.gaps)
    best = actions == env.best
    share_best = best.mean(axis=1)  # 처음부터 끝까지 좋은 손잡이를 고른 비율
    cp = [c for c in checkpoints if c <= n]
    final = reg[:, -1]
    counts = bd.counts_over_time(actions, env.K)[:, -1, :]
    most = np.argmax(counts, axis=1)
    return {
        "regret_final_mean": float(final.mean()),
        "regret_final_se": float(final.std(ddof=1) / math.sqrt(len(final))),
        "regret_final_p5": float(np.percentile(final, 5)),
        "regret_final_p50": float(np.percentile(final, 50)),
        "regret_final_p95": float(np.percentile(final, 95)),
        "regret_mean_at": {str(c): float(reg[:, c - 1].mean()) for c in cp},
        "regret_p5_at": {str(c): float(np.percentile(reg[:, c - 1], 5)) for c in cp},
        "regret_p95_at": {str(c): float(np.percentile(reg[:, c - 1], 95)) for c in cp},
        "best_arm_rate_at": {str(c): float(best[:, c - 1].mean()) for c in cp},
        "best_share_mean": float(share_best.mean()),
        "frac_runs_best_share_below_10pct": float((share_best < STUCK_SHARE).mean()),
        "frac_runs_best_share_below_1pct": float((share_best < 0.01).mean()),
        "most_pulled_arm_hist": [int((most == j).sum()) for j in range(env.K)],
        "mean_pulls_per_arm": [float(x) for x in counts.mean(axis=0)],
    }


def run_timed(policy, env, n_steps, seed=POLICY_SEED):
    t0 = time.perf_counter()
    acts, rews = bd.run(policy, env, n_steps, np.random.default_rng(seed))
    dt = time.perf_counter() - t0
    return acts, rews, dt


def e0_hand():
    p_bad, p_good = P2
    lock_min = (1 - p_good) * p_bad
    b1 = math.sqrt(2 * math.log(10) / 1)
    b9 = math.sqrt(2 * math.log(10) / 9)
    return {
        "p_two_arm": P2,
        "p_good_first_0": 1 - p_good,
        "p_bad_first_1": p_bad,
        "lock_lower_bound": lock_min,
        "ucb_example": {
            "n": 10,
            "abandoned": {"pulls": 1, "successes": 0, "mean": 0.0, "bonus": b1, "index": 0.0 + b1},
            "favored": {"pulls": 9, "successes": 4, "mean": 4 / 9, "bonus": b9, "index": 4 / 9 + b9},
            "two_ln_10": 2 * math.log(10),
        },
    }


def e1_two_arm():
    env = bd.BernoulliBandit(P2, SEEDS, N_STEPS)
    acts, rews, _ = run_timed(bd.greedy(), env, N_STEPS)
    s = summarize(acts, env)
    # 첫 두 번(손잡이 0, 1을 한 번씩)의 결과로 나눠 보기
    first_bad, first_good = rews[:, 0], rews[:, 1]
    pattern = {}
    share_best = (acts == env.best).mean(axis=1)
    for fb in (0, 1):
        for fg in (0, 1):
            m = (first_bad == fb) & (first_good == fg)
            pattern[f"bad{fb}_good{fg}"] = {
                "runs": int(m.sum()),
                "frac_runs": float(m.mean()),
                "stuck_below_10pct": int((share_best[m] < STUCK_SHARE).sum()),
                "mean_best_share": float(share_best[m].mean()) if m.any() else None,
            }
    # 로그: "좋은 쪽 첫 결과 0, 나쁜 쪽 첫 결과 1"인 첫 시드 하나, 처음 12번
    m = np.where((first_bad == 1) & (first_good == 0))[0]
    sidx = int(m[0])
    log = []
    for t in range(12):
        log.append({"seed": SEEDS[sidx], "episode": 0, "t": t, "s": 0, "a": int(acts[sidx, t]),
                    "r": float(rews[sidx, t]), "s_next": 0, "done": False,
                    "info": {"policy": "greedy", "p": P2}})
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pulls = bd.counts_over_time(acts[sidx:sidx + 1], 2)[0, -1]
    # 같은 2팔에서 UCB1·Thompson·eps 0.1도
    others = {}
    for pol in (bd.eps_greedy(0.1), bd.ucb1(), bd.thompson_beta()):
        a2, _, _ = run_timed(pol, env, N_STEPS)
        sh = (a2 == env.best).mean(axis=1)
        others[pol.name] = {"frac_runs_best_share_below_10pct": float((sh < STUCK_SHARE).mean()),
                            "best_share_mean": float(sh.mean()),
                            "regret_final_mean": float(bd.pseudo_regret(a2, env.gaps)[:, -1].mean())}
    return {"p": P2, "n_steps": N_STEPS, "runs": len(SEEDS), "greedy": s, "first_pull_patterns": pattern,
            "log_seed": SEEDS[sidx], "log_seed_final_pulls": [int(x) for x in pulls],
            "log_seed_first_actions": [int(x) for x in acts[sidx, :12]],
            "log_seed_first_rewards": [float(x) for x in rews[sidx, :12]],
            "others": others, "log_records": len(log)}


def e2_ten_arm():
    env = bd.BernoulliBandit(P10, SEEDS, N_STEPS)
    out = {}
    for pol in policies():
        acts, _, dt = run_timed(pol, env, N_STEPS)
        s = summarize(acts, env)
        s["seconds"] = dt
        s["microseconds_per_step_per_run"] = dt / (N_STEPS * env.S) * 1e6
        out[pol.name] = s
    lr, c = bd.lai_robbins_curve(P10, np.array(CHECKPOINTS))
    return {"p": P10, "n_steps": N_STEPS, "runs": len(SEEDS), "policy_seed": POLICY_SEED, "by_policy": out,
            "lai_robbins_const": c, "lai_robbins_at": {str(k): float(v) for k, v in zip(CHECKPOINTS, lr)},
            "auer_bound_at": {str(k): float(bd.auer_ucb1_bound(P10, k)) for k in CHECKPOINTS},
            "gaps": [float(g) for g in env.gaps],
            "kl_to_best": [float(x) for x in bd.bernoulli_kl(np.array(P10[:-1]), P10[-1])],
            "ops_per_step": {"greedy": "K번 나눗셈 + argmax", "ucb1": "K번 나눗셈·로그·제곱근 + argmax",
                             "thompson": "K번 베타 표본 + argmax", "K": len(P10)}}


def e3_eps():
    env = bd.BernoulliBandit(P10, SEEDS, N_STEPS)
    out = {}
    for eps in (0.01, 0.1, 0.3):
        acts, _, _ = run_timed(bd.eps_greedy(eps), env, N_STEPS)
        s = summarize(acts, env)
        out[str(eps)] = {k: s[k] for k in ("regret_final_mean", "regret_final_p5", "regret_final_p95",
                                           "best_share_mean", "frac_runs_best_share_below_10pct",
                                           "best_arm_rate_at")}
        # 마지막 100번 동안 eps 탐험 때문에 나쁜 손잡이를 고르는 몫의 이론값: eps * (K-1)/K
        out[str(eps)]["forced_bad_rate"] = eps * (env.K - 1) / env.K
    return out


def e4_gap():
    out = {}
    for d in (0.02, 0.05, 0.1, 0.2):
        p = [0.5, round(0.5 + d, 2)]
        env = bd.BernoulliBandit(p, SEEDS, N_STEPS)
        row = {}
        for pol in policies():
            acts, _, _ = run_timed(pol, env, N_STEPS)
            s = summarize(acts, env)
            row[pol.name] = {"regret_final_mean": s["regret_final_mean"], "regret_final_p95": s["regret_final_p95"],
                             "best_share_mean": s["best_share_mean"],
                             "frac_runs_best_share_below_10pct": s["frac_runs_best_share_below_10pct"]}
        row["lai_robbins_at_1000"] = float(bd.lai_robbins_curve(p, N_STEPS)[0])
        row["worst_case_linear_regret"] = d * N_STEPS
        out[str(d)] = row
    return out


def e5_long():
    n = 10_000
    seeds = list(range(200))
    env = bd.BernoulliBandit(P10, seeds, n)
    cps = [100, 1000, 3000, 10_000]
    out = {}
    for pol in policies():
        acts, _, dt = run_timed(pol, env, n)
        reg = bd.pseudo_regret(acts, env.gaps)
        out[pol.name] = {"regret_mean_at": {str(c): float(reg[:, c - 1].mean()) for c in cps},
                         "regret_p95_at": {str(c): float(np.percentile(reg[:, c - 1], 95)) for c in cps},
                         "growth_1000_to_10000": float(reg[:, -1].mean() - reg[:, 999].mean()),
                         "seconds": dt}
    lr, c = bd.lai_robbins_curve(P10, np.array(cps))
    return {"n_steps": n, "runs": len(seeds), "by_policy": out, "lai_robbins_const": c,
            "lai_robbins_at": {str(k): float(v) for k, v in zip(cps, lr)},
            "lai_robbins_growth_1000_to_10000": float(c * (math.log(10_000) - math.log(1000))),
            "auer_bound_at": {str(k): float(bd.auer_ucb1_bound(P10, k)) for k in cps}}


def e6_impl():
    seeds = list(range(20))
    env = bd.BernoulliBandit(P10, seeds, N_STEPS)
    res = {}
    for name, pol in (("ucb1", bd.ucb1()), ("thompson", bd.thompson_beta())):
        a_np, _ = bd.run(pol, env, N_STEPS, np.random.default_rng(7))
        a_t = bdt.run(name, env, N_STEPS, np.random.default_rng(7)).numpy()
        res[name] = {"steps_compared": int(a_np.size), "mismatches": int((a_np != a_t).sum())}
    rng = np.random.default_rng(3)
    N = rng.integers(0, 50, size=(5, 10)).astype(float)
    X = np.floor(rng.random((5, 10)) * (N + 1))
    with np.errstate(invalid="ignore"):
        i_np = bd.ucb1_index(N, X, 200)
    i_t = bdt.ucb1_index(torch.from_numpy(N), torch.from_numpy(X), 200).numpy()
    fin = np.isfinite(i_np)
    res["ucb1_index_max_abs_diff"] = float(np.max(np.abs(i_np[fin] - i_t[fin])))
    res["ucb1_inf_positions_match"] = bool(np.array_equal(~fin, ~np.isfinite(i_t)))
    return res


def main():
    t0 = time.perf_counter()
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "setup": {"reward_table": "반복 s마다 default_rng(s)로 표를 만들고, 손잡이 a의 k번째 결과 = table[s,k,a]",
                  "policy_seed": POLICY_SEED, "init": "욕심쟁이·eps·UCB1은 처음 K번 손잡이 0..K-1을 한 번씩, "
                  "Thompson은 Beta(1,1)에서 바로 시작", "tie_break": "최댓값이 여럿이면 그중 균등하게",
                  "regret": "sum_j Delta_j T_j(n)", "stuck_rule": "1,000번 중 좋은 손잡이를 고른 비율 < 10%"},
        "E0": e0_hand(),
        "E1": e1_two_arm(),
        "E2": e2_ten_arm(),
        "E3": e3_eps(),
        "E4": e4_gap(),
        "E5": e5_long(),
        "E6": e6_impl(),
    }
    res["total_seconds"] = time.perf_counter() - t0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res["E0"], ensure_ascii=False))
    e1 = res["E1"]
    print("E1 greedy", {k: e1["greedy"][k] for k in ("frac_runs_best_share_below_10pct", "frac_runs_best_share_below_1pct",
                                                     "best_share_mean", "regret_final_mean")})
    print("E1 patterns", e1["first_pull_patterns"])
    print("E1 log seed", e1["log_seed"], e1["log_seed_first_actions"], e1["log_seed_first_rewards"], e1["log_seed_final_pulls"])
    print("E1 others", e1["others"])
    for k, v in res["E2"]["by_policy"].items():
        print("E2", k, round(v["regret_final_mean"], 2), round(v["regret_final_p5"], 2), round(v["regret_final_p95"], 2),
              round(v["best_share_mean"], 3), v["frac_runs_best_share_below_10pct"], v["best_arm_rate_at"],
              round(v["microseconds_per_step_per_run"], 3), v["most_pulled_arm_hist"])
    print("E2 LR", res["E2"]["lai_robbins_const"], res["E2"]["lai_robbins_at"], res["E2"]["auer_bound_at"]["1000"])
    print("E3", json.dumps(res["E3"], ensure_ascii=False))
    print("E4", json.dumps(res["E4"], ensure_ascii=False))
    print("E5", json.dumps(res["E5"], ensure_ascii=False))
    print("E6", res["E6"])
    print("total", res["total_seconds"])


if __name__ == "__main__":
    main()
