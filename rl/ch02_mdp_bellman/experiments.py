"""강화학습 2장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch02_mdp_bellman/experiments.py`

E0 손계산 확인: slip 0, gamma 0.9에서 최적 경로 위 칸들의 V* (10, 9, 8.1, ...)
E1 실패 재현: 밴딧식 근시 정책 vs 최적 정책, 10시드 x 1,000 에피소드, 방문 히트맵
   E1b 먼 구석 (4,0)에서 출발했을 때 두 정책
E2 할인율과 경계: gamma in {0.3, 0.5, 0.9, 0.99}에서 각 칸이 +1과 +10 중 어디로 가는지, 시작 칸의 전환 gamma
E3 미끄러짐: slip in {0, 0.1, 0.3}에서 두 정책의 평균·분산, 벨만 방정식이 예측한 값과 실제 평균 비교
E4 구현 대조: NumPy vs PyTorch 백업 최대 차이, 반복 횟수, 백업 한 번의 곱셈 수

결과는 results/ch02.json, 상호작용 로그 일부는 logs/ch02_rollouts.jsonl 에 저장한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch02_gridworld as gw  # noqa: E402
import rl_ch02_mdp_np as mdp  # noqa: E402
import rl_ch02_mdp_torch as mdpt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch02.json"
LOG = HERE / "logs" / "ch02_rollouts.jsonl"
SEEDS = list(range(10))
EPISODES = 1000
S0 = gw.to_index(gw.START, gw.W)
TERM = {gw.to_index(t, gw.W) for t in gw.TERMINALS}
SMALL_I, BIG_I = gw.to_index(gw.SMALL, gw.W), gw.to_index(gw.BIG, gw.W)


def grid_list(v):
    return [[float(x) for x in row] for row in np.asarray(v).reshape(gw.H, gw.W)]


def e0_hand(gamma=0.9):
    P, R = gw.chapter_grid(0.0)
    V, iters = mdp.optimal_values(P, R, gamma)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]  # 시작에서 +10까지 6걸음 경로의 칸들
    return {
        "gamma": gamma,
        "path": [list(p) for p in path],
        "V_star_on_path": [float(V[gw.to_index(p, gw.W)]) for p in path],
        "ten_gamma_pow5": 10 * gamma**5,
        "small_option_value": 1.0,
        "V_star_grid": grid_list(V),
        "iterations": iters,
    }


def run_policy(pi, P, R, gamma, log=None, log_episodes=0, s0=S0):
    per_seed_mean, all_returns, steps, end_small = [], [], [], 0
    visits = np.zeros(gw.H * gw.W)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rs = []
        for ep in range(EPISODES):
            lg = log if (log is not None and seed == 0 and ep < log_episodes) else None
            G, T, vis = mdp.rollout(pi, P, R, gamma, s0, TERM, rng, log=lg, seed=seed, episode=ep)
            rs.append(G)
            steps.append(T)
            end_small += vis[-1] == SMALL_I
            for s in set(vis):
                visits[s] += 1
        per_seed_mean.append(float(np.mean(rs)))
        all_returns.extend(rs)
    n = len(SEEDS) * EPISODES
    all_returns = np.array(all_returns)
    return {
        "mean_return": float(all_returns.mean()),
        "std_return_episodes": float(all_returns.std()),
        "var_return_episodes": float(all_returns.var()),
        "se_mean": float(all_returns.std() / np.sqrt(n)),
        "per_seed_mean_min": float(min(per_seed_mean)),
        "per_seed_mean_max": float(max(per_seed_mean)),
        "mean_steps": float(np.mean(steps)),
        "frac_end_small": float(end_small / n),
        "visit_frac_grid": grid_list(visits / n),
    }


def e1_failure(gamma=0.9):
    P, R = gw.chapter_grid(0.0)
    pis = {"myopic": mdp.myopic_policy(P, R), "optimal": mdp.optimal_policy(P, R, gamma)}
    out = {"gamma": gamma, "slip": 0.0, "seeds": len(SEEDS), "episodes_per_seed": EPISODES}
    for name, pi in pis.items():
        res = run_policy(pi, P, R, gamma)
        Vpi, _ = mdp.evaluate(pi, P, R, gamma)
        res["bellman_V_start"] = float(Vpi[S0])
        res["start_action_probs"] = [float(x) for x in pi[S0]]
        out[name] = res
    # E1b: 보상이 한 걸음 안에 없는 칸 (4,0)에서 출발하면 근시 정책은 모든 행동이 동점(0)이라 무작위로 걷는다
    far = gw.to_index((4, 0), gw.W)
    out["from_far_corner"] = {"start": [4, 0]}
    for name, pi in pis.items():
        res = run_policy(pi, P, R, gamma, s0=far)
        Vpi, _ = mdp.evaluate(pi, P, R, gamma)
        res["bellman_V_start"] = float(Vpi[far])
        res["start_action_probs"] = [float(x) for x in pi[far]]
        res["mc_minus_bellman"] = res["mean_return"] - float(Vpi[far])
        out["from_far_corner"][name] = res
    # E1c: 근시 정책의 실측-방정식 차이가 표준오차 3배쯤 나와서, 다른 시드(123)로 60,000에피소드를 따로 확인한다
    rng = np.random.default_rng(123)
    g = np.array([mdp.rollout(pis["myopic"], P, R, gamma, far, TERM, rng, max_steps=5000)[0] for _ in range(60_000)])
    out["from_far_corner"]["myopic_recheck_seed123_60k"] = {
        "mean_return": float(g.mean()), "se_mean": float(g.std() / np.sqrt(len(g)))}
    return out


def heads_to(P, R, gamma):
    """slip 0에서 최적 정책(동점이면 번호가 작은 행동)을 따라가면 어느 종료 칸에 닿는지 칸마다 적는다."""
    V, _ = mdp.optimal_values(P, R, gamma)
    Q = mdp.q_from_v(V, P, R, gamma)
    grid = []
    for s in range(gw.H * gw.W):
        if s in TERM:
            grid.append("T+1" if s == SMALL_I else "T+10")
            continue
        cur, n = s, 0
        while cur not in TERM and n < 100:
            cur = int(np.argmax(P[cur, int(np.argmax(Q[cur]))]))
            n += 1
        grid.append("+1" if cur == SMALL_I else "+10")
    return grid, V, Q


def e2_gamma():
    P, R = gw.chapter_grid(0.0)
    rows = {}
    for gamma in (0.3, 0.5, 0.9, 0.99):
        grid, V, Q = heads_to(P, R, gamma)
        rows[str(gamma)] = {
            "start_best_action": gw.ACTION_NAMES[int(np.argmax(Q[S0]))],
            "V_star_start": float(V[S0]),
            "Q_start": {gw.ACTION_NAMES[a]: float(Q[S0, a]) for a in range(4)},
            "states_to_small": sum(g == "+1" for g in grid),
            "states_to_big": sum(g == "+10" for g in grid),
            "heads_to_grid": [grid[i * gw.W:(i + 1) * gw.W] for i in range(gw.H)],
        }
    # 시작 칸에서 오른쪽(+1)이 최선인 마지막 gamma와, 아래(+10 방향)가 최선인 첫 gamma를 촘촘히 찾는다
    switch = None
    prev = None
    for gamma in np.round(np.arange(0.50, 0.75, 0.001), 3):
        V, _ = mdp.optimal_values(P, R, float(gamma))
        Q = mdp.q_from_v(V, P, R, float(gamma))
        best = "small" if Q[S0, 1] > Q[S0, 2] else "big"
        if prev == "small" and best == "big":
            switch = float(gamma)
            break
        prev = best
    return {"by_gamma": rows, "start_switch_gamma_grid_0p001": switch,
            "start_switch_gamma_exact": float(0.1 ** (1 / 5))}


def e3_slip(gamma=0.9):
    out = {}
    log = []
    for slip in (0.0, 0.1, 0.3):
        P, R = gw.chapter_grid(slip)
        pis = {"myopic": mdp.myopic_policy(P, R), "optimal": mdp.optimal_policy(P, R, gamma)}
        row = {}
        for name, pi in pis.items():
            res = run_policy(pi, P, R, gamma, log=log if slip == 0.1 else None, log_episodes=3)
            Vpi, iters = mdp.evaluate(pi, P, R, gamma)
            res["bellman_V_start"] = float(Vpi[S0])
            res["mc_minus_bellman"] = res["mean_return"] - float(Vpi[S0])
            res["evaluation_iterations"] = iters
            res.pop("visit_frac_grid")
            row[name] = res
        V, _ = mdp.optimal_values(P, R, gamma)
        Q = mdp.q_from_v(V, P, R, gamma)
        row["optimal_start_action"] = gw.ACTION_NAMES[int(np.argmax(Q[S0]))]
        row["optimal_arrows"] = [
            ["T" if (i * gw.W + j) in TERM else gw.ARROWS[int(np.argmax(Q[i * gw.W + j]))] for j in range(gw.W)]
            for i in range(gw.H)]
        out[str(slip)] = row
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"gamma": gamma, "by_slip": out, "log_records": len(log)}


def e4_impl(gamma=0.9):
    rng = np.random.default_rng(0)
    diffs = {}
    for slip in (0.0, 0.3):
        P, R = gw.chapter_grid(slip)
        V = rng.standard_normal(P.shape[0])
        pi = rng.random((P.shape[0], 4))
        pi /= pi.sum(1, keepdims=True)
        Pt, Rt, Vt, pit = (torch.tensor(x, dtype=torch.float64) for x in (P, R, V, pi))
        d1 = np.max(np.abs(mdp.bellman_expectation_backup(V, pi, P, R, gamma)
                           - mdpt.bellman_expectation_backup(Vt, pit, Pt, Rt, gamma).numpy()))
        d2 = np.max(np.abs(mdp.bellman_optimality_backup(V, P, R, gamma)
                           - mdpt.bellman_optimality_backup(Vt, Pt, Rt, gamma).numpy()))
        diffs[str(slip)] = {"expectation_max_abs_diff": float(d1), "optimality_max_abs_diff": float(d2)}
    S, A = gw.H * gw.W, 4
    iters = {}
    for gamma_ in (0.5, 0.9, 0.99):
        P, R = gw.chapter_grid(0.1)
        _, k = mdp.optimal_values(P, R, gamma_)
        iters[str(gamma_)] = k
    return {"numpy_vs_torch": diffs, "multiplies_per_backup": S * A * S,
            "optimality_iterations_slip0p1_by_gamma": iters}


def main():
    res = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "torch_threads": torch.get_num_threads()},
        "grid": {"h": gw.H, "w": gw.W, "start": list(gw.START), "small": list(gw.SMALL), "big": list(gw.BIG),
                 "rewards": {"small": 1.0, "big": 10.0, "step": 0.0},
                 "slip_rule": "확률 slip로 네 방향 중 하나를 균등하게(의도한 방향 포함)"},
        "E0": e0_hand(),
        "E1": e1_failure(),
        "E2": e2_gamma(),
        "E3": e3_slip(),
        "E4": e4_impl(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k] for k in ("E0", "E2")}, ensure_ascii=False)[:3000])
    for k in ("E1",):
        for p in ("myopic", "optimal"):
            r = res[k][p]
            print(k, p, r["mean_return"], r["bellman_V_start"], r["frac_end_small"], r["mean_steps"])
            f = res[k]["from_far_corner"][p]
            print(k, "far", p, f["mean_return"], f["std_return_episodes"], f["bellman_V_start"], f["frac_end_small"], f["mean_steps"], f["start_action_probs"])
    for slip, row in res["E3"]["by_slip"].items():
        for p in ("myopic", "optimal"):
            r = row[p]
            print("E3", slip, p, round(r["mean_return"], 4), round(r["std_return_episodes"], 4),
                  round(r["bellman_V_start"], 4), round(r["frac_end_small"], 4), round(r["mean_steps"], 2))
        print(row["optimal_arrows"])
    print(res["E4"])


if __name__ == "__main__":
    main()
