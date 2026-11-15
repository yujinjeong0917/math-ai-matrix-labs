"""강화학습 2장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch02_mdp_bellman/experiments.py`

E0 손계산 확인: slip 0, gamma 0.9에서 최적 경로 위 칸들의 V* (10, 9, 8.1, ...)
E1 실패 재현: 밴딧식 근시 정책 vs 최적 정책, 10시드 x 1,000 에피소드, 방문 히트맵
   E1b 먼 구석 (4,0)에서 출발했을 때 두 정책
E2 할인율과 경계: gamma in {0.3, 0.5, 0.9, 0.99}에서 각 칸이 +1과 +10 중 어디로 가는지, 시작 칸의 전환 gamma
E3 미끄러짐: slip in {0, 0.1, 0.3}에서 두 정책의 평균·분산, 벨만 방정식이 예측한 값과 실제 평균 비교
E4 구현 대조: NumPy vs PyTorch 백업 최대 차이, 반복 횟수, 백업 한 번의 곱셈 수

결과는 results/ch02.json, 상호작용 로그 일부는 logs/ch02_rollouts.jsonl 에 저장한다.

입력: 없음(격자, 시드, gamma, slip은 코드 상수). 출력: 위 두 파일과 표준출력 요약.
핵심 식: 최적 V* = 벨만 최적방정식의 고정점, 정책 가치 V^pi = 벨만 기대방정식의 고정점(백업은 rl_ch02_mdp_np.py).
gamma = 할인율(기본 0.9). 탐험 파라미터는 없다(정책이 고정이고 행동은 정책 확률에서 뽑는다). 학습률도 없다.
비교 기준: 방정식이 예측한 V^pi(시작 칸)과 1,000 에피소드 평균 리턴(몬테카를로)이 표준오차 안에서 맞는지.
numpy와 torch가 필요하다.
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
# 시드 0~9마다 rng를 새로 만들어 에피소드 1,000개씩. 총 10,000 에피소드
SEEDS = list(range(10))
EPISODES = 1000
# 상태 번호 s = 행 * 폭 + 열
S0 = gw.to_index(gw.START, gw.W)
TERM = {gw.to_index(t, gw.W) for t in gw.TERMINALS}
SMALL_I, BIG_I = gw.to_index(gw.SMALL, gw.W), gw.to_index(gw.BIG, gw.W)


def grid_list(v):
    return [[float(x) for x in row] for row in np.asarray(v).reshape(gw.H, gw.W)]


def e0_hand(gamma=0.9):
    P, R = gw.chapter_grid(0.0)
    V, iters = mdp.optimal_values(P, R, gamma)
    # +10 칸 (2,4)까지의 최단 경로(6걸음). 이 칸들의 V*는 10, 9, 8.1 ...의 역순 (손계산 10*0.9^k)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]  # 시작에서 +10까지 6걸음 경로의 칸들
    return {
        "gamma": gamma,
        "path": [list(p) for p in path],
        "V_star_on_path": [float(V[gw.to_index(p, gw.W)]) for p in path],
        # 식: 6걸음째에 받는 +10의 할인값 10*gamma^5 (보상은 t=0부터 gamma^t로 할인, 6번째 걸음의 t=5)
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
        # rs: 시드 하나의 에피소드별 할인 리턴
        rs = []
        for ep in range(EPISODES):
            lg = log if (log is not None and seed == 0 and ep < log_episodes) else None
            G, T, vis = mdp.rollout(pi, P, R, gamma, s0, TERM, rng, log=lg, seed=seed, episode=ep)
            rs.append(G)
            steps.append(T)
            # 마지막으로 들어간 종료 칸이 +1 칸이면 센다
            end_small += vis[-1] == SMALL_I
            # set으로 한 에피소드 안 중복 방문은 한 번만 센다(방문 비율)
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
        # 식: 표준오차 = 표준편차 / sqrt(에피소드 수 n). 에피소드끼리 독립이라는 가정
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
    # myopic = 한 걸음 기대보상만 보는 정책(밴딧식, gamma=0 효과), optimal = 최적방정식 해 V*에서 얻은 탐욕 정책
    out = {"gamma": gamma, "slip": 0.0, "seeds": len(SEEDS), "episodes_per_seed": EPISODES}
    for name, pi in pis.items():
        res = run_policy(pi, P, R, gamma)
        Vpi, _ = mdp.evaluate(pi, P, R, gamma)
        res["bellman_V_start"] = float(Vpi[S0])
        # 위 bellman_V_start는 식 V^pi = 기대방정식의 고정점(해석 해). mean_return(몬테카를로)과 비교한다
        res["start_action_probs"] = [float(x) for x in pi[S0]]
        out[name] = res
    # E1b: 보상이 한 걸음 안에 없는 칸 (4,0)에서 출발하면 근시 정책은 모든 행동이 동점(0)이라 무작위로 걷는다
    # (4,0)은 보상 칸에서 멀다. 근시 정책은 이 칸에서 모든 행동 E[r]=0이라 동점, 균등 무작위로 걷는다
    far = gw.to_index((4, 0), gw.W)
    out["from_far_corner"] = {"start": [4, 0]}
    for name, pi in pis.items():
        res = run_policy(pi, P, R, gamma, s0=far)
        Vpi, _ = mdp.evaluate(pi, P, R, gamma)
        res["bellman_V_start"] = float(Vpi[far])
        res["start_action_probs"] = [float(x) for x in pi[far]]
        # 몬테카를로 평균 - 방정식 값. 0에 가까워야 한다(표준오차 단위로 비교)
        res["mc_minus_bellman"] = res["mean_return"] - float(Vpi[far])
        out["from_far_corner"][name] = res
    # E1c: 근시 정책의 실측-방정식 차이가 표준오차 3배쯤 나와서, 다른 시드(123)로 60,000에피소드를 따로 확인한다
    # max_steps=5000: 근시 정책은 무작위 보행이라 기본 500걸음에서 잘리면 값이 편향될 수 있어 늘린다(편향 방지가 의도로 보이나 확인 필요)
    rng = np.random.default_rng(123)
    g = np.array([mdp.rollout(pis["myopic"], P, R, gamma, far, TERM, rng, max_steps=5000)[0] for _ in range(60_000)])
    out["from_far_corner"]["myopic_recheck_seed123_60k"] = {
        "mean_return": float(g.mean()), "se_mean": float(g.std() / np.sqrt(len(g)))}
    return out


def heads_to(P, R, gamma):
    """slip 0에서 최적 정책(동점이면 번호가 작은 행동)을 따라가면 어느 종료 칸에 닿는지 칸마다 적는다."""
    V, _ = mdp.optimal_values(P, R, gamma)
    Q = mdp.q_from_v(V, P, R, gamma)
    # 각 칸에서 최적 행동(Q의 argmax, 동점이면 번호 작은 쪽)을 따라 가다가 닿는 종료 칸을 기록
    grid = []
    for s in range(gw.H * gw.W):
        if s in TERM:
            grid.append("T+1" if s == SMALL_I else "T+10")
            continue
        cur, n = s, 0
        while cur not in TERM and n < 100:
            # slip 0이라 P[s,a]가 원핫이므로 argmax가 다음 칸. n<100은 무한 루프 방지
            cur = int(np.argmax(P[cur, int(np.argmax(Q[cur]))]))
            n += 1
        grid.append("+1" if cur == SMALL_I else "+10")
    return grid, V, Q


def e2_gamma():
    P, R = gw.chapter_grid(0.0)
    rows = {}
    # gamma를 바꿔 가며 시작 칸의 최선 행동이 +1(오른쪽)에서 +10 쪽(아래)으로 바뀌는 경계를 본다
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
    # round는 arange의 부동소수 꼬리(0.5010000000000001 등)를 제거. Q[S0,1] = 오른쪽(+1), Q[S0,2] = 아래
    for gamma in np.round(np.arange(0.50, 0.75, 0.001), 3):
        V, _ = mdp.optimal_values(P, R, float(gamma))
        Q = mdp.q_from_v(V, P, R, float(gamma))
        best = "small" if Q[S0, 1] > Q[S0, 2] else "big"
        if prev == "small" and best == "big":
            switch = float(gamma)
            break
        prev = best
    return {"by_gamma": rows, "start_switch_gamma_grid_0p001": switch,
            # 식: +1(즉시, 할인 없음)과 10*gamma^5가 같아지는 gamma = 0.1^(1/5) 약 0.631 (slip 0에서 시작 칸의 전환점)
            "start_switch_gamma_exact": float(0.1 ** (1 / 5))}


def e3_slip(gamma=0.9):
    out = {}
    log = []
    for slip in (0.0, 0.1, 0.3):
        P, R = gw.chapter_grid(slip)
        pis = {"myopic": mdp.myopic_policy(P, R), "optimal": mdp.optimal_policy(P, R, gamma)}
        row = {}
        for name, pi in pis.items():
            # 로그는 slip 0.1일 때 시드 0의 앞 3 에피소드만 저장(run_policy가 seed==0 조건을 건다)
            res = run_policy(pi, P, R, gamma, log=log if slip == 0.1 else None, log_episodes=3)
            Vpi, iters = mdp.evaluate(pi, P, R, gamma)
            res["bellman_V_start"] = float(Vpi[S0])
            res["mc_minus_bellman"] = res["mean_return"] - float(Vpi[S0])
            res["evaluation_iterations"] = iters
            # E3 결과에는 방문 격자를 넣지 않는다(E3의 비교 대상은 리턴과 방정식 값)
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
        # V와 pi는 무작위(임의 입력). 백업 식 자체가 같은지만 확인하므로 진짜 해일 필요가 없다
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
    # gamma가 1에 가까울수록 수축률이 느려 반복 횟수가 는다
    for gamma_ in (0.5, 0.9, 0.99):
        P, R = gw.chapter_grid(0.1)
        _, k = mdp.optimal_values(P, R, gamma_)
        iters[str(gamma_)] = k
    # 백업 한 번의 곱셈 수 S*A*S'. P와 (R+gamma V)의 원소별 곱 (S, A, S')
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
