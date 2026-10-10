"""강화학습 9장 실험. `cd math-ai-matrix-labs && uv run python rl/ch09_dqn/experiments.py`

E0 손계산: 동전 K개(참값 0, +1/-1 반반)의 최댓값 평균, DQN 목표와 Double 목표 한 개씩
E1 실패 1(최댓값의 치우침, 신경망 없이): 참값이 모두 0인 행동 K개, 잡음 N(0,1), 표본 10,000개.
   단일 추정(한 벌로 고르고 매기기) vs 두 벌로 나누기(Double), 정규분포 최댓값 기댓값의 수치적분과 비교.
   참값이 서로 다를 때(한 행동만 0.5)는 Double이 낮춰 잡는지도 본다.
E2 연속 전이가 얼마나 닮았나: 무작위 행동으로 모은 cart-pole 전이에서 이웃한 두 상태의 상관
E3 보폭·복사 간격 고르기: 재생+타깃(단일) 하나로 시드 100~102에서 lr x C 격자, 기준은 마지막 100판 평균 리턴
E4 실패 2(장치 빼기): 다섯 조건 x 시드 0~4, 50,000걸음
E5 구현 대조: NumPy 손 역전파 vs PyTorch autograd, 손 역전파 vs 유한 차분
E6 계산량: 이론 FLOP, 실측 시간, 버퍼 메모리

결과는 results/ch09.json, 로그는 logs/ch09_transitions.jsonl.
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
import rl_ch09_cartpole as cp  # noqa: E402
import rl_ch09_dqn_np as m  # noqa: E402
import rl_ch09_dqn_torch as mt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch09.json"
LOG = HERE / "logs" / "ch09_transitions.jsonl"
SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [100, 101, 102]
KS = [2, 4, 8, 16, 32]
CONDITIONS = {
    "none": dict(use_replay=False, use_target=False, double=False),
    "target_only": dict(use_replay=False, use_target=True, double=False),
    "replay_only": dict(use_replay=True, use_target=False, double=False),
    "replay_target": dict(use_replay=True, use_target=True, double=False),
    "replay_target_double": dict(use_replay=True, use_target=True, double=True),
}


def r6(x):
    if x is None:
        return None
    x = float(x)
    return float(round(x, 6)) if np.isfinite(x) else None


# ------------------------------------------------------------------ E0

def e0_hand():
    coins = {str(K): m.coin_max_exact(K) for K in (1, 2, 3, 4)}
    closed = {str(K): 1 - 2 * 0.5 ** K for K in (1, 2, 3, 4)}
    # DQN 목표 한 개: 보상 1, 사본(타깃망)이 다음 상태에 매긴 점수 [왼쪽 3.0, 오른쪽 3.4]
    r, g = 1.0, 0.99
    q_tgt_next = [3.0, 3.4]
    q_now_next = [5.1, 4.8]   # 지금 망은 왼쪽을 더 높게 본다
    y_single = r + g * max(q_tgt_next)
    a_dbl = int(np.argmax(q_now_next))
    y_double = r + g * q_tgt_next[a_dbl]
    return {"coin_max_mean": coins, "coin_closed_form_1_minus_2_half_pow_K": closed,
            "coin_split_mean": 0.0,
            "target_example": {"r": r, "gamma": g, "q_target_next": q_tgt_next, "q_online_next": q_now_next,
                               "y_single": r6(y_single), "double_pick": ["left", "right"][a_dbl],
                               "y_double": r6(y_double)}}


# ------------------------------------------------------------------ E1

def e1_max_bias():
    n = 10_000
    equal, unequal = {}, {}
    for K in KS:
        res = m.max_bias_experiment(K, n, seed=K)
        res["integral"] = m.expected_max_normal(K)
        equal[str(K)] = {k: r6(v) for k, v in res.items()}
        mu = np.zeros(K)
        mu[0] = 0.5
        res2 = m.max_bias_experiment(K, n, seed=1000 + K, mu=mu)
        unequal[str(K)] = {k: r6(v) for k, v in res2.items()}
    return {"n": n, "noise": "N(0,1)", "equal_true_values_0": equal,
            "one_action_0.5_rest_0": unequal}


# ------------------------------------------------------------------ E2

def e2_correlation():
    env = cp.CartPole(0, 500)
    rng = np.random.default_rng(0)
    S = []
    s = env.reset()
    eps = 0
    while len(S) < 5000:
        s2, _, fell, trunc = env.step(int(rng.integers(2)))
        S.append(s)
        s = s2
        if fell or trunc:
            s = env.reset()
            eps += 1
            S.append(None)  # 판 경계 표시
    pairs = [(a, b) for a, b in zip(S[:-1], S[1:]) if a is not None and b is not None]
    X = np.array([p[0] for p in pairs])
    Y = np.array([p[1] for p in pairs])
    names = ["x", "x_dot", "angle", "angle_dot"]
    lag1 = {nm: r6(np.corrcoef(X[:, i], Y[:, i])[0, 1]) for i, nm in enumerate(names)}
    allS = np.array([q for q in S if q is not None])
    # 연속 32개 묶음 vs 균등 32개 묶음: 묶음 안 각도의 표준편차(넓게 퍼질수록 다양한 장면)
    rng2 = np.random.default_rng(1)
    sd_consec, sd_unif = [], []
    for _ in range(2000):
        i = int(rng2.integers(0, len(allS) - 32))
        sd_consec.append(allS[i:i + 32, 2].std())
        sd_unif.append(allS[rng2.integers(0, len(allS), 32), 2].std())
    return {"policy": "무작위(왼쪽/오른쪽 반반)", "transitions": len(pairs), "episodes": eps,
            "mean_episode_len": r6(len(allS) / max(eps, 1)),
            "lag1_corr": lag1,
            "angle_sd_within_batch32": {"consecutive": r6(np.mean(sd_consec)), "uniform": r6(np.mean(sd_unif))},
            "angle_sd_all": r6(allS[:, 2].std())}


# ------------------------------------------------------------------ E3, E4

def summarize(run, cfg):
    L = np.array(run.ep_len)
    reach = [st for ln, st in zip(run.ep_len, run.ep_end_step) if ln >= cfg.cap]
    half = [i for i, st in enumerate(run.eval_step) if st > cfg.steps // 2]
    q0 = np.array(run.eval_q0)
    g0 = np.array(run.eval_g0)
    return {
        "episodes": int(len(L)),
        "last100_return": r6(L[-100:].mean()) if len(L) else None,
        "step_first_500": int(reach[0]) if reach else None,
        "diverged_at": run.diverged_at,
        "max_abs_q_train": r6(run.max_abs_q),
        "eval_steps": run.eval_step,
        "eval_q0": [r6(x) for x in q0], "eval_g0": [r6(x) for x in g0], "eval_len": [r6(x) for x in run.eval_len],
        "loss": [r6(x) for x in run.loss],
        "loss_max": r6(max(x for x in run.loss if x is not None)) if run.loss else None,
        "q0_max": r6(q0.max()) if len(q0) else None,
        "q0_mean_2nd_half": r6(q0[half].mean()) if half else None,
        "g0_mean_2nd_half": r6(g0[half].mean()) if half else None,
        "gap_mean_2nd_half": r6((q0[half] - g0[half]).mean()) if half else None,
        "frac_eval_q0_over_100": r6(np.mean(q0 > 100.0)) if len(q0) else None,
        "final_eval_len": r6(run.eval_len[-1]) if run.eval_len else None,
        "seconds": r6(run.seconds), "updates": run.updates, "target_copies": run.target_copies,
    }


def e3_sweep():
    grid = {}
    t0 = time.time()
    for lr in (3e-4, 1e-3):
        for C in (100, 500):
            cfg = m.Config(lr=lr, C=C)
            vals = []
            for sd in SWEEP_SEEDS:
                run = m.dqn(sd, cfg=cfg, **CONDITIONS["replay_target"])
                vals.append(np.mean(run.ep_len[-100:]))
            grid[f"lr={lr:g},C={C}"] = {"per_seed": [r6(v) for v in vals], "mean": r6(np.mean(vals))}
    best = max(grid, key=lambda k: grid[k]["mean"])
    lr = float(best.split(",")[0].split("=")[1])
    C = int(best.split(",")[1].split("=")[1])
    return {"seeds": SWEEP_SEEDS, "condition": "replay_target", "criterion": "마지막 100판 평균 리턴",
            "grid": grid, "chosen": best, "seconds": r6(time.time() - t0)}, lr, C


def e4_conditions(lr, C):
    cfg = m.Config(lr=lr, C=C)
    out = {"config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, "seeds": SEEDS, "runs": {}, "summary": {}}
    for name, kw in CONDITIONS.items():
        runs = [summarize(m.dqn(sd, cfg=cfg, **kw), cfg) for sd in SEEDS]
        out["runs"][name] = runs
        last = np.array([r["last100_return"] for r in runs], dtype=float)
        reach = [r["step_first_500"] for r in runs]
        out["summary"][name] = {
            "last100_mean": r6(last.mean()), "last100_min": r6(last.min()), "last100_max": r6(last.max()),
            "last100_per_seed": [r6(x) for x in last],
            "step_first_500_per_seed": reach,
            "n_reached_500": int(sum(x is not None for x in reach)),
            "n_diverged": int(sum(r["diverged_at"] is not None for r in runs)),
            "max_abs_q_per_seed": [r["max_abs_q_train"] for r in runs],
            "max_abs_q_median": r6(np.median([r["max_abs_q_train"] for r in runs])),
            "q0_max_per_seed": [r["q0_max"] for r in runs],
            "loss_max_per_seed": [r["loss_max"] for r in runs],
            "q0_peak_step_per_seed": [r["eval_steps"][int(np.argmax(r["eval_q0"]))] for r in runs],
            "q0_median_at": {str(st): r6(np.median([r["eval_q0"][r["eval_steps"].index(st)] for r in runs]))
                             for st in (2_500, 7_500, 12_500, 25_000, 50_000)},
            "g0_median_at": {str(st): r6(np.median([r["eval_g0"][r["eval_steps"].index(st)] for r in runs]))
                             for st in (2_500, 7_500, 12_500, 25_000, 50_000)},
            "loss_max_median": r6(np.median([r["loss_max"] for r in runs])),
            "q0_mean_2nd_half": r6(np.mean([r["q0_mean_2nd_half"] for r in runs])),
            "g0_mean_2nd_half": r6(np.mean([r["g0_mean_2nd_half"] for r in runs])),
            "gap_mean_2nd_half": r6(np.mean([r["gap_mean_2nd_half"] for r in runs])),
            "gap_2nd_half_per_seed": [r["gap_mean_2nd_half"] for r in runs],
            "frac_eval_q0_over_100": r6(np.mean([r["frac_eval_q0_over_100"] for r in runs])),
            "final_eval_len_per_seed": [r["final_eval_len"] for r in runs],
            "us_per_env_step": r6(1e6 * sum(r["seconds"] for r in runs) / (cfg.steps * len(runs))),
        }
    return out, cfg


# ------------------------------------------------------------------ E5

def e5_impl():
    cfg = m.Config()
    rng = np.random.default_rng(5)
    net = m.MLP([4, 64, 64, 2], 5)
    tgt = m.MLP([4, 64, 64, 2], 6)
    buf = m.ReplayBuffer(1000, 7)
    env = cp.CartPole(5, 500)
    s = env.reset()
    for _ in range(300):
        a = int(rng.integers(2))
        s2, r, fell, trunc = env.step(a)
        buf.add(s, a, r, s2, fell)
        s = env.reset() if (fell or trunc) else s2
    batch = buf.sample_uniform(32)
    S, A, R, S2, F = batch
    res = {}
    for double in (False, True):
        y = m.td_target(net, tgt, R, S2, F, cfg.gamma, double)
        loss_np, g_np, _ = net.grad(m.feat(S), A, y)
        loss_t, g_t, y_t = mt.loss_and_grads(net, tgt, batch, cfg.gamma, double)
        rel = max(float(np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a)))) for a, b in zip(g_np, g_t) if np.abs(a).max() > 0)
        absd = max(float(np.max(np.abs(a - b))) for a, b in zip(g_np, g_t))
        res["double" if double else "single"] = {"loss_np": loss_np, "loss_torch": loss_t,
                                                  "loss_abs_diff": abs(loss_np - loss_t),
                                                  "target_max_abs_diff": float(np.max(np.abs(y - y_t))),
                                                  "grad_max_abs_diff": absd, "grad_max_rel_diff": rel}
    # 유한 차분: 몇 개 가중치만
    y = m.td_target(net, tgt, R, S2, F, cfg.gamma, False)
    _, g_np, _ = net.grad(m.feat(S), A, y)
    h = 1e-6
    errs = []
    for li, (i, j) in [(0, (0, 3)), (1, (5, 7)), (2, (10, 1))]:
        W = net.W[li]
        old = W[i, j]
        W[i, j] = old + h
        lp = net.grad(m.feat(S), A, y)[0]
        W[i, j] = old - h
        lm = net.grad(m.feat(S), A, y)[0]
        W[i, j] = old
        errs.append(abs((lp - lm) / (2 * h) - g_np[li][i, j]))
    res["finite_diff_max_abs_err"] = float(max(errs))
    res["rtol"] = 1e-6
    return res


# ------------------------------------------------------------------ E6

def time_one_update(n=2000):
    """미니배치 32개로 한 번 고치는 시간: NumPy 손 구현 vs PyTorch(float32, nn.Linear, torch.optim.Adam)."""
    rng = np.random.default_rng(0)
    S, S2 = rng.normal(size=(32, 4)), rng.normal(size=(32, 4))
    A, R, F = rng.integers(0, 2, 32), np.ones(32), np.zeros(32)
    net = m.MLP([4, 64, 64, 2], 0)
    tgt = net.clone()
    opt = m.Adam(net.params(), 1e-3)
    t = time.time()
    for _ in range(n):
        y = m.td_target(net, tgt, R, S2, F, 0.99, False)
        _, g, _ = net.grad(m.feat(S), A, y)
        opt.step(net.params(), g)
    t_np = (time.time() - t) / n

    def mk():
        return torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 2))
    torch.manual_seed(0)
    model, tmodel = mk(), mk()
    o = torch.optim.Adam(model.parameters(), 1e-3)
    St, S2t = torch.tensor(S / m.SCALE, dtype=torch.float32), torch.tensor(S2 / m.SCALE, dtype=torch.float32)
    At, Rt = torch.tensor(A), torch.tensor(R, dtype=torch.float32)
    t = time.time()
    for _ in range(n):
        with torch.no_grad():
            y = Rt + 0.99 * tmodel(S2t).max(1).values
        q = model(St).gather(1, At[:, None]).squeeze(1)
        loss = 0.5 * ((y - q) ** 2).mean()
        o.zero_grad()
        loss.backward()
        o.step()
    t_torch = (time.time() - t) / n
    return {"updates_timed": n, "us_per_update_numpy": r6(1e6 * t_np), "us_per_update_torch_cpu": r6(1e6 * t_torch)}


def e6_cost(cfg, summary):
    sizes = [4, cfg.hidden, cfg.hidden, 2]
    macs = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    fwd = 2 * macs   # 곱하고 더하기 = 2 FLOP
    B = cfg.batch
    per_step = {
        "act_select_forward": fwd,
        "update_forward_backward_S": 3 * fwd * B,   # 순전파 1 + 역전파 약 2
        "target_forward_S2": fwd * B,
        "double_extra_forward_S2": fwd * B,
    }
    buf = m.ReplayBuffer(cfg.buffer, 0)
    return {"layer_sizes": sizes, "params": int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))),
            "flop_forward_per_state": fwd, "flop_per_env_step_breakdown": per_step,
            "flop_per_env_step_single": int(per_step["act_select_forward"] + per_step["update_forward_backward_S"]
                                             + per_step["target_forward_S2"]),
            "flop_per_env_step_double": int(sum(per_step.values())),
            "buffer_bytes": buf.nbytes(), "buffer_capacity": cfg.buffer,
            "us_per_env_step": {k: v["us_per_env_step"] for k, v in summary.items()},
            "one_update_timing": time_one_update()}


# ------------------------------------------------------------------ main

def main():
    t_all = time.time()
    res = {"meta": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                    "machine": f"{platform.system()} {platform.machine()}", "torch_threads": torch.get_num_threads()}}
    res["E0_hand"] = e0_hand()
    res["E1_max_bias"] = e1_max_bias()
    res["E2_correlation"] = e2_correlation()
    res["E3_sweep"], lr, C = e3_sweep()
    res["E4_conditions"], cfg = e4_conditions(lr, C)
    res["E5_impl"] = e5_impl()
    res["E6_cost"] = e6_cost(cfg, res["E4_conditions"]["summary"])

    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("w") as f:
        rows = []
        small = m.Config(lr=lr, C=C, steps=1_200, eval_every=10 ** 9)
        m.dqn(0, cfg=small, log=rows.append, **CONDITIONS["replay_target"])   # 처음 1,200걸음(대부분 무작위 행동)
        for row in rows:
            row["info"]["condition"] = "replay_target, 학습 초기"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        run = m.dqn(0, cfg=cfg, **CONDITIONS["replay_target"])
        env = cp.CartPole(900_000, cfg.cap)
        s = env.reset()
        t = 0
        qs = []
        while True:
            q = run.net.forward(m.feat(s)[None])[0]
            a = int(np.argmax(q))
            qs += [float(x) for x in q]
            s2, r, fell, trunc = env.step(a)
            f.write(json.dumps({"seed": 0, "episode": "eval", "t": t, "s": [round(float(x), 4) for x in s], "a": a,
                                "r": r, "s_next": [round(float(x), 4) for x in s2], "done": bool(fell or trunc),
                                "info": {"condition": "replay_target, 50,000걸음 학습 뒤 탐욕 정책",
                                         "q": [round(float(x), 3) for x in q], "fell": bool(fell),
                                         "truncated": bool(trunc)}}, ensure_ascii=False) + "\n")
            s, t = s2, t + 1
            if fell or trunc:
                break
        res["E4_conditions"]["logged_eval_episode_len"] = t
        res["E4_conditions"]["logged_eval_episode_q_min"] = r6(min(qs))
        res["E4_conditions"]["logged_eval_episode_q_max"] = r6(max(qs))

    res["meta"]["seconds"] = r6(time.time() - t_all)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["E0_hand"], ensure_ascii=False))
    print(json.dumps(res["E1_max_bias"], ensure_ascii=False)[:2500])
    print(json.dumps(res["E2_correlation"], ensure_ascii=False))
    print(json.dumps(res["E3_sweep"], ensure_ascii=False))
    for k, v in res["E4_conditions"]["summary"].items():
        print(k, json.dumps(v, ensure_ascii=False))
    print(json.dumps(res["E5_impl"], ensure_ascii=False))
    print(json.dumps(res["E6_cost"], ensure_ascii=False))
    print("seconds", res["meta"]["seconds"])


if __name__ == "__main__":
    main()
