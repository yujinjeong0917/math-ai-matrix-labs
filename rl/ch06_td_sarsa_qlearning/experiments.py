"""강화학습 6장 실험. `cd math-ai-matrix-labs && uv run python rl/ch06_td_sarsa_qlearning/experiments.py`

E0 손계산: 5장 첫 화면의 복도 네 판(같은 경로)에 TD(0)(alpha 0.5, gamma 0.9)를 걸어 값이 바뀐 갱신만 모은다
E1 실패 1(MC는 느리다): Sutton(1988)과 같은 꼴의 5칸 무작위 보행, V0 = 0.5, gamma 1, 시드 100개 x 1,000판
   TD(0)와 상수 alpha MC를 alpha {0.01, 0.05, 0.1, 0.2}에서, 5장의 1/N 평균 MC도 함께. RMSE(다섯 칸)
E2 실패 1b(끝나지 않는 판): 절벽 격자, Q = 0, epsilon = 0, 동률이면 번호가 가장 작은 행동
   MC 제어 vs Sarsa vs Q-러닝, 한 걸음 보상 -1과 0 두 경우, 판 3개, 상한 10,000걸음
E3 실패 2(온·오프폴리시): 절벽 격자 4 x 10, 절벽 -50, epsilon 0.1, alpha 0.5, gamma 1, 500판
   Sarsa / Q-러닝 / Expected Sarsa 시드 100개, 상수 alpha MC 제어 시드 10개. 온라인 리턴, 추락, 탐욕 정책
   epsilon 0.01로 줄였을 때(Sarsa, Q-러닝)도 함께
E4 부트스트랩의 대가: 결정적 사슬 10칸, alpha 1. TD(0)가 정확해지기까지의 판 수 vs MC
E5 구현 대조: NumPy vs PyTorch(전이 기록 재생, 시드 100개 동시 TD(0)·MC)
E6 계산량: 갱신 한 번의 시간, 메모리

결과는 results/ch06.json, 로그는 logs/ch06_transitions.jsonl.
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
import rl_ch06_envs as E  # noqa: E402
import rl_ch06_td_np as T  # noqa: E402
import rl_ch06_td_torch as TT  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch06.json"
LOG = HERE / "logs" / "ch06_transitions.jsonl"
# E1에서 비교할 학습률 alpha
ALPHAS = (0.01, 0.05, 0.1, 0.2)


def r6(x):
    return float(round(float(x), 6))


def ci(a):
    a = np.asarray(a, dtype=float)
    m, sd = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
    # 식: 95% 신뢰구간 반폭 = 1.96 x sd / sqrt(n)(정규 근사). sd는 표본표준편차(ddof = 1)
    h = 1.96 * sd / np.sqrt(len(a))
    return {"mean": r6(m), "lo": r6(m - h), "hi": r6(m + h), "sd": r6(sd), "median": r6(np.median(a)),
            "min": r6(a.min()), "max": r6(a.max())}


# ---------------- E0 ----------------

def e0_hand():
    alpha, gamma = 0.5, 0.9
    V = [0.0, 0.0, 0.0]
    rows, per_ep = [], []
    for k, path in enumerate(E.CH05_FIRST_FOUR):
        tr = E.corridor_transitions(path)
        n_changed = 0
        for t, (s, r, s2, done) in enumerate(tr):
            old = V[s]
            # 갱신 전에 읽어 둔 다음 가치(표에 적기 위한 값). td0_update 안의 목표와 같은 값이다
            nxt = 0.0 if done else V[s2]
            # 식: V(s) <- V(s) + 0.5 [r + 0.9 V(s') - V(s)] (alpha 0.5, gamma 0.9)
            new = T.td0_update(V, s, r, s2, done, alpha, gamma)
            # 값이 실제로 바뀐 갱신만 표에 적는다(목표 = 현재 값이면 TD 오차가 0)
            if new != old:
                n_changed += 1
                rows.append({"episode": k + 1, "t": t + 1, "state": "AB"[s], "old": r6(old), "r": r,
                             "next_value": r6(nxt), "target": r6(r + gamma * nxt), "new": r6(new)})
        per_ep.append({"episode": k + 1, "path": path, "updates": len(tr), "changed": n_changed,
                       "V_A": r6(V[0]), "V_B": r6(V[1])})
    # 같은 네 판에 5장의 첫 방문 MC(1/N 평균)를 걸어 비교한다
    mc = T.mc_sample_average([E.corridor_transitions(p) for p in E.CH05_FIRST_FOUR], 3, gamma)
    return {"alpha": alpha, "gamma": gamma, "V_exact": {"A": 0.647482, "B": 0.791367},
            "changed_updates": rows, "after_each_episode": per_ep,
            "mc_first_visit_after_4": {"A": r6(mc[0]), "B": r6(mc[1])},
            "note": "경로는 5장 첫 화면(시드 0의 첫 네 판)과 같다. V_exact는 3장의 연립방정식 값."}


# ---------------- E1 ----------------

def rmse(Vhist, truth, keep):
    H = np.asarray(Vhist)
    # 식: RMSE_k = sqrt(mean_{s in keep} (V_k(s) - V*(s))^2). Vhist[k]는 k+1번째 판 뒤의 V, keep은 종료 칸을 뺀 5칸. 반환 길이 K
    return np.sqrt(np.mean((H[:, keep] - truth[keep]) ** 2, axis=1))


def e1_walk(seeds=range(100), K=1000, n=5):
    truth = E.walk_true_values(n)
    keep = np.arange(1, n + 1)
    marks = (1, 5, 10, 25, 50, 100, 300, 1000)
    curves = {}
    first_ep_changed = {"td": [], "mc": []}
    lengths = []
    t0 = time.perf_counter()
    for sd in seeds:
        # 시드 sd마다 판 K개를 한 번 만들어 모든 방법에 똑같이 넣는다(공정한 비교)
        uni = E.Uniforms(np.random.default_rng(1000 + sd))
        eps = [E.walk_episode(uni, n) for _ in range(K)]
        lengths.extend(len(e) for e in eps)
        for a in ALPHAS:
            for name, fn in (("td", T.td0), ("mc", T.mc_constant_alpha)):
                rec = []
                # gamma = 1, 모든 칸의 초기값 V0 = 0.5. record에는 판마다 V의 사본이 쌓인다
                fn(eps, n + 2, a, 1.0, v0=0.5, record=rec)
                curves.setdefault(f"{name}_{a}", []).append(rmse(rec, truth, keep))
        rec = []
        T.mc_sample_average(eps, n + 2, 1.0, v0=0.5, record=rec)
        curves.setdefault("mc_avg", []).append(rmse(rec, truth, keep))
        # 첫 판 뒤에 값이 바뀐 칸 수 (alpha 0.1)
        for name, fn in (("td", T.td0), ("mc", T.mc_constant_alpha)):
            # 첫 판만 넣고 0.5에서 바뀐 칸 수(alpha 0.1). TD는 종료 바로 옆 한 칸, MC는 지나간 칸 전부(test_td_changes_one_state... 참고)
            V1 = fn(eps[:1], n + 2, 0.1, 1.0, v0=0.5)
            first_ep_changed[name].append(int(np.sum(V1[keep] != 0.5)))
    secs = time.perf_counter() - t0
    out = {"n_states": n, "v0": 0.5, "gamma": 1.0, "seeds": len(list(seeds)), "episodes": K,
           "true_values": [r6(x) for x in truth[keep]],
           "episode_length": {"mean": r6(np.mean(lengths)), "max": int(np.max(lengths))},
           "by_method": {}, "seconds": r6(secs)}
    for key, cs in curves.items():
        C = np.stack(cs)  # [seeds, K]
        out["by_method"][key] = {
            str(m): ci(C[:, m - 1]) for m in marks
        }
        out["by_method"][key]["mean_over_first_100"] = r6(C[:, :100].mean())
    # 판 수별로 가장 좋은 alpha
    best = {}
    for m in marks:
        row = {}
        for name in ("td", "mc"):
            vals = {a: out["by_method"][f"{name}_{a}"][str(m)]["mean"] for a in ALPHAS}
            # 그 판 수에서 RMSE 평균이 가장 낮은 alpha. TD와 MC에서 따로 고른다
            a_best = min(vals, key=vals.get)
            row[name] = {"alpha": a_best, "rmse": vals[a_best]}
        best[str(m)] = row
    out["best_alpha_by_episodes"] = best
    # 같은 판 묶음에서 시드별 비교: 100판 시점, 각 방법의 (100판 기준) 최적 alpha
    a_td, a_mc = best["100"]["td"]["alpha"], best["100"]["mc"]["alpha"]
    # [:, 99]는 100번째 판 뒤(0부터 세므로 99)
    td100 = np.stack(curves[f"td_{a_td}"])[:, 99]
    mc100 = np.stack(curves[f"mc_{a_mc}"])[:, 99]
    out["paired_100"] = {"td_alpha": a_td, "mc_alpha": a_mc, "td_better_seeds": int(np.sum(td100 < mc100)),
                         "of": len(td100)}
    out["first_episode_changed_states_alpha0.1"] = {k: ci(v) for k, v in first_ep_changed.items()}
    return out


# ---------------- E2 ----------------

def e2_stuck(f):
    out = {}
    for step in (-1.0, 0.0):
        env = E.Cliff(step=step)
        row = {}
        for m in ("mc", "sarsa", "q"):
            # 갇힌 MC 판의 앞 6줄과 마지막 줄만 로그로 남긴다(전체는 10,000줄)
            log = [] if (m == "mc" and step == -1.0) else None
            # alpha 0.5, gamma 1, epsilon 0, 시드 0. tie = "first"라 Q = 0에서는 항상 행동 0(위)을 골라 맨 윗줄 벽에서 제자리
            o = T.control(env, m, 3, 0.5, 1.0, 0.0, 0, max_steps=10_000, log=log, tie="first")
            row[m] = {"truncated_of_3": o["truncated"], "lengths": o["lengths"], "updates": o["updates"],
                      "nonzero_Q": int(np.sum(o["Q"] != 0.0)), "returns": [r6(x) for x in o["returns"]]}
            if log is not None:
                ep0 = [x for x in log if x[0] == 0]
                keep = ep0[:6] + ep0[-1:]
                for (ep, t, s, a, r, s2, done, fell) in keep:
                    info = {"env": "cliff4x10", "method": "mc_control_eps0", "step_reward": step}
                    if t == len(ep0) - 1:
                        info["truncated"] = True
                        info["note"] = f"상한 {len(ep0)}걸음에서 잘림, 리턴을 모름"
                    f.write(json.dumps({"seed": 0, "episode": ep, "t": t, "s": s, "a": a, "r": r, "s_next": s2,
                                        "done": bool(done), "info": info}, ensure_ascii=False) + "\n")
                row[m]["first_states"] = [x[2] for x in ep0[:8]]
        out[f"step_{step}"] = row
    return out


# ---------------- E3 ----------------

def classify(env, path):
    """경로가 지나간 가장 높은 줄(0 = 맨 위). 아래 줄(절벽 줄)은 h-1."""
    return int(min(s // env.w for s in path))


def run_cliff(env, method, seeds, eps, episodes=500):
    # R / R100: 시드별 온라인 리턴 평균(전체 / 마지막 100판), F / F100: 추락 횟수, G / L: 탐욕 경로의 리턴 / 길이, reach: 탐욕 경로가 도착한 시드 수, rows: 경로가 닿은 가장 높은 줄별 시드 수
    R, R100, F, F100, G, L, reach, rows, curves = [], [], [], [], [], [], 0, {}, []
    trunc = []
    for sd in seeds:
        # alpha 0.5, gamma 1, epsilon = eps(E3: 0.1 또는 0.01). 시드 sd로 난수를 고정한다
        o = T.control(env, method, episodes, 0.5, 1.0, eps, sd)
        rets = np.array(o["returns"])
        R.append(rets.mean())
        R100.append(rets[-100:].mean())
        F.append(sum(o["falls"]))
        F100.append(sum(o["falls"][-100:]))
        trunc.append(o["truncated"])
        # 25판 묶음 평균 온라인 리턴 곡선(500판이면 점 20개)
        curves.append([rets[i:i + 25].mean() for i in range(0, episodes, 25)])
        # 학습이 끝난 Q로 epsilon = 0 탐욕 경로를 한 번 걸어 도착하는지, 어느 줄을 지나는지 본다
        g, n, ok, path = T.greedy_rollout(env, o["Q"])
        reach += ok
        if ok:
            G.append(g)
            L.append(n)
            k = classify(env, path)
            rows[str(k)] = rows.get(str(k), 0) + 1
    C = np.array(curves)
    return {"seeds": len(list(seeds)), "online_return_all": ci(R), "online_return_last100": ci(R100),
            "falls_total": ci(F), "falls_last100": ci(F100), "truncated_episodes_total": int(sum(trunc)),
            "greedy_reached": int(reach), "greedy_loops": int(len(list(seeds)) - reach),
            "greedy_return_when_reached": ci(G) if G else None,
            "greedy_length_counts": {str(k): int(v) for k, v in zip(*np.unique(L, return_counts=True))} if L else {},
            "greedy_highest_row_counts": rows,
            "curve_block25_mean": [r6(x) for x in C.mean(axis=0)]}


def e3_cliff():
    env = E.Cliff()
    # 최적 리턴: 절벽 바로 위 줄(2번 줄)로 위 1 + 오른쪽 w-1 + 아래 1 = w+1걸음, 걸음마다 -1이라 -(w+1) = -11
    # 한 줄 위(1번)는 2걸음, 맨 위(0번)는 4걸음 더 걸어 -(w+3), -(w+5)
    out = {"h": env.h, "w": env.w, "cliff_reward": env.cliff_r, "step_reward": env.step_r, "alpha": 0.5,
           "gamma": 1.0, "episodes": 500, "greedy_eval": "epsilon 0, 동률이면 번호가 가장 작은 행동, 상한 100걸음",
           "optimal_return": -(env.w + 1), "row_paths": {"2": -(env.w + 1), "1": -(env.w + 3), "0": -(env.w + 5)}}
    t0 = time.perf_counter()
    for m in ("sarsa", "q", "expected"):
        out[f"{m}_eps0.1"] = run_cliff(env, m, range(100), 0.1)
    for m in ("sarsa", "q"):
        out[f"{m}_eps0.01"] = run_cliff(env, m, range(100), 0.01)
    out["seconds_td"] = r6(time.perf_counter() - t0)
    t0 = time.perf_counter()
    # MC 제어는 시드 10개(TD 계열은 100개)
    out["mc_eps0.1"] = run_cliff(env, "mc", range(10), 0.1)
    out["seconds_mc"] = r6(time.perf_counter() - t0)
    return out


def e3_log(f):
    """시드 0의 Sarsa·Q-러닝 첫 판과 마지막 판을 공통 스키마로 남긴다."""
    env = E.Cliff()
    for m in ("sarsa", "q"):
        log = []
        T.control(env, m, 500, 0.5, 1.0, 0.1, 0, log=log)
        for (ep, t, s, a, r, s2, done, fell) in log:
            # 500판 중 첫 판(0)과 마지막 판(499)만 남긴다
            if ep not in (0, 499):
                continue
            f.write(json.dumps({"seed": 0, "episode": ep, "t": t, "s": s, "a": a, "r": r, "s_next": s2,
                                "done": bool(done), "info": {"env": "cliff4x10", "method": m, "fell": bool(fell)}},
                               ensure_ascii=False) + "\n")


# ---------------- E4 ----------------

def e4_chain(n=10, gamma=0.9):
    tr = []
    for s in range(n - 1):
        s2, r, done = E.chain_step(s, 1, n)
        tr.append((s, r, s2, done))
    # 식: V*(s) = gamma^(n-2-s)(오른쪽으로만 가는 정책, 보상 1은 마지막 걸음). 종료 칸 n-1은 0
    truth = np.array([gamma ** (n - 2 - s) for s in range(n - 1)] + [0.0])
    need = {}
    for name, fn in (("td", T.td0), ("mc", T.mc_constant_alpha)):
        for k in range(1, 3 * n):
            # alpha = 1로 같은 판을 k번 반복해 정확해지는 최소 판 수 k를 찾는다(허용 오차 1e-12)
            V = fn([tr] * k, n, 1.0, gamma)
            if np.max(np.abs(V - truth)) < 1e-12:
                need[name] = k
                break
    # TD(0)는 한 판에서 앞에서부터 고치므로 보상 1에 닿은 마지막 칸 하나만 0이 아니게 된다
    V1 = T.td0([tr], n, 1.0, gamma)
    # q_learning_chain_maxerr: Q-러닝의 고정점이 Q*인지(n = 6, 2000판, epsilon 0.3). [:5]는 종료 칸 행을 뺀 것
    return {"n_states": n, "nonterminal": n - 1, "gamma": gamma, "alpha": 1.0, "episodes_to_exact": need,
            "td_after_1_nonzero": int(np.sum(V1 != 0)),
            "q_learning_chain_maxerr": r6(np.max(np.abs(T.q_learning_chain(6, 0.9, 0.5, 2000, 0.3, 0)[:5]
                                                       - E.chain_q_star(6, 0.9)[:5])))}


# ---------------- E5 ----------------

def e5_match():
    env = E.Cliff()
    out = {}
    for m in ("sarsa", "q", "expected"):
        log = []
        o = T.control(env, m, 50, 0.5, 1.0, 0.1, 0, log=log)
        # control()이 남긴 기록을 NumPy로 재생한 Q와 PyTorch로 재생한 Q를 비교한다(순서 동일, atol 1e-12). replay_vs_run은 재생 Q와 실제 학습 Q의 차
        Qn = T.replay(log, env.S, env.A, m, 0.5, 1.0, 0.1)
        Qt = TT.replay_t(log, env.S, env.A, m, 0.5, 1.0, 0.1).numpy()
        out[m] = {"transitions": len(log), "replay_vs_run_maxdiff": float(np.max(np.abs(Qn - o["Q"]))),
                  "numpy_vs_torch_maxdiff": float(np.max(np.abs(Qn - Qt))),
                  "pass": bool(np.allclose(Qn, Qt, atol=1e-12, rtol=0))}
    # 시드 100개 x 판 100개의 무작위 보행. 배치 TD / MC(torch)와 시드별 NumPy 결과를 판마다 비교한다
    runs = []
    for sd in range(100):
        uni = E.Uniforms(np.random.default_rng(1000 + sd))
        runs.append([E.walk_episode(uni) for _ in range(100)])
    for name, fn, fb in (("td0", T.td0, TT.td0_batch), ("mc", T.mc_constant_alpha, TT.mc_batch)):
        Hn = []
        for b in range(100):
            rec = []
            fn(runs[b], 7, 0.1, 1.0, v0=0.5, record=rec)
            Hn.append(rec)
        # [시드, 판, 칸]을 [판, 시드, 칸]으로 바꿔 torch 쪽 hist [K, B, n_states]와 축을 맞춘다
        Hn = np.transpose(np.array(Hn), (1, 0, 2))
        Ht = fb(runs, 7, 0.1, 1.0, v0=0.5).numpy()
        out[f"walk_{name}_100seeds"] = {"maxdiff": float(np.max(np.abs(Hn - Ht))),
                                        "pass": bool(np.allclose(Hn, Ht, atol=1e-12, rtol=0))}
    return out


# ---------------- E6 ----------------

def e6_cost():
    env = E.Cliff()
    out = {}
    for m in ("sarsa", "q", "expected"):
        t0 = time.perf_counter()
        o = T.control(env, m, 500, 0.5, 1.0, 0.1, 0)
        dt = time.perf_counter() - t0
        # steps = 갱신 걸음 총수. us_per_step = 걸음당 마이크로초(1e6 x 초 / 걸음). 시간은 기계마다 다르다
        steps = sum(o["lengths"])
        out[m] = {"steps": int(steps), "seconds": r6(dt), "us_per_step": r6(1e6 * dt / steps)}
    uni = E.Uniforms(np.random.default_rng(7))
    eps = [E.walk_episode(uni) for _ in range(20_000)]
    steps = sum(len(e) for e in eps)
    for name, fn in (("td0", T.td0), ("mc_const", T.mc_constant_alpha)):
        t0 = time.perf_counter()
        fn(eps, 7, 0.1, 1.0, v0=0.5)
        dt = time.perf_counter() - t0
        out[f"walk_{name}"] = {"steps": int(steps), "seconds": r6(dt), "us_per_step": r6(1e6 * dt / steps)}
    out["memory"] = {"Q_table_floats": env.S * env.A, "V_table_floats_walk": 7,
                     "mc_episode_storage": "판 길이 T만큼 (s, a, r)를 들고 있다가 끝에서 거꾸로 더한다",
                     "longest_walk_episode": int(max(len(e) for e in eps))}
    return out


def main():
    t_all = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "machine": platform.machine(), "system": platform.system(), "threads": 2}}
    LOG.parent.mkdir(exist_ok=True)
    OUT.parent.mkdir(exist_ok=True)
    res["E0_hand"] = e0_hand()
    res["E1_walk"] = e1_walk()
    with LOG.open("w") as f:
        res["E2_stuck"] = e2_stuck(f)
        e3_log(f)
    res["E3_cliff"] = e3_cliff()
    res["E4_chain"] = e4_chain()
    res["E5_match"] = e5_match()
    res["E6_cost"] = e6_cost()
    res["seconds_total"] = r6(time.perf_counter() - t_all)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E0_hand", "E4_chain", "E5_match")}, ensure_ascii=False, indent=1))
    print("total seconds", res["seconds_total"])


if __name__ == "__main__":
    main()
