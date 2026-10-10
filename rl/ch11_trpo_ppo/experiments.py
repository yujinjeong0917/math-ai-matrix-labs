"""강화학습 11장 실험. `cd math-ai-matrix-labs && uv run python rl/ch11_trpo_ppo/experiments.py`

E0 손계산: 행동 둘(왼쪽/오른쪽), 손잡이 숫자 하나(z)로 오른쪽 확률 = sigmoid(z). 같은 손잡이 이동이
   어디서 출발하느냐에 따라 확률을 얼마나 바꾸나, PPO의 잘라 내기가 이득을 어디서 멈추나
E1 실패 1(보폭): 바닐라 정책경사 학습률 3종(10, 100, 1000) x 시드 10개. 갱신 전후 평균 KL과 리턴
E2 TRPO(delta = 0.01, 논문 값) 선탐색 있음/없음 x 시드 10개
E3 PPO eps = 0.1, 0.2, 0.3과 잘라 내기 없음(eps = inf), Adam 학습률 0.05 x 시드 10개
   + Adam 학습률 0.01, 0.05, 0.2에서 eps = 0.2와 잘라 내기 없음 비교
E4 실패 2(시드): 설정마다 시드 {0..4}와 {5..9}의 Welch t 검정(양측), 10개를 5+5로 나누는 126가지 전부,
   평균의 부트스트랩 95% 구간(백분위 방법, 1만 번), 시드 3개만 골랐을 때 TRPO와 PPO의 순위가 뒤집히는 비율
E5 계산량: 갱신 한 번에 묶음 전체를 훑는 횟수와 실측 시간
E6 구현 대조: NumPy vs PyTorch
E7 재현: 같은 시드로 두 번 돌리면 같은 숫자가 나오나

모든 설정은 같은 환경(cart-pole, 최대 200걸음), 같은 묶음 크기(갱신마다 16판), 같은 갱신 횟수(60번),
같은 이득 추정(할인 0.99 리턴을 묶음 안에서 평균 0, 표준편차 1로 맞춤), 같은 초기값(w = 0)을 쓴다.
"최종 리턴"은 마지막 10번 갱신의 묶음 평균 판 길이를 다시 평균 낸 값이다.

결과는 results/ch11.json, 로그는 logs/ch11_updates.jsonl(갱신별 KL·리턴)과 logs/ch11_episodes.jsonl(판 기록).
"""

import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_ch11_cartpole as cp  # noqa: E402
import rl_ch11_ppo_np as m  # noqa: E402
import rl_ch11_ppo_torch as mt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch11.json"
LOG_UPD = HERE / "logs" / "ch11_updates.jsonl"
LOG_EP = HERE / "logs" / "ch11_episodes.jsonl"
SEEDS = list(range(10))
N_ITERS = 60
INF = float("inf")

CONFIGS = {
    "pg_lr10": ("pg", {"lr": 10.0}),
    "pg_lr100": ("pg", {"lr": 100.0}),
    "pg_lr1000": ("pg", {"lr": 1000.0}),
    "trpo": ("trpo", {"delta": 0.01}),
    "trpo_no_linesearch": ("trpo", {"delta": 0.01, "line_search": False}),
    "ppo_eps0.1": ("ppo", {"eps": 0.1, "adam_lr": 0.05}),
    "ppo_eps0.2": ("ppo", {"eps": 0.2, "adam_lr": 0.05}),
    "ppo_eps0.3": ("ppo", {"eps": 0.3, "adam_lr": 0.05}),
    "ppo_noclip": ("ppo", {"eps": INF, "adam_lr": 0.05}),
}
LR_SWEEP = {f"ppo_{tag}_adam{lr}": ("ppo", {"eps": eps, "adam_lr": lr})
            for lr in (0.01, 0.2) for tag, eps in (("eps0.2", 0.2), ("noclip", INF))}


def r4(x):
    return None if x is None else float(round(x, 4))


def final_return(h):
    return float(np.mean(h["ret"][-10:]))


# ---------------- E0 ----------------
def e0_hand():
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731

    def kl(p, q):
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    out = {"p_at_z": {str(z): r4(sig(z)) for z in (0, 1, 4, 5)}}
    out["same_step_from_0"] = {"p_before": r4(sig(0)), "p_after": r4(sig(1)), "kl": r4(kl(sig(0), sig(1)))}
    out["same_step_from_4"] = {"p_before": r4(sig(4)), "p_after": r4(sig(5)), "kl": r4(kl(sig(4), sig(5)))}
    # 오른쪽이 좋았다(이득 +1). 오른쪽 확률을 0.5에서 0.6, 0.73으로 올렸을 때 비율과 잘라 낸 목적
    rows = []
    for p_new in (0.6, sig(1)):
        r = p_new / 0.5
        rows.append({"p_new": r4(p_new), "ratio": r4(r), "unclipped": r4(r * 1.0),
                     "clipped": r4(min(r * 1.0, min(max(r, 0.8), 1.2) * 1.0))})
    out["clip_rows_adv_plus1"] = rows
    out["z_for_ratio_1.2"] = r4(math.log(0.6 / 0.4))
    # 0.5에서 출발해 KL이 정확히 0.01(TRPO의 delta)이 되는 오른쪽 확률: 4q(1-q) = e^{-0.02}
    q = 0.5 + math.sqrt(0.25 - math.exp(-0.02) / 4)
    out["p_at_kl_0.01_from_0.5"] = {"p": r4(q), "kl_check": r4(kl(0.5, q))}
    # 이득이 -1이면(오른쪽이 나빴다) 비율을 0.8 아래로 내려도 더 얻는 게 없다
    out["clip_adv_minus1_ratio0.6"] = r4(min(0.6 * -1.0, min(max(0.6, 0.8), 1.2) * -1.0))
    return out


# ---------------- 학습 묶음 ----------------
def run_all(configs):
    runs = {}
    for name, (algo, kw) in configs.items():
        t0 = time.perf_counter()
        runs[name] = [m.train(algo, s, n_iters=N_ITERS, **kw) for s in SEEDS]
        print(f"{name}: {time.perf_counter() - t0:.1f}s", flush=True)
    return runs


def summarize(hs):
    finals = [final_return(h) for h in hs]
    mean, lo, hi = m.bootstrap_ci(finals, seed=0)
    cols = [m.collapsed(h["ret"], h["steps"]) for h in hs]
    kls = np.concatenate([h["kl"] for h in hs])
    return {
        "final_per_seed": [r4(x) for x in finals],
        "final_mean": r4(mean), "final_ci95": [r4(lo), r4(hi)],
        "final_std": r4(np.std(finals, ddof=1)),
        "final_min": r4(min(finals)), "final_max": r4(max(finals)),
        "collapsed_seeds": [s for s, c in zip(SEEDS, cols) if c is not None],
        "collapse_iter": {str(s): c for s, c in zip(SEEDS, cols) if c is not None},
        "kl_median": r4(np.median(kls)), "kl_p95": r4(np.quantile(kls, 0.95)), "kl_max": r4(kls.max()),
        "frac_updates_kl_gt_0.01": r4(np.mean(kls > 0.01)),
        "seeds_final_below_20": int(sum(f < 20 for f in finals)),
        "seeds_final_below_20_list": [s for s, f in zip(SEEDS, finals) if f < 20],
        "drop_events_total": int(sum(m.drop_events(h["ret"]) for h in hs)),
        "seeds_with_drop": int(sum(m.drop_events(h["ret"]) > 0 for h in hs)),
        "total_env_steps_mean": int(np.mean([h["steps"][-1] for h in hs])),
        "update_ms_median": r4(np.median(np.concatenate([h["update_ms"] for h in hs]))),
    }


def kl_vs_drop(hs, big=0.1, small=0.01):
    """갱신 i의 KL과 다음 묶음의 리턴 변화 ret[i+1] - ret[i]. KL이 큰 갱신 뒤에 리턴이 더 떨어지나."""
    kl = np.concatenate([np.array(h["kl"][:-1]) for h in hs])
    d = np.concatenate([np.diff(h["ret"]) for h in hs])
    out = {}
    for tag, mask in (("kl_gt_" + str(big), kl > big), ("kl_le_" + str(small), kl <= small),
                      ("between", (kl > small) & (kl <= big))):
        out[tag] = {"n": int(mask.sum()), "mean_return_change": r4(d[mask].mean()) if mask.any() else None,
                    "frac_drop_over_50": r4(np.mean(d[mask] < -50)) if mask.any() else None}
    return out


def seed_split(finals):
    finals = np.asarray(finals)
    t, df, p = m.welch_t(finals[:5], finals[5:])
    ps, diffs = [], []
    for g in itertools.combinations(range(10), 5):
        if 0 not in g:  # {g, 나머지}와 {나머지, g}는 같은 나눔이라 한 번만 센다
            continue
        rest = [i for i in range(10) if i not in g]
        ps.append(m.welch_t(finals[list(g)], finals[rest])[2])
        diffs.append(abs(finals[list(g)].mean() - finals[rest].mean()))
    ps = np.array(ps)
    return {"split_0_4_vs_5_9": {"mean_a": r4(finals[:5].mean()), "mean_b": r4(finals[5:].mean()),
                                 "t": r4(t), "df": r4(df), "p_two_sided": r4(p)},
            "n_splits": int(ps.size), "splits_p_lt_0.05": int((ps < 0.05).sum()),
            "min_p": r4(ps.min()), "max_abs_mean_diff": r4(max(diffs))}


def three_seed_flip(fa, fb):
    """같은 시드 3개를 골라 두 설정을 비교할 때(120가지), A의 평균이 B보다 높게 나오는 비율."""
    fa, fb = np.asarray(fa), np.asarray(fb)
    wins = [fa[list(c)].mean() > fb[list(c)].mean() for c in itertools.combinations(range(10), 3)]
    return {"n_triples": len(wins), "a_wins": int(sum(wins)), "a_mean_10": r4(fa.mean()), "b_mean_10": r4(fb.mean()),
            "welch_10_vs_10": dict(zip(("t", "df", "p_two_sided"), map(r4, m.welch_t(fa, fb))))}


# ---------------- 로그 ----------------
def write_logs(runs):
    with LOG_UPD.open("w") as f:
        for name in ("pg_lr1000", "ppo_noclip_adam0.2", "ppo_eps0.2_adam0.2", "ppo_eps0.2", "trpo"):
            for s in (0, 1):
                h = runs[name][s]
                for i, (ret, st, kl) in enumerate(zip(h["ret"], h["steps"], h["kl"])):
                    f.write(json.dumps({"config": name, "seed": s, "update": i, "return": ret, "env_steps": st,
                                        "kl_after_update": r4(kl) if math.isfinite(kl) else None}) + "\n")
    # 판 기록(공통 스키마): 시드 0의 PPO(eps 0.2) 최종 정책과, 한쪽 행동으로 굳어 버린 바닐라(학습률 1000) 시드의 최종 정책
    col = [s for s in SEEDS if final_return(runs["pg_lr1000"][s]) < 20]
    pick = [("ppo_eps0.2", 0)] + ([("pg_lr1000", col[0])] if col else [])
    with LOG_EP.open("w") as f:
        for name, s in pick:
            w = np.array(runs[name][s]["final_w"])
            rng = np.random.default_rng(10_000 + s)
            st = cp.reset(1, rng)
            for t in range(cp.MAX_STEPS):
                p = float(m.sigmoid(m.features(st) @ w)[0])
                a = int(rng.random() < p)
                nxt, done = cp.step(st, np.array([a]))
                f.write(json.dumps({"seed": s, "episode": 0, "t": t, "s": [r4(v) for v in st[0]], "a": a, "r": 1.0,
                                    "s_next": [r4(v) for v in nxt[0]], "done": bool(done[0]),
                                    "info": {"config": name, "p_right": r4(p)}}) + "\n")
                st = nxt
                if done[0]:
                    break


# ---------------- E5 ----------------
def e5_compute(runs):
    tr = [i for h in runs["trpo"] for i in h["info"]]
    cg = np.array([i["cg_iters"] for i in tr])
    bt = np.array([i["backtracks"] for i in tr])
    acc = np.array([i["accepted"] for i in tr])
    # 묶음 전체를 훑는 횟수(행렬-벡터 곱 하나 = 1회). TRPO: 기울기 1 + 켤레기울기 k + s^T F s 1 + 선탐색 시도마다 2(KL, 대리 목적)
    trpo_pass = 1 + cg + 1 + 1 + 2 * (bt + acc)  # 1: 선탐색 전 대리 목적 L0
    return {
        "pg": {"passes_per_update": 1, "ms_median": r4(np.median([x for h in runs["pg_lr100"] for x in h["update_ms"]]))},
        "trpo": {"cg_iters_values": sorted(set(cg.tolist())), "cg_iters_mean": r4(cg.mean()),
                 "backtracks_hist": {str(k): int((bt == k).sum()) for k in sorted(set(bt.tolist()))},
                 "rejected_updates": int((~acc).sum()), "n_updates": int(acc.size),
                 "passes_per_update_mean": r4(trpo_pass.mean()),
                 "ms_median": r4(np.median([x for h in runs["trpo"] for x in h["update_ms"]]))},
        "ppo": {"passes_per_update": 10, "minibatch_steps": 40,
                "ms_median": r4(np.median([x for h in runs["ppo_eps0.2"] for x in h["update_ms"]])),
                "clip_frac_median_eps0.2": r4(np.median([i["clip_frac"] for h in runs["ppo_eps0.2"] for i in h["info"]]))},
        "note": "실측 시간은 기기마다 다르다. 정책 매개변수가 5개뿐이라 TRPO의 켤레기울기가 5번 안에 끝난다.",
    }


# ---------------- E6 ----------------
def _first_flip(hn, wt, upto):
    """두 구현의 매개변수로 같은 난수열을 다시 걸어, 갈라진 갱신의 묶음에서 처음 행동이 달라진 판과 걸음을 찾는다."""
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    w1, w2 = np.zeros(5), np.zeros(5)
    for k in range(upto + 1):
        b1, b2 = m.collect(w1, 16, rng1), m.collect(w2, 16, rng2)
        if k == upto:
            for e in range(16):
                a1, a2 = b1["a"][b1["ep"] == e], b2["a"][b2["ep"] == e]
                L = min(len(a1), len(a2))
                d = np.flatnonzero(a1[:L] != a2[:L])
                if d.size:
                    return {"update": k, "episode": e, "step": int(d[0]),
                            "episode_len_numpy": int(len(a1)), "episode_len_torch": int(len(a2))}
        w1, w2 = np.array(hn["w"][k]), wt[k]
    return None


def e6_torch():
    rng = np.random.default_rng(3)
    w = rng.normal(0, 0.5, 5)
    batch = m.collect(w, 16, np.random.default_rng(4))
    adv = m.advantages(batch)
    w2 = w + rng.normal(0, 0.3, 5)  # 비율이 1에서 벗어나 잘라 내기가 실제로 걸리는 점
    out = {}
    for eps in (0.1, 0.2, INF):
        ln, gn = m.ppo_loss_grad_w(w2, batch["phi"], batch["a"], batch["logp_old"], adv, eps)
        lt, gt = mt.loss_and_grad_w(w2, batch["phi"], batch["a"], batch["logp_old"], adv, eps)
        r = np.exp(m.log_prob(w2, batch["phi"], batch["a"]) - batch["logp_old"])
        out[f"eps={eps}"] = {"loss_rel_diff": float(abs(ln - lt) / abs(ln)),
                             "grad_max_rel_diff": float(np.max(np.abs(gn - gt) / np.abs(gt))),
                             "clipped_frac": r4(np.mean((r < 1 - eps) | (r > 1 + eps))) if math.isfinite(eps) else 0.0}
    v = rng.normal(size=5)
    fn = m.fisher_vector_product(w, batch["phi"], v)
    ft = mt.fisher_vector_product(w, batch["phi"], v)
    out["fisher_vector_product_max_rel_diff"] = float(np.max(np.abs(fn - ft) / np.abs(ft)))
    F = m.fisher_matrix(w, batch["phi"])
    g = m.pg_gradient(w, batch, adv)
    x, k = m.conjugate_gradient(lambda u: F @ u, g, iters=10)
    out["cg_vs_solve_max_rel_diff"] = float(np.max(np.abs(x - np.linalg.solve(F, g)) / np.abs(np.linalg.solve(F, g))))
    out["cg_iters"] = k
    # PPO 갱신 한 번(10에폭 x 미니배치 4개 = Adam 40걸음)
    wn, _ = m.ppo_update(w, batch, adv, 0.2, 10, 4, m.Adam(5, 0.05), np.random.default_rng(7))
    wt = mt.ppo_update(w, batch, adv, 0.2, 10, 4, 0.05, np.random.default_rng(7))
    out["ppo_update_w_max_abs_diff"] = float(np.max(np.abs(wn - wt)))
    # 학습 루프 전체: 시드 0, 갱신 20번
    hn = m.train("ppo", 0, n_iters=20, eps=0.2, adam_lr=0.05)
    rt, wt = mt.train_ppo(0, 20, eps=0.2, adam_lr=0.05)
    dw = np.max(np.abs(np.array(hn["w"]) - wt), axis=1)
    same = [a == b for a, b in zip(hn["ret"], rt)]
    first_diff = same.index(False) if False in same else None
    out["train_loop_20_updates"] = {
        "first_update_with_different_return": first_diff,  # 0부터 센 갱신 번호
        "w_max_abs_diff_before_divergence": float(dw[: first_diff - 1].max()) if first_diff else float(dw.max()),
        "w_max_abs_diff_by_update": [float(x) for x in dw],
        "returns_numpy": hn["ret"], "returns_torch": rt,
        "first_action_flip": _first_flip(hn, wt, first_diff) if first_diff else None}
    out["rtol"] = 1e-9
    return out


def e7_repeat():
    a = m.train("ppo", 3, n_iters=20, eps=0.2, adam_lr=0.05)
    b = m.train("ppo", 3, n_iters=20, eps=0.2, adam_lr=0.05)
    c = m.train("ppo", 4, n_iters=20, eps=0.2, adam_lr=0.05)
    return {"same_seed_identical": a["ret"] == b["ret"] and a["final_w"] == b["final_w"],
            "other_seed_identical": a["ret"] == c["ret"],
            "seed3_returns_last5": a["ret"][-5:], "seed4_returns_last5": c["ret"][-5:]}


def main():
    t0 = time.perf_counter()
    LOG_UPD.parent.mkdir(exist_ok=True)
    OUT.parent.mkdir(exist_ok=True)
    res = {"E0_hand": e0_hand()}
    runs = run_all(CONFIGS)
    sweep = run_all(LR_SWEEP)
    res["E1_E2_E3_summary"] = {k: summarize(v) for k, v in runs.items()}
    res["E3_adam_lr_sweep"] = {k: summarize(v) for k, v in sweep.items()}
    res["E1_kl_vs_next_return_change"] = {k: kl_vs_drop(runs[k]) for k in ("pg_lr10", "pg_lr100", "pg_lr1000", "ppo_noclip", "ppo_eps0.2", "trpo")}
    res["E1_kl_vs_next_return_change"].update({k: kl_vs_drop(sweep[k]) for k in ("ppo_noclip_adam0.2", "ppo_eps0.2_adam0.2")})
    res["curves_mean_by_update"] = {k: [r4(x) for x in np.mean([h["ret"] for h in v], axis=0)] for k, v in {**runs, **sweep}.items()}
    res["curves_seed_min_by_update"] = {k: [r4(x) for x in np.min([h["ret"] for h in v], axis=0)] for k, v in {**runs, **sweep}.items()}
    res["curves_per_seed"] = {k: [[r4(x) for x in h["ret"]] for h in v] for k, v in {**runs, **sweep}.items()}
    res["env_steps_by_update_seed0"] = {k: v[0]["steps"] for k, v in runs.items()}
    pool = {**runs, **sweep}
    finals = {k: [final_return(h) for h in v] for k, v in pool.items()}
    res["E4_seed_split"] = {k: seed_split(f) for k, f in finals.items()}
    res["E4_three_seed_flip"] = {"trpo_vs_ppo_eps0.2": three_seed_flip(finals["trpo"], finals["ppo_eps0.2"]),
                                 "ppo_eps0.2_vs_ppo_eps0.1": three_seed_flip(finals["ppo_eps0.2"], finals["ppo_eps0.1"]),
                                 "pg_lr100_vs_trpo": three_seed_flip(finals["pg_lr100"], finals["trpo"])}
    res["E1_stuck"] = {f"{k}_seed{s}": {"final_w": [r4(x) for x in pool[k][s]["final_w"]], **{a: r4(b) for a, b in m.stuck_diagnosis(pool[k][s]["final_w"]).items()},
                                         "kl_first5": [r4(x) for x in pool[k][s]["kl"][:5]], "ret_first8": pool[k][s]["ret"][:8]}
                       for k in ("pg_lr1000", "ppo_noclip_adam0.2") for s in SEEDS if final_return(pool[k][s]) < 20}
    res["E1_stuck_reference_ppo_eps0.2_seed0"] = {a: r4(b) for a, b in m.stuck_diagnosis(runs["ppo_eps0.2"][0]["final_w"]).items()}
    res["E5_compute"] = e5_compute(runs)
    write_logs(pool)
    res["E6_torch"] = e6_torch()
    res["E7_repeat"] = e7_repeat()
    res["setup"] = {"seeds": SEEDS, "n_iters": N_ITERS, "episodes_per_update": 16, "max_steps": cp.MAX_STEPS,
                    "gamma": m.GAMMA, "ppo_epochs": 10, "ppo_minibatches": 4, "trpo_delta": 0.01, "trpo_cg_iters": 10,
                    "collapse_rule": "리턴 < 0.5 x 그때까지 최고치(최고치 50 이상일 때), 이후 1만 걸음 안의 모든 갱신에서도 그 아래",
                    "final_return": "마지막 10번 갱신의 묶음 평균 판 길이의 평균", "bootstrap": "백분위, 10,000번, 시드 0"}
    res["env"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "machine": f"{platform.system()} {platform.machine()}", "threads": 2,
                  "total_seconds": r4(time.perf_counter() - t0)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E1_E2_E3_summary",)}, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
