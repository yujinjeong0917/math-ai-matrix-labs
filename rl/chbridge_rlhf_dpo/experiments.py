"""강화학습 연결 장 실험. `cd math-ai-matrix-labs && uv run python rl/chbridge_rlhf_dpo/experiments.py`

장난감 문장 공간(단어 6개, 길이 8, 6^8 = 1,679,616개)에서 숨은 참 보상 r*를 정해 두고, 그 보상으로 만든
"둘 중 어느 쪽이 나은가" 비교 500쌍만 학습에 준다. 시드마다 비교 500쌍을 새로 뽑고, 같은 쌍을 RLHF와 DPO가 함께 쓴다.

E0 손계산: BT 확률, 참 보상 예, 시드 0의 보상 모델 점수(소수 둘째 자리), 단어별 기울인 확률, 검증기 기본 성공률
E1 보상 모델: 단어 개수 모델(6개 점수)과 앞 단어까지 보는 모델(336개 점수, L2 두 가지)의 비교 정확도
E2 실패(과최적화): 단어 개수 보상 모델에 대해 KL 계수 beta = 0, 0.01, 0.1, 1로 정책경사 600걸음 x 시드 10개
E3 정확한 최적해: pi ∝ pi_ref exp(score / beta)를 전체 공간에서 계산(학습된 보상, 참 보상). 정책경사가 그 해에
   닿았는지, beta가 "참 보상 대 KL" 곡선 자체를 바꾸는지(Gao 등 2022의 관찰을 이 장난감에서 다시 재기)
E4 DPO: (a) 정책을 단어 개수 모델과 같은 표현력으로 묶은 DPO vs 단어 개수 RLHF의 정확한 최적해,
   (b) 336칸 전체 DPO vs 앞 단어까지 보는 보상 모델의 정확한 최적해. 참 보상, KL, 걸음 수, 시드 10개
E5 계산량: 한 걸음에 계산하는 단어 로그확률 수와 실측 시간(평가 없이)
E6 RLVR: 장난감 산수 검증기(첫 단어 a가 문제, 두 번째 단어가 (a + 2) mod 6이면 정답)로 정책경사
E7 구현 대조: NumPy vs PyTorch
E8 재현: 같은 시드로 두 번 돌리면 같은 숫자가 나오나

결과는 results/chbridge.json, 로그는 logs/chbridge_samples.jsonl(생성 문장, 공통 스키마)과 logs/chbridge_curves.jsonl.
"""

import os

for _k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rl_chbridge_rlhf_dpo_np as m  # noqa: E402
import rl_chbridge_rlhf_dpo_torch as mt  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "chbridge.json"
LOG_SAMPLES = HERE / "logs" / "chbridge_samples.jsonl"
LOG_CURVES = HERE / "logs" / "chbridge_curves.jsonl"
SEEDS = list(range(10))
N_PAIRS = 500
PG_STEPS, PG_LR, PG_BATCH, PG_EVAL = 600, 0.01, 64, 10
PG_BETAS = [0.0, 0.01, 0.1, 1.0]
DPO_STEPS, DPO_LR, DPO_EVAL = 1000, 0.01, 50
DPO_BETAS = [0.01, 0.1, 1.0]
TILT_BETAS = [10.0, 3.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
EDGE_L2 = [1e-3, 1e-2]
KL_BUDGETS = [0.5, 1.0, 3.0, 6.0]


def r(x, k=4):
    return None if x is None else float(round(float(x), k))


def word_str(y):
    return "".join(m.WORDS[int(v)] for v in y)


def pairs_for(seed):
    return m.make_pairs(N_PAIRS, np.random.default_rng([seed, 1]))


def heldout():
    rng = np.random.default_rng(12345)
    A = m.sample(m.zeros_theta(), 4000, rng)
    B = m.sample(m.zeros_theta(), 4000, rng)
    return A, B


def mean_sd(xs):
    xs = np.asarray(xs, float)
    return {"mean": r(xs.mean()), "sd": r(xs.std(ddof=1)) if xs.size > 1 else 0.0, "min": r(xs.min()), "max": r(xs.max())}


def interp_at_kl(curve, kl_target):
    """학습 곡선(KL이 대체로 커지는 순서)에서 KL = kl_target일 때의 참 보상을 처음 지나는 구간에서 선형 보간."""
    ks = [c["kl"] for c in curve]
    rs = [c["r_true"] for c in curve]
    for i in range(1, len(ks)):
        if ks[i - 1] <= kl_target <= ks[i] and ks[i] > ks[i - 1]:
            w = (kl_target - ks[i - 1]) / (ks[i] - ks[i - 1])
            return rs[i - 1] + w * (rs[i] - rs[i - 1])
    return None


def best_within(points, budget):
    """(kl, r_true) 점들 중 KL <= budget인 것의 참 보상 최댓값."""
    ok = [p[1] for p in points if p[0] <= budget + 1e-12]
    return max(ok) if ok else None


def main():
    t_all = time.time()
    res = {}
    tab = m._enum_tables()
    Y_all = m.all_sequences()
    A_ho, B_ho = heldout()
    curves_log = []
    samples_log = []

    # ---------- E0 손계산 ----------
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    Yw0, Yl0 = pairs_for(0)
    u0 = m.bt_reward_model_fit(Yw0, Yl0)
    u0r = np.round(u0, 2)
    spam = np.array([[5] * 8])
    best = np.array([[5, 5, 4, 4, 3, 3, 2, 2]])
    tilt_hand = {}
    for beta in (1.0, 0.1):
        w = np.exp(u0r / beta)
        tilt_hand[str(beta)] = {"weights": [r(x) for x in w], "probs": [r(x) for x in w / w.sum()]}
    res["E0_hand"] = {
        "bt_gap1": r(sig(1.0)),
        "bt_gap0.5": r(sig(0.5)),
        "W": m.W.tolist(),
        "penalty": m.PENALTY,
        "words": m.WORDS,
        "r_true_spam": r(m.true_reward(spam)[0]),
        "r_true_best": r(m.true_reward(best)[0]),
        "r_true_max_over_space": r(tab["r_true"].max()),
        "n_argmax": int((np.abs(tab["r_true"] - tab["r_true"].max()) < 1e-9).sum()),
        "E_ref_r_true": r(tab["r_true"].mean()),
        "E_ref_uniq": r(tab["uniq"].mean()),
        "seed0_u": [r(x) for x in u0],
        "seed0_u_round2": u0r.tolist(),
        "seed0_rhat_spam_round2": r(8 * u0r[5]),
        "seed0_rhat_best_round2": r(float(m.counts(best)[0] @ u0r)),
        "seed0_first_pairs": [
            {"win": word_str(Yw0[i]), "lose": word_str(Yl0[i]),
             "r_true_win": r(m.true_reward(Yw0[i:i + 1])[0]), "r_true_lose": r(m.true_reward(Yl0[i:i + 1])[0])}
            for i in range(3)
        ],
        "tilt_per_word_seed0_round2": tilt_hand,
        "rlvr_base_success": r(1 / 6),
        "space_size": int(m.V**m.L),
    }

    # ---------- E1 보상 모델 ----------
    rm = {}
    for seed in SEEDS:
        Yw, Yl = pairs_for(seed)
        u = m.bt_reward_model_fit(Yw, Yl)
        row = {"u": [r(x) for x in u], "acc_heldout": r(m.rm_accuracy(u, A_ho, B_ho)),
               "train_label_agrees_with_true": r(float((m.true_reward(Yw) > m.true_reward(Yl)).mean())),
               "order_matches_W": bool(np.all(np.argsort(u) == np.arange(m.V)))}
        for l2 in EDGE_L2:
            phi = m.bt_edge_fit(Yw, Yl, l2=l2)
            d = m.edge_features(A_ho) @ phi - m.edge_features(B_ho) @ phi
            row[f"edge_acc_heldout_l2={l2}"] = r(m.pair_accuracy(d, A_ho, B_ho))
        rm[seed] = row
    ho_true_better = m.true_reward(A_ho) - m.true_reward(B_ho)
    res["E1_reward_model"] = {
        "per_seed": rm,
        "acc_linear": mean_sd([rm[s]["acc_heldout"] for s in SEEDS]),
        **{f"acc_edge_l2={l2}": mean_sd([rm[s][f"edge_acc_heldout_l2={l2}"] for s in SEEDS]) for l2 in EDGE_L2},
        "train_labels_agree_with_true": mean_sd([rm[s]["train_label_agrees_with_true"] for s in SEEDS]),
        "heldout_pairs": 4000,
        "heldout_ties_removed": int((np.abs(ho_true_better) <= 1e-12).sum()),
        "seeds_with_u_order_equal_W": int(sum(rm[s]["order_matches_W"] for s in SEEDS)),
        "u_bar_minus_ga_mean": r(np.mean([rm[s]["u"][5] - rm[s]["u"][0] for s in SEEDS])),
    }
    print(f"E1 done {time.time() - t_all:.1f}s")

    # ---------- E2 과최적화: KL 정규 정책경사 ----------
    pg = {str(b): {} for b in PG_BETAS}
    pg_theta_final = {}
    for seed in SEEDS:
        Yw, Yl = pairs_for(seed)
        u = m.bt_reward_model_fit(Yw, Yl)
        for beta in PG_BETAS:
            acc_track = []

            def log_samples(k, theta, rng, beta=beta, seed=seed, u=u, acc_track=acc_track):
                if beta == 0.0:  # 정책이 만든 문장 쌍에서 보상 모델이 여전히 맞히나
                    rr = np.random.default_rng([seed, 9, k])
                    A = m.sample(theta, 1000, rr)
                    B = m.sample(theta, 1000, rr)
                    rt = m.true_reward(A) - m.true_reward(B)
                    keep = np.abs(rt) > 1e-12
                    acc_track.append({"step": k, "acc": r(m.rm_accuracy(u, A, B)) if keep.sum() > 20 else None,
                                      "non_tied_pairs": int(keep.sum())})
                if seed == 0 and beta in (0.0, 1.0) and k in (0, 60, 150, 300, 600):
                    rr = np.random.default_rng([seed, 7, k])
                    Ys = m.sample(theta, 4, rr)
                    for ep, y in enumerate(Ys):
                        for t in range(m.L):
                            samples_log.append({
                                "seed": seed, "episode": ep, "t": t, "s": [int(v) for v in y[:t]], "a": int(y[t]),
                                "r": r(float((m.counts(y[None]) @ u)[0])) if t == m.L - 1 else 0.0,
                                "s_next": [int(v) for v in y[:t + 1]], "done": t == m.L - 1,
                                "info": {"config": f"rlhf_pg_beta{beta}", "step": k, "text": word_str(y),
                                         "r_true": r(m.true_reward(y[None])[0]) if t == m.L - 1 else None}})

            theta, curve = m.kl_regularized_pg(u, beta, seed, steps=PG_STEPS, batch=PG_BATCH, lr=PG_LR,
                                               eval_every=PG_EVAL, log_samples=log_samples)
            pg_theta_final[(seed, beta)] = theta
            peak = max(curve, key=lambda c: c["r_true"])
            last = curve[-1]
            pg[str(beta)][seed] = {
                "peak_r_true": r(peak["r_true"]), "peak_step": peak["step"], "peak_kl": r(peak["kl"]),
                "final_r_true": r(last["r_true"]), "final_r_hat": r(last["r_hat"]), "final_kl": r(last["kl"]),
                "final_uniq": r(last["uniq"]), "final_p_top": r(last["p_top"]),
                "final_top_text": word_str(Y_all[int(np.argmax(np.exp(m.logprob(theta, Y_all))))]),
                "curve": [{"step": c["step"], "kl": r(c["kl"]), "r_true": r(c["r_true"]), "r_hat": r(c["r_hat"]),
                           "uniq": r(c["uniq"])} for c in curve],
                "rm_acc_on_policy_pairs": acc_track if beta == 0.0 else None,
            }
            for c in curve:
                curves_log.append({"method": "rlhf_pg", "beta": beta, "seed": seed, **{k: r(v) if isinstance(v, float) else v
                                   for k, v in c.items() if k != "exp_counts"}})
    summ = {}
    for b in PG_BETAS:
        rows = [pg[str(b)][s] for s in SEEDS]
        summ[str(b)] = {
            k: mean_sd([x[k] for x in rows])
            for k in ("peak_r_true", "peak_step", "peak_kl", "final_r_true", "final_r_hat", "final_kl", "final_uniq",
                      "final_p_top")
        }
        summ[str(b)]["drop_peak_to_final"] = mean_sd([x["peak_r_true"] - x["final_r_true"] for x in rows])
        summ[str(b)]["seeds_final_below_ref"] = int(sum(x["final_r_true"] < tab["r_true"].mean() for x in rows))
        summ[str(b)]["top_texts"] = sorted({x["final_top_text"] for x in rows})
        mean_curve = []
        for i in range(len(rows[0]["curve"])):
            mean_curve.append({"step": rows[0]["curve"][i]["step"],
                               **{k: r(np.mean([x["curve"][i][k] for x in rows])) for k in ("kl", "r_true", "r_hat", "uniq")}})
        summ[str(b)]["mean_curve"] = mean_curve
    res["E2_pg_summary"] = summ
    res["E2_pg_per_seed"] = {b: {s: {k: v for k, v in x.items() if k != "curve"} for s, x in d.items()} for b, d in pg.items()}
    acc0 = [pg["0.0"][s]["rm_acc_on_policy_pairs"] for s in SEEDS]
    acc_by_step = {}
    for i, a in enumerate(acc0[0]):
        vals = [acc0[s][i]["acc"] for s in SEEDS if acc0[s][i]["acc"] is not None]
        acc_by_step[a["step"]] = {"mean": r(np.mean(vals)) if vals else None, "n_seeds": len(vals),
                                  "mean_non_tied_pairs": r(np.mean([acc0[s][i]["non_tied_pairs"] for s in SEEDS]), 1)}
    res["E2_rm_acc_on_policy_pairs_beta0"] = acc_by_step
    print(f"E2 done {time.time() - t_all:.1f}s")

    # ---------- E3 정확한 최적해와 KL 계수의 역할 ----------
    tilt_lin = {}
    tilt_true = {}
    pg_vs_tilt = {}
    for seed in SEEDS:
        Yw, Yl = pairs_for(seed)
        u = m.bt_reward_model_fit(Yw, Yl)
        rhat_all = tab["counts"] @ u
        tilt_lin[seed] = {str(b): {k: r(v) for k, v in m.tilt_stats(rhat_all, b).items()} for b in TILT_BETAS}
        for b in (0.1, 1.0):
            th_star = m.tilt_theta_linear(u, b)
            pg_vs_tilt.setdefault(str(b), []).append({
                "kl_pg_to_opt": r(m.kl_between(pg_theta_final[(seed, b)], th_star), 6),
                "pg_r_true": pg[str(b)][seed]["final_r_true"], "opt_r_true": tilt_lin[seed][str(b)]["r_true"],
                "pg_kl": pg[str(b)][seed]["final_kl"], "opt_kl": tilt_lin[seed][str(b)]["kl"]})
    for b in TILT_BETAS:
        tilt_true[str(b)] = {k: r(v) for k, v in m.tilt_stats(tab["r_true"], b).items()}
    tilt_lin_mean = {str(b): {k: r(np.mean([tilt_lin[s][str(b)][k] for s in SEEDS])) for k in ("r_true", "r_score", "kl", "uniq")}
                     for b in TILT_BETAS}
    best_tilt = [max(tilt_lin[s].values(), key=lambda x: x["r_true"]) for s in SEEDS]
    # beta가 곡선을 바꾸나: 같은 KL에서의 참 보상
    kl_targets = [0.1, 0.3, 1.0, 2.0, 4.0]
    same_kl = {}
    for b in PG_BETAS:
        same_kl[str(b)] = {}
        for kt in kl_targets:
            vals = [interp_at_kl(pg[str(b)][s]["curve"], kt) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            same_kl[str(b)][str(kt)] = {"mean": r(np.mean(vals)) if vals else None, "n_seeds": len(vals)}
    same_kl["tilt_linear_rm"] = {}
    for kt in kl_targets:
        vals = []
        for s in SEEDS:
            pts = sorted([(v["kl"], v["r_true"]) for v in tilt_lin[s].values()])
            vals.append(interp_at_kl([{"kl": 0.0, "r_true": float(tab["r_true"].mean())}] +
                                     [{"kl": k, "r_true": rt} for k, rt in pts], kt))
        vals = [v for v in vals if v is not None]
        same_kl["tilt_linear_rm"][str(kt)] = {"mean": r(np.mean(vals)) if vals else None, "n_seeds": len(vals)}
    res["E3_exact"] = {
        "tilt_linear_rm_mean": tilt_lin_mean,
        "tilt_true_reward": tilt_true,
        "best_tilt_linear_per_seed": {"r_true": mean_sd([x["r_true"] for x in best_tilt]),
                                      "kl": mean_sd([x["kl"] for x in best_tilt])},
        "pg_final_vs_exact_opt": {b: {"kl_pg_to_opt": mean_sd([x["kl_pg_to_opt"] for x in v]),
                                      "pg_r_true": mean_sd([x["pg_r_true"] for x in v]),
                                      "opt_r_true": mean_sd([x["opt_r_true"] for x in v]),
                                      "pg_kl": mean_sd([x["pg_kl"] for x in v]),
                                      "opt_kl": mean_sd([x["opt_kl"] for x in v])} for b, v in pg_vs_tilt.items()},
        "r_true_at_same_kl": same_kl,
    }
    print(f"E3 done {time.time() - t_all:.1f}s")

    # ---------- E4 DPO ----------
    dpo = {str(b): {} for b in DPO_BETAS}
    shared = {str(b): [] for b in DPO_BETAS}
    edge_tilt = {str(l2): {} for l2 in EDGE_L2}
    for seed in SEEDS:
        Yw, Yl = pairs_for(seed)
        u = m.bt_reward_model_fit(Yw, Yl)
        u_noreg = m.bt_reward_model_fit(Yw, Yl, l2=1e-10)
        for beta in DPO_BETAS:
            # (a) 표현력을 묶은 DPO
            th_s, a = m.train_dpo_shared(Yw, Yl, beta, steps=3000, lr=0.05)
            th_opt = m.tilt_theta_linear(u_noreg, beta)
            st = m.exact_stats(th_s, u)
            g_last = m.dpo_grad_theta(th_s, Yw, Yl, beta)[1].sum(axis=(0, 1))
            shared[str(beta)].append({
                "kl_dpo_to_rlhf_opt": r(m.kl_between(th_s, th_opt), 8),
                "dpo_r_true": r(st["r_true"]), "dpo_kl": r(st["kl"]),
                "opt_r_true": r(m.tilt_stats(tab["counts"] @ u_noreg, beta)["r_true"]),
                "grad_norm_end": float(np.abs(g_last).max()),
                "beta_a_minus_u_shiftfree": r(np.abs((beta * a - beta * a.mean()) - (u_noreg - u_noreg.mean())).max(), 8)})
            # (b) 336칸 전체 DPO
            theta = m.zeros_theta()
            opt = m.Adam(theta.shape, DPO_LR)
            curve = []
            for k in range(DPO_STEPS + 1):
                if k % DPO_EVAL == 0:
                    s_ = m.exact_stats(theta, u)
                    d = m.implicit_reward(theta, A_ho, beta) - m.implicit_reward(theta, B_ho, beta)
                    lw, ll = m.logprob(theta, Yw), m.logprob(theta, Yl)
                    curve.append({"step": k, "kl": r(s_["kl"]), "r_true": r(s_["r_true"]), "uniq": r(s_["uniq"]),
                                  "acc_heldout": r(m.pair_accuracy(d, A_ho, B_ho)),
                                  "dpo_loss": r(m.dpo_grad_theta(theta, Yw, Yl, beta)[0]),
                                  "train_pair_acc": r(float((lw > ll).mean())),
                                  "mean_logp_win": r(lw.mean()), "mean_logp_lose": r(ll.mean())})
                    curves_log.append({"method": "dpo_full", "beta": beta, "seed": seed, **curve[-1]})
                    if seed == 0 and beta == 0.1 and k in (0, 200, 1000):
                        Ys = m.sample(theta, 4, np.random.default_rng([seed, 8, k]))
                        for ep, y in enumerate(Ys):
                            for t in range(m.L):
                                samples_log.append({
                                    "seed": seed, "episode": ep, "t": t, "s": [int(v) for v in y[:t]], "a": int(y[t]),
                                    "r": r(m.implicit_reward(theta, y[None], beta)[0]) if t == m.L - 1 else 0.0,
                                    "s_next": [int(v) for v in y[:t + 1]], "done": t == m.L - 1,
                                    "info": {"config": "dpo_full_beta0.1", "step": k, "text": word_str(y),
                                             "r_true": r(m.true_reward(y[None])[0]) if t == m.L - 1 else None}})
                if k == DPO_STEPS:
                    break
                _, g = m.dpo_grad_theta(theta, Yw, Yl, beta)
                theta = opt.step(theta, -g)
            peak = max(curve, key=lambda c: c["r_true"])
            dpo[str(beta)][seed] = {"curve": curve, "peak_r_true": peak["r_true"], "peak_step": peak["step"],
                                    "peak_kl": peak["kl"], "final": curve[-1]}
        for l2 in EDGE_L2:
            phi = m.bt_edge_fit(Yw, Yl, l2=l2)
            sc = m.edge_score_all(phi)
            edge_tilt[str(l2)][seed] = {str(b): {k: r(v) for k, v in m.tilt_stats(sc, b).items()} for b in TILT_BETAS}
        print(f"  E4 seed {seed} {time.time() - t_all:.1f}s")

    dsum = {}
    for b in DPO_BETAS:
        rows = [dpo[str(b)][s] for s in SEEDS]
        mean_curve = [{"step": rows[0]["curve"][i]["step"],
                       **{k: r(np.mean([x["curve"][i][k] for x in rows]))
                          for k in ("kl", "r_true", "uniq", "acc_heldout", "dpo_loss", "train_pair_acc",
                                    "mean_logp_win", "mean_logp_lose")}}
                      for i in range(len(rows[0]["curve"]))]
        dsum[str(b)] = {
            "peak_r_true": mean_sd([x["peak_r_true"] for x in rows]),
            "peak_step": mean_sd([x["peak_step"] for x in rows]),
            "peak_kl": mean_sd([x["peak_kl"] for x in rows]),
            "final_r_true": mean_sd([x["final"]["r_true"] for x in rows]),
            "final_kl": mean_sd([x["final"]["kl"] for x in rows]),
            "final_uniq": mean_sd([x["final"]["uniq"] for x in rows]),
            "final_acc_heldout": mean_sd([x["final"]["acc_heldout"] for x in rows]),
            "kl_at_step": {str(st): mean_sd([x["curve"][st // DPO_EVAL]["kl"] for x in rows]) for st in (50, 200, 500, 1000)},
            "r_true_at_step": {str(st): mean_sd([x["curve"][st // DPO_EVAL]["r_true"] for x in rows]) for st in (50, 200, 500, 1000)},
            "mean_curve": mean_curve,
        }
    edge_mean = {str(l2): {str(b): {k: r(np.mean([edge_tilt[str(l2)][s][str(b)][k] for s in SEEDS]))
                                    for k in ("r_true", "kl", "uniq")} for b in TILT_BETAS} for l2 in EDGE_L2}
    # 같은 KL 예산 안에서 얻은 참 보상의 최댓값(시드별로 구해 평균)
    budgets = {}
    for K in KL_BUDGETS:
        row = {}
        row["rlhf_pg_linear_all_beta_all_steps"] = mean_sd(
            [best_within([(c["kl"], c["r_true"]) for b in PG_BETAS for c in pg[str(b)][s]["curve"]], K) for s in SEEDS])
        row["tilt_linear_rm"] = mean_sd([best_within([(v["kl"], v["r_true"]) for v in tilt_lin[s].values()] + [(0.0, float(tab["r_true"].mean()))], K) for s in SEEDS])
        for l2 in EDGE_L2:
            row[f"tilt_edge_rm_l2={l2}"] = mean_sd([best_within([(v["kl"], v["r_true"]) for v in edge_tilt[str(l2)][s].values()] + [(0.0, float(tab["r_true"].mean()))], K) for s in SEEDS])
        row["dpo_full_all_beta_all_steps"] = mean_sd(
            [best_within([(c["kl"], c["r_true"]) for b in DPO_BETAS for c in dpo[str(b)][s]["curve"]], K) for s in SEEDS])
        row["tilt_true_reward_oracle"] = r(best_within([(v["kl"], v["r_true"]) for v in tilt_true.values()], K))
        budgets[str(K)] = row
    res["E4_dpo"] = {
        "full_summary": dsum,
        "shared_vs_rlhf_opt": {b: {"kl_dpo_to_rlhf_opt": mean_sd([x["kl_dpo_to_rlhf_opt"] for x in v]),
                                   "kl_max": r(max(x["kl_dpo_to_rlhf_opt"] for x in v), 8),
                                   "dpo_r_true": mean_sd([x["dpo_r_true"] for x in v]),
                                   "opt_r_true": mean_sd([x["opt_r_true"] for x in v]),
                                   "dpo_kl": mean_sd([x["dpo_kl"] for x in v]),
                                   "grad_norm_end_max": float(max(x["grad_norm_end"] for x in v)),
                                   "beta_a_vs_u_max_diff": r(max(x["beta_a_minus_u_shiftfree"] for x in v), 8)}
                               for b, v in shared.items()},
        "edge_rm_tilt_mean": edge_mean,
        "best_r_true_within_kl_budget": budgets,
    }
    print(f"E4 done {time.time() - t_all:.1f}s")

    # ---------- E5 계산량 ----------
    Yw, Yl = pairs_for(0)
    u = m.bt_reward_model_fit(Yw, Yl)
    rng = np.random.default_rng(0)
    theta = m.zeros_theta()

    def pg_step(theta):
        Y = m.sample(theta, PG_BATCH, rng)
        R = m.counts(Y) @ u - 0.1 * (m.logprob(theta, Y) - m.logprob_ref(Y))
        b = (R.sum() - R) / (PG_BATCH - 1)
        return m.grad_logprob(theta, Y, (R - b) / PG_BATCH)

    def timeit(fn, n=200):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1000)

    t_rm = timeit(lambda: m.bt_reward_model_fit(Yw, Yl), 50)
    t_pg = timeit(lambda: pg_step(theta))
    t_dpo = timeit(lambda: m.dpo_grad_theta(theta, Yw, Yl, 0.1))
    res["E5_compute"] = {
        "rlhf_rm_fit_ms": r(t_rm, 3),
        "rlhf_pg_step_ms": r(t_pg, 3),
        "dpo_step_ms": r(t_dpo, 3),
        "rlhf_token_logprobs_per_step": PG_BATCH * m.L * 2,
        "rlhf_note": "표본 64개를 한 단어씩 뽑으며 8 x 64번 + 그 표본의 log pi 8 x 64번(기준 정책은 균등이라 상수)",
        "dpo_token_logprobs_per_step": 2 * N_PAIRS * m.L,
        "dpo_note": "선호 쌍마다 log pi(yw), log pi(yl) 두 개를 단어 8개씩. 기준 정책 로그확률 두 개는 한 번 계산해 두면 끝",
        "rlhf_total_steps": PG_STEPS, "dpo_total_steps": DPO_STEPS,
        "rlhf_total_ms_est": r(t_rm + PG_STEPS * t_pg, 1),
        "dpo_total_ms_est": r(DPO_STEPS * t_dpo, 1),
        "rlhf_samples_generated": PG_STEPS * PG_BATCH,
        "dpo_samples_generated": 0,
    }

    # ---------- E6 RLVR ----------
    rv = {}
    for seed in SEEDS:
        th_v, curve = m.rlvr_pg(seed, steps=200, lr=0.05)
        ls = m.log_softmax(th_v).reshape(-1)[tab["flat"]]
        pv = np.exp(ls.sum(axis=1))
        kl_pos = [float(pv @ (ls[:, t] + math.log(m.V))) for t in range(m.L)]
        rv[seed] = {"final_success": r(curve[-1]["success"]), "final_kl": r(curve[-1]["kl"]),
                    "kl_answer_position": r(kl_pos[1]), "kl_other_positions": r(sum(kl_pos) - kl_pos[1]),
                    "final_r_true": r(curve[-1]["r_true"]),
                    "step_success_over_0.9": next((c["step"] for c in curve if c["success"] > 0.9), None),
                    "curve": [{k: r(v) if isinstance(v, float) else v for k, v in c.items()} for c in curve]}
    res["E6_rlvr"] = {
        "base_success": r(1 / 6),
        "final_success": mean_sd([rv[s]["final_success"] for s in SEEDS]),
        "final_kl": mean_sd([rv[s]["final_kl"] for s in SEEDS]),
        "final_r_true": mean_sd([rv[s]["final_r_true"] for s in SEEDS]),
        "step_success_over_0.9": mean_sd([rv[s]["step_success_over_0.9"] for s in SEEDS]),
        "min_kl_for_success1": r(math.log(6), 4),
        "kl_answer_position": mean_sd([rv[s]["kl_answer_position"] for s in SEEDS]),
        "kl_other_positions": mean_sd([rv[s]["kl_other_positions"] for s in SEEDS]),
        "per_seed": {s: {k: v for k, v in x.items() if k != "curve"} for s, x in rv.items()},
    }

    # ---------- E7 NumPy vs PyTorch ----------
    Yw, Yl = pairs_for(3)
    rng = np.random.default_rng(5)
    th = rng.normal(0, 0.7, (m.L, m.V + 1, m.V))
    out = {}
    for beta in (0.01, 0.1, 1.0):
        loss_np, g_np = m.dpo_grad_theta(th, Yw, Yl, beta)
        tt = torch.tensor(th, requires_grad=True)
        loss_t = mt.dpo_loss_theta(tt, Yw, Yl, beta)
        loss_t.backward()
        out[f"dpo_beta{beta}"] = {"loss_rel": float(abs(loss_np - loss_t.item()) / abs(loss_t.item())),
                                  "grad_rel": float(np.abs(g_np - tt.grad.numpy()).max() / np.abs(tt.grad.numpy()).max())}
    uu = rng.normal(0, 0.5, m.V)
    lb, gb = m.bt_loss(uu, Yw, Yl)
    ut = torch.tensor(uu, requires_grad=True)
    lt = mt.bt_loss(ut, Yw, Yl)
    lt.backward()
    out["bt"] = {"loss_rel": float(abs(lb - lt.item()) / abs(lt.item())),
                 "grad_rel": float(np.abs(gb - ut.grad.numpy()).max() / np.abs(ut.grad.numpy()).max())}
    Ys = m.sample(th, 64, rng)
    wts = rng.normal(0, 1, 64)
    gl = m.grad_logprob(th, Ys, wts)
    tt = torch.tensor(th, requires_grad=True)
    (mt.logprob(tt, Ys) @ torch.tensor(wts)).backward()
    out["grad_logprob"] = {"grad_rel": float(np.abs(gl - tt.grad.numpy()).max() / np.abs(tt.grad.numpy()).max())}
    th_np, _ = m.train_dpo(Yw, Yl, 0.1, steps=50, lr=0.05, eval_every=10**9)
    th_t = mt.train_dpo(Yw, Yl, 0.1, steps=50, lr=0.05)
    out["dpo_train_50_steps"] = {"param_max_abs_diff": float(np.abs(th_np - th_t).max()),
                                 "kl_between_policies": float(m.kl_between(th_np, th_t))}
    res["E7_torch"] = out

    # ---------- E8 재현 ----------
    Yw, Yl = pairs_for(0)
    u = m.bt_reward_model_fit(Yw, Yl)
    _, c1 = m.kl_regularized_pg(u, 0.0, 0, steps=100, lr=PG_LR, eval_every=50)
    _, c2 = m.kl_regularized_pg(u, 0.0, 0, steps=100, lr=PG_LR, eval_every=50)
    res["E8_repeat"] = {"identical": all(a["r_true"] == b["r_true"] and a["kl"] == b["kl"] for a, b in zip(c1, c2)),
                        "r_true_step100": r(c1[-1]["r_true"], 10)}

    res["setup"] = {
        "vocab": m.V, "length": m.L, "space_size": int(m.V**m.L), "W": m.W.tolist(), "penalty": m.PENALTY,
        "n_pairs": N_PAIRS, "seeds": SEEDS, "pg": {"steps": PG_STEPS, "lr": PG_LR, "batch": PG_BATCH, "eval_every": PG_EVAL,
                                                  "betas": PG_BETAS, "optimizer": "Adam(0.9, 0.999, 1e-8)"},
        "dpo": {"steps": DPO_STEPS, "lr": DPO_LR, "eval_every": DPO_EVAL, "betas": DPO_BETAS, "batch": "전체 500쌍"},
        "rm_linear_l2": 1e-3, "rm_edge_l2": EDGE_L2, "tilt_betas": TILT_BETAS,
        "policy_params": m.L * (m.V + 1) * m.V,
    }
    res["env"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                  "platform": platform.platform(), "threads": 2, "seconds_total": r(time.time() - t_all, 1)}

    samp = {}
    for x in samples_log:
        if x["done"] and x["info"]["config"] == "rlhf_pg_beta0.0":
            samp.setdefault(str(x["info"]["step"]), []).append(
                {"text": x["info"]["text"], "r_true": x["info"]["r_true"], "r_hat": x["r"]})
    res["E2_samples_seed0_beta0"] = samp
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    LOG_SAMPLES.parent.mkdir(exist_ok=True)
    LOG_SAMPLES.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in samples_log))
    LOG_CURVES.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in curves_log))
    print(f"done {time.time() - t_all:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
