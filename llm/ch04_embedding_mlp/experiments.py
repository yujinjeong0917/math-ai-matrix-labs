"""4장 비교 실험. `uv run python llm/ch04_embedding_mlp/experiments.py` 로 실행한다.

E1 실패 재현: 보간 트라이그램이 학습에 없는 "강아지가 잔다"에 주는 확률
E2 MLP 비교: 같은 말뭉치에서 MLP(공유 임베딩)를 시드 5개 x weight decay 4개로 학습
E2b 학습이 길어지면: 시드 0에서 스텝에 따른 확률 비의 변화
E3 파라미터 수: 문맥 길이 k에 따른 확률표 칸 수 vs MLP 파라미터 수
E4 학습 안정성: 학습률 3개, 시드 5개, 손실 곡선과 발산 여부
E5 계산량: 스텝당 FLOP과 실측 시간, n-gram의 세기·조회 시간
E0 구현 검증: 손 역전파 vs 중앙 차분, NumPy vs PyTorch autograd (테스트와 같은 값을 기록용으로)

결과는 results/ch04.json 에 저장하고, 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import ch04_grammar as G
import mlp_lm_np as M
import mlp_lm_torch as T
from ch04_ngram import InterpolatedTrigram

OUT = Path(__file__).parent / "results" / "ch04.json"
V = len(G.VOCAB)
I = G.IDX
K = G.CONTEXT
M_DIM, H_DIM, LR, STEPS, BATCH = 8, 32, 0.3, 6000, 32
SEEDS = range(5)
WDS = (0.0, 1e-4, 1e-3, 3e-3)


def data():
    S = G.make_splits(0)
    return {k: G.to_examples(v) for k, v in S.items()}, S


def probe_ctx():
    """비교할 세 확률의 문맥: (오늘, 강아지가), (오늘, 고양이가)."""
    return np.array([[I["오늘"], I["강아지가"]], [I["오늘"], I["고양이가"]]])


def ratios(p_dog, p_cat):
    """p_dog, p_cat: 다음 단어 분포 (|V|,)."""
    return {
        "P(잔다|오늘 강아지가)": float(p_dog[I["잔다"]]),
        "P(잔다|오늘 고양이가)": float(p_cat[I["잔다"]]),
        "P(읽는다|오늘 강아지가)": float(p_dog[I["읽는다"]]),
        "ratio_unseen_over_seen": float(p_dog[I["잔다"]] / p_cat[I["잔다"]]),
        "ratio_plausible_over_impossible": float(p_dog[I["잔다"]] / p_dog[I["읽는다"]]),
    }


def e1_ngram(D):
    ng = InterpolatedTrigram(V).fit(*D["train"])
    lam = ng.tune(*D["valid"])
    dist = lambda s: np.array([ng.prob(I["오늘"], I[s], w) for w in range(V)])
    out = {
        "lambdas_tuned_on_valid": [float(x) for x in lam],
        "ppl_test_seen": float(np.exp(ng.nll(*D["test_seen"]))),
        "ppl_test_unseen": float(np.exp(ng.nll(*D["test_unseen"]))),
        "probes": ratios(dist("강아지가"), dist("고양이가")),
        "hand_calc": {
            "N_train_predictions": ng.N,
            "V": V,
            "count(잔다)": ng.c1[I["잔다"]],
            "count(읽는다)": ng.c1[I["읽는다"]],
            "count(고양이가 잔다)": ng.c2[(I["고양이가"], I["잔다"])],
            "count(고양이가 *)": ng.c1ctx[I["고양이가"]],
            "count(오늘 고양이가 잔다)": ng.c3[(I["오늘"], I["고양이가"], I["잔다"])],
            "count(오늘 고양이가 *)": ng.c2ctx[(I["오늘"], I["고양이가"])],
            "count(강아지가 잔다)": ng.c2[(I["강아지가"], I["잔다"])],
        },
        "stored_entries": ng.stored_entries(),
        "table_size": ng.table_size(3),
    }
    # 공정성 점검: 일부러 테스트(처음 보는 쌍) 세트에 맞춰 l을 고르면 n-gram은 어디까지 가나
    ng.tune(*D["test_unseen"])
    out["oracle_lambdas_tuned_on_test_unseen"] = [float(x) for x in ng.lambdas]
    out["oracle_ppl_test_unseen"] = float(np.exp(ng.nll(*D["test_unseen"])))
    out["oracle_ppl_test_seen"] = float(np.exp(ng.nll(*D["test_seen"])))
    out["oracle_probes"] = ratios(dist("강아지가"), dist("고양이가"))
    return out


def train_one(D, seed, wd, lr=LR, steps=STEPS, log_every=0):
    rng = np.random.default_rng(seed)
    p = M.init_params(V, M_DIM, K, H_DIM, rng)
    log = M.train(p, *D["train"], lr=lr, steps=steps, batch=BATCH, rng=rng, wd=wd, log_every=log_every, valid=D["valid"])
    return p, log


def evaluate(p, D):
    P, _ = M.forward(p, probe_ctx())
    C = p["C"]
    return {
        "valid_nll": M.loss(p, *D["valid"]),
        "ppl_test_seen": float(np.exp(M.loss(p, *D["test_seen"]))),
        "ppl_test_unseen": float(np.exp(M.loss(p, *D["test_unseen"]))),
        **ratios(P[0], P[1]),
        "cos(고양이가,강아지가)": M.cosine(C[I["고양이가"]], C[I["강아지가"]]),
        "cos(고양이가,학생이)": M.cosine(C[I["고양이가"]], C[I["학생이"]]),
        "cos(잔다,뛴다)": M.cosine(C[I["잔다"]], C[I["뛴다"]]),
        "cos(잔다,읽는다)": M.cosine(C[I["잔다"]], C[I["읽는다"]]),
    }


def summarize(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "min": float(np.min([r[k] for r in rows])),
                "max": float(np.max([r[k] for r in rows])), "per_seed": [float(r[k]) for r in rows]} for k in keys}


def e2_mlp(D):
    out = {}
    for wd in WDS:
        out[f"wd={wd:g}"] = summarize([evaluate(train_one(D, s, wd)[0], D) for s in SEEDS])
    chosen = min(out, key=lambda k: out[k]["valid_nll"]["mean"])
    return {"config": {"m": M_DIM, "h": H_DIM, "k": K, "lr": LR, "steps": STEPS, "batch": BATCH, "direct": True},
            "by_weight_decay": out, "chosen_by_valid_nll": chosen}


def e2b_trajectory(D):
    out = {}
    for wd in (0.0, 3e-3):
        rng = np.random.default_rng(0)
        p = M.init_params(V, M_DIM, K, H_DIM, rng)
        rows = []
        for step in range(0, 12001, 1000):
            if step:
                M.train(p, *D["train"], lr=LR, steps=1000, batch=BATCH, rng=rng, wd=wd)
            r = evaluate(p, D)
            rows.append({"step": step, "ratio_unseen_over_seen": r["ratio_unseen_over_seen"],
                         "ratio_plausible_over_impossible": r["ratio_plausible_over_impossible"],
                         "ppl_test_unseen": r["ppl_test_unseen"], "ppl_test_seen": r["ppl_test_seen"]})
        out[f"wd={wd:g}"] = rows
    return out


def e3_params():
    rows = []
    for Vx, m, h in ((V, M_DIM, H_DIM), (17000, 60, 100)):
        for k in range(1, 7):
            rows.append({"V": Vx, "m": m, "h": h, "k": k, "n": k + 1,
                         "prob_table_cells": sum(Vx**j for j in range(1, k + 2)),
                         "mlp_params": M.n_params(Vx, m, k, h)})
    return rows


def e4_lr(D):
    out = {}
    for lr in (0.03, 0.3, 3.0):
        runs = []
        for s in SEEDS:
            p, log = train_one(D, s, 0.0, lr=lr, steps=STEPS, log_every=500)
            final = M.loss(p, *D["valid"]) if all(np.all(np.isfinite(v)) for v in p.values()) else float("inf")
            diverged = any(not np.isfinite(r[1]) for r in log) or not np.isfinite(final)
            first_bad = next((r[0] for r in log if not (np.isfinite(r[1]) and np.isfinite(r[2]))), None)
            runs.append({"diverged": diverged, "first_nonfinite_logged_step": first_bad,
                         "valid_curve": [[r[0], r[2] if np.isfinite(r[2]) else None] for r in log],
                         "final_valid_nll": final if np.isfinite(final) else None})
        out[f"lr={lr:g}"] = {"diverged_seeds": sum(r["diverged"] for r in runs), "runs": runs}
    return out


def _median_time(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def e5_compute(D):
    rng = np.random.default_rng(0)
    p = M.init_params(V, M_DIM, K, H_DIM, rng)
    ctx, tgt = D["train"]
    idx = rng.integers(0, len(tgt), size=BATCH)

    def step():
        P, cache = M.forward(p, ctx[idx])
        M.sgd_step(p, M.backward(p, cache, tgt[idx]), 0.0)

    def mlp_epoch():
        for _ in range(len(tgt) // BATCH):
            step()

    t_step = _median_time(lambda: [step() for _ in range(100)]) / 100
    ng_fit = _median_time(lambda: InterpolatedTrigram(V).fit(ctx, tgt))
    ng = InterpolatedTrigram(V).fit(ctx, tgt)
    tc, tt = D["test_seen"]
    ng_query = _median_time(lambda: ng.probs(tc, tt)) / len(tt)
    return {
        "mlp_step_flops": M.step_flops(V, M_DIM, K, H_DIM, BATCH),
        "mlp_forward_flops_per_example": M.step_flops(V, M_DIM, K, H_DIM, 1) // 3,
        "mlp_step_ms": t_step * 1e3,
        "mlp_total_train_s_estimate": t_step * STEPS,
        "train_predictions": int(len(tgt)),
        "epochs_seen": STEPS * BATCH / len(tgt),
        "ngram_fit_ms": ng_fit * 1e3,
        "ngram_query_us_per_prediction": ng_query * 1e6,
        "note": "FLOP은 행렬곱만 센다(tanh, softmax, 임베딩 조회 제외). 역전파는 순전파의 2배로 셈.",
    }


def e0_checks():
    rng = np.random.default_rng(0)
    p = M.init_params(V, 3, K, 5, rng)
    ctx, tgt = rng.integers(0, V, size=(12, K)), rng.integers(0, V, size=12)
    _, cache = M.forward(p, ctx)
    g = M.backward(p, cache, tgt)
    worst_rel = 0.0
    for name in p:
        num = np.zeros_like(p[name])
        it = np.nditer(p[name], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[name][i]
            p[name][i] = old + 1e-6
            lp = M.loss(p, ctx, tgt)
            p[name][i] = old - 1e-6
            lm = M.loss(p, ctx, tgt)
            p[name][i] = old
            num[i] = (lp - lm) / 2e-6
        worst_rel = max(worst_rel, float(np.linalg.norm(num - g[name]) / (np.linalg.norm(num) + np.linalg.norm(g[name]))))
    model = T.NPLM.from_numpy(p)
    T.loss(model, torch.from_numpy(ctx), torch.from_numpy(tgt)).backward()
    gt = model.grads_as_numpy()
    worst_abs = max(float(np.max(np.abs(g[k] - gt[k]))) for k in g)
    return {"finite_difference_worst_rel_err": worst_rel, "torch_autograd_worst_abs_diff": worst_abs}


def main():
    torch.set_num_threads(1)
    D, S = data()
    results = {
        "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "machine": platform.machine(), "torch_threads": 1},
        "corpus": {"V": V, "vocab": G.VOCAB, "held_out_pairs": G.HELD_OUT,
                   "sentences": {k: len(v) for k, v in S.items()},
                   "predictions": {k: int(len(v[1])) for k, v in D.items()},
                   "allowed_pairs_in_train": len(G.allowed_pairs()),
                   "true_ppl_seen_distribution": G.true_ppl()},
        "E0": e0_checks(),
        "E1": e1_ngram(D),
        "E2": e2_mlp(D),
        "E2b": e2b_trajectory(D),
        "E3": e3_params(),
        "E4": e4_lr(D),
        "E5": e5_compute(D),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in ("E0", "E1", "E5")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
