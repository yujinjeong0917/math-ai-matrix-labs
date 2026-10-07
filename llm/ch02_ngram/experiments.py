"""2장 비교 실험. `uv run python llm/ch02_ngram/experiments.py` 로 실행한다.

E1 희소성: n = 1..6에서 처음 보는 문맥 비율, 확률 0 문장 비율, 표의 칸 수 (MLE)
E2 평활 비교: n × {MLE, 덧셈(δ=1), 덧셈(δ 조정), 보간(λ 조정), 보간형 KN(D=0.75)}의 테스트 PPL
   δ와 λ는 검증 세트에서 고르고, 숫자는 테스트 세트에서 잰다.
E3 장거리 일치: 주어와 동사 사이 부사 k개일 때 KN n-gram의 수 일치 정확도.
   학습 문장 2만 개와 20만 개(10배)를 비교한다.
E4 계산량: 카운트 시간과 실제로 채워진 칸 수 vs 꽉 찬 표의 칸 수
E5 PyTorch 대조: cross_entropy로 다시 잰 PPL과 NumPy PPL의 차이
E0 바닥값: 문법의 실제 확률로 잰 테스트 PPL
E6 Kneser-Ney 직관: 'the'의 유니그램 확률 vs continuation 확률

데이터는 1장 합성 문법(부사 k = 0..6)이다. 시드 3개는 말뭉치 생성 시드다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import ngram_np as ng
import ngram_torch as nt

OUT = Path(__file__).parent / "results" / "ch02.json"
KS = range(7)
NS = (1, 2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
N_TRAIN, N_VALID, N_TEST, N_BIG = 20_000, 2_000, 5_000, 200_000
DELTAS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)
LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
D_KN = 0.75

WORDS, STOI = ng.grammar_vocab()
V, BOS = len(WORDS), STOI[ng.BOS]


def data(seed, n_train=N_TRAIN):
    # 검증·테스트는 학습 크기와 무관한 별도 생성기에서 뽑는다: 2만/20만 비교가 같은 테스트 문장을 쓴다
    train = ng.make_corpus(n_train, KS, np.random.default_rng(seed))
    rng_eval = np.random.default_rng(seed + 1000)
    valid = ng.make_corpus(N_VALID, KS, rng_eval)
    test = ng.make_corpus(N_TEST, KS, rng_eval)
    return train, valid, test


def fit(train_ids, N):
    counts = ng.count_ngrams(train_ids, N, BOS)
    return counts, ng.continuation_counts(counts, N)


def oracle_ppl(corpus):
    """문법을 만든 실제 확률로 잰 PPL. 어떤 모델도 기대값으로는 이보다 낮을 수 없는 바닥이다.

    문장 확률 = P(the)=1 · P(주어)=1/16 · Π_j [P(부사를 하나 더 | 지금까지 j개) · 1/12]
              · P(여기서 멈춤 | j=k) · P(동사)=1/8 · P(</s>)=1.  k는 0..6에서 고르게 뽑는다.
    """
    K = len(KS)
    logs, T = 0.0, 0
    for toks, vpos, _, _ in corpus:
        k = vpos - 2
        lp = np.log(1 / (2 * len(ng.NOUNS)))
        for j in range(k):
            lp += np.log((K - 1 - j) / (K - j)) + np.log(1 / len(ng.ADVERBS))
        lp += np.log(1 / (K - k)) + np.log(1 / len(ng.VERBS))
        logs += lp
        T += k + 4
    return float(np.exp(-logs / T))


def _ppl_or_none(x):
    return None if not np.isfinite(x) else float(x)


def e1_e2(seed):
    train, valid, test = data(seed)
    tr, va, te = (ng.to_ids(c, STOI) for c in (train, valid, test))
    counts, cont = fit(tr, max(NS))
    rows = []
    for n in NS:
        st = ng.sparsity_stats(counts, n, te, BOS)
        mk = lambda method, **kw: ng.NGramModel(counts, n, V, method=method, cont=cont, **kw)
        d_best = min(DELTAS, key=lambda d: ng.perplexity(mk("add", delta=d), va, BOS))
        l_best = min(LAMBDAS, key=lambda l: ng.perplexity(mk("interp", lam=l), va, BOS))
        rows.append({
            "n": n,
            "dense_cells_V_pow_n": V**n,
            "seen_contexts": len(counts[n]),
            "nonzero_cells": ng.nonzero_cells(counts, n),
            **st,
            "ppl": {
                "mle": _ppl_or_none(ng.perplexity(mk("mle"), te, BOS)),
                "add1": ng.perplexity(mk("add", delta=1.0), te, BOS),
                "add_tuned": ng.perplexity(mk("add", delta=d_best), te, BOS),
                "interp_tuned": ng.perplexity(mk("interp", lam=l_best), te, BOS),
                "kn": ng.perplexity(mk("kn", D=D_KN), te, BOS),
            },
            "delta_best": d_best,
            "lambda_best": l_best,
        })
    return {"train_tokens": sum(len(s) for s in tr), "test_tokens": rows[0]["tokens"], "rows": rows,
            "oracle_test_ppl": oracle_ppl(test)}


def e3_agreement(seed, n_train):
    train, _, test = data(seed, n_train)
    tr = ng.to_ids(train, STOI)
    te_ids = ng.to_ids(test, STOI)
    counts, cont = fit(tr, max(NS))
    out = {}
    for n in NS[1:]:
        m = ng.NGramModel(counts, n, V, method="kn", D=D_KN, cont=cont)
        out[str(n)] = {
            "agreement_by_k": {str(k): v for k, v in ng.agreement_accuracy(m, test, STOI, BOS).items()},
            "unseen_context_rate": ng.sparsity_stats(counts, n, te_ids, BOS)["unseen_context_rate"],
            "kn_ppl": ng.perplexity(m, te_ids, BOS),
        }
    return out


def e4_compute(seed=0):
    train, _, _ = data(seed)
    tr = ng.to_ids(train, STOI)
    rows = []
    for n in NS:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            c = ng.count_ngrams(tr, n, BOS)
            times.append(time.perf_counter() - t0)
        nz = ng.nonzero_cells(c, n)
        rows.append({"n": n, "count_seconds_median": float(np.median(times)), "nonzero_cells": nz,
                     "dense_cells": V**n, "dense_float64_bytes": 8 * V**n,
                     "fill_ratio": nz / V**n})
    return rows


def e5_torch(seed=0):
    train, _, test = data(seed)
    tr, te = ng.to_ids(train, STOI), ng.to_ids(test, STOI)
    counts, cont = fit(tr, 3)
    out = {}
    for method, kw in (("add", {"delta": 0.01}), ("interp", {"lam": 0.7}), ("kn", {"D": D_KN})):
        m = ng.NGramModel(counts, 3, V, method=method, cont=cont, **kw)
        rows, ys = nt.rows_and_targets(m, ng.positions(te, 3, BOS))
        p_np = ng.perplexity(m, te, BOS)
        p_t = nt.perplexity_torch(rows, ys)
        out[method] = {"numpy": p_np, "torch": p_t, "abs_diff": abs(p_np - p_t),
                       "max_row_sum_error": float(np.abs(rows.sum(1) - 1).max()), "tokens": int(len(ys))}
    return out


def e6_the(seed=0):
    train, _, _ = data(seed)
    counts, cont = fit(ng.to_ids(train, STOI), 2)
    uni, con = counts[1][()], cont[1][()]
    the = STOI["the"]
    return {
        "the_count": uni[the], "unigram_ml_the": uni[the] / sum(uni.values()),
        "the_distinct_left": con[the], "continuation_the": con[the] / sum(con.values()),
        "often_count": uni[STOI["often"]], "often_distinct_left": con[STOI["often"]],
        "continuation_often": con[STOI["often"]] / sum(con.values()),
        "unigram_ml_often": uni[STOI["often"]] / sum(uni.values()),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    t_all = time.perf_counter()
    res = {"env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "machine": platform.machine()},
           "setup": {"V": V, "ks": list(KS), "n_train": N_TRAIN, "n_valid": N_VALID, "n_test": N_TEST,
                     "n_big": N_BIG, "seeds": list(SEEDS), "deltas": list(DELTAS), "lambdas": list(LAMBDAS),
                     "kn_discount": D_KN}}

    per_seed = {str(s): e1_e2(s) for s in SEEDS}
    res["E1_E2_per_seed"] = per_seed
    summary = []
    for i, n in enumerate(NS):
        rs = [per_seed[str(s)]["rows"][i] for s in SEEDS]
        summary.append({
            "n": n,
            "dense_cells_V_pow_n": rs[0]["dense_cells_V_pow_n"],
            "nonzero_cells_mean": _mean([r["nonzero_cells"] for r in rs]),
            "seen_contexts_mean": _mean([r["seen_contexts"] for r in rs]),
            "unseen_context_rate_mean": _mean([r["unseen_context_rate"] for r in rs]),
            "unseen_ngram_rate_mean": _mean([r["unseen_ngram_rate"] for r in rs]),
            "zero_prob_sentence_rate_mean": _mean([r["zero_prob_sentence_rate"] for r in rs]),
            "mle_ppl_finite_seeds": sum(r["ppl"]["mle"] is not None for r in rs),
            "ppl_mean": {k: _mean([r["ppl"][k] for r in rs]) for k in ("mle", "add1", "add_tuned", "interp_tuned", "kn")},
            "ppl_min_max": {k: [min(r["ppl"][k] for r in rs), max(r["ppl"][k] for r in rs)]
                            for k in ("add1", "add_tuned", "interp_tuned", "kn")},
            "delta_best": [r["delta_best"] for r in rs],
            "lambda_best": [r["lambda_best"] for r in rs],
        })
    res["E1_E2_summary"] = summary
    res["train_tokens_mean"] = _mean([per_seed[str(s)]["train_tokens"] for s in SEEDS])
    res["oracle_test_ppl_mean"] = _mean([per_seed[str(s)]["oracle_test_ppl"] for s in SEEDS])
    res["test_tokens_mean"] = _mean([per_seed[str(s)]["test_tokens"] for s in SEEDS])

    e3 = {"20000": {str(s): e3_agreement(s, N_TRAIN) for s in SEEDS},
          "200000": {str(s): e3_agreement(s, N_BIG) for s in SEEDS}}
    e3_sum = {}
    for size, by_seed in e3.items():
        e3_sum[size] = {}
        for n in NS[1:]:
            ks = by_seed["0"][str(n)]["agreement_by_k"].keys()
            e3_sum[size][str(n)] = {
                "agreement_by_k_mean": {k: _mean([by_seed[str(s)][str(n)]["agreement_by_k"][k] for s in SEEDS]) for k in ks},
                "unseen_context_rate_mean": _mean([by_seed[str(s)][str(n)]["unseen_context_rate"] for s in SEEDS]),
                "kn_ppl_mean": _mean([by_seed[str(s)][str(n)]["kn_ppl"] for s in SEEDS]),
            }
    res["E3_per_seed"] = e3
    res["E3_summary"] = e3_sum
    res["E4"] = e4_compute()
    res["E5"] = e5_torch()
    res["E6"] = e6_the()
    res["total_seconds"] = time.perf_counter() - t_all

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({"E1_E2_summary": summary, "E3_summary": e3_sum, "E4": res["E4"], "E5": res["E5"],
                      "E6": res["E6"], "total_seconds": res["total_seconds"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
