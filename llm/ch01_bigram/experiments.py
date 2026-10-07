"""1장 비교 실험. `uv run python llm/ch01_bigram/experiments.py` 로 실행한다.

E0 손 계산 예: 문장 4개짜리 말뭉치의 카운트 표와 확률
E1 수 일치: 주어와 동사 사이 부사 k개(k = 0, 1, 3, 5, 8), 유니그램 / 바이그램 / (예고) 트라이그램
E2 테스트 PPL: 유니그램 vs 바이그램, NumPy와 PyTorch 계산 대조
E3 학습으로 얻은 표: nn.Embedding(V, V) + 교차엔트로피가 카운트 MLE와 같아지는지
E4 생성: 바이그램으로 문장을 뽑아 수 일치 오류율과 부사 개수 분포를 본다
E5 계산량: 카운트 시간 O(T), 표 메모리 V^2

시드 3개(0, 1, 2)는 데이터 생성 시드다. 결과는 results/ch01.json 에 저장하고,
챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import bigram_np as bnp
from bigram_torch import fit_bigram, perplexity_torch

OUT = Path(__file__).parent / "results" / "ch01.json"
SEEDS = (0, 1, 2)
N_TRAIN, N_TEST = 20_000, 2_000
TRAIN_KS = range(9)  # 학습 문장의 부사 개수는 0..8에서 고르게
TEST_KS = (0, 1, 3, 5, 8)
STOI, ITOS = bnp.grammar_vocab()
V = len(ITOS)


def ids_of(corpus):
    return [np.array([STOI[t] for t in toks]) for toks, *_ in corpus]


def data(seed):
    rng = np.random.default_rng(seed)
    train = bnp.make_corpus(N_TRAIN, TRAIN_KS, rng)
    tests = {k: bnp.make_corpus(N_TEST, [k], rng) for k in TEST_KS}
    mixed = bnp.make_corpus(N_TEST, TRAIN_KS, rng)
    return train, tests, mixed


def e0_hand():
    """첫 화면의 손 계산 예. 문장 4개를 직접 세면 같은 값이 나와야 한다."""
    sents = [["dog", "runs"], ["dogs", "run"], ["dog", "often", "runs"], ["dogs", "often", "run"]]
    stoi, itos = bnp.build_vocab([t for s in sents for t in s])
    counts = bnp.count_sentences([[stoi[t] for t in s] for s in sents], len(itos))
    P = bnp.bigram_mle(counts)
    rows = {}
    for a in ("dog", "dogs", "often"):
        rows[a] = {
            b: {"count": int(counts[stoi[a], stoi[b]]), "prob": float(P[stoi[a], stoi[b]])}
            for b in ("runs", "run", "often")
        }
    return {"sentences": [" ".join(s) for s in sents], "rows": rows}


def trigram_fn(train):
    c3, c2 = Counter(), Counter()
    for toks, *_ in train:
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            c3[(a, b, c)] += 1
            c2[(a, b)] += 1

    def f(ctx, w):
        a, b = ctx[-2], ctx[-1]
        return c3[(a, b, w)] / c2[(a, b)] if c2[(a, b)] else 0.0

    return f


def e1_agreement():
    out = {m: {str(k): [] for k in TEST_KS} for m in ("unigram", "bigram", "trigram")}
    bits = {m: {str(k): [] for k in TEST_KS} for m in ("unigram", "bigram")}  # 트라이그램은 못 본 문맥에서 무한대
    ties = {m: {str(k): [] for k in TEST_KS} for m in ("unigram", "bigram", "trigram")}
    no_tie = {m: {str(k): [] for k in TEST_KS} for m in ("unigram", "bigram", "trigram")}
    for seed in SEEDS:
        train, tests, _ = data(seed)
        tr_ids = ids_of(train)
        P = bnp.bigram_mle(bnp.count_sentences(tr_ids, V))
        p1 = bnp.unigram_mle(tr_ids, V)
        fns = {
            "unigram": lambda ctx, w: p1[STOI[w]],
            "bigram": lambda ctx, w: P[STOI[ctx[-1]], STOI[w]],
            "trigram": trigram_fn(train),
        }
        for k in TEST_KS:
            for m, f in fns.items():
                s = bnp.agreement_scores(f, tests[k], STOI)
                out[m][str(k)].append(s["acc"])
                if m in bits:
                    bits[m][str(k)].append(s["bits"])
                ties[m][str(k)].append(s["ties"])
                if s["ties"] < 1:  # 동률을 뺀 문장만의 정확도: (acc - 0.5*ties) / (1 - ties)
                    no_tie[m][str(k)].append((s["acc"] - 0.5 * s["ties"]) / (1 - s["ties"]))
    def summ(table):
        return {m: {k: {"by_seed": v, "mean": float(np.mean(v))} for k, v in d.items()} for m, d in table.items()}

    return {"accuracy": summ(out), "verb_surprise_bits": summ(bits), "tie_rate": summ(ties),
            "accuracy_excluding_ties": summ(no_tie), "chance": 0.5}


def e2_perplexity():
    rows = []
    for seed in SEEDS:
        train, _, mixed = data(seed)
        tr_ids, te_ids = ids_of(train), ids_of(mixed)
        P = bnp.bigram_mle(bnp.count_sentences(tr_ids, V))
        p1 = bnp.unigram_mle(tr_ids, V)
        zero = sum(int((P[s[:-1], s[1:]] == 0).sum()) for s in te_ids)
        rows.append({
            "seed": seed,
            "unigram_ppl": bnp.unigram_perplexity(p1, te_ids),
            "bigram_ppl": bnp.perplexity(P, te_ids),
            "bigram_ppl_torch": perplexity_torch(P, te_ids),
            "test_tokens": int(sum(len(s) - 1 for s in te_ids)),
            "zero_prob_tokens": zero,
        })
    return {
        "by_seed": rows,
        "unigram_mean": float(np.mean([r["unigram_ppl"] for r in rows])),
        "bigram_mean": float(np.mean([r["bigram_ppl"] for r in rows])),
        "max_abs_diff_numpy_torch": float(max(abs(r["bigram_ppl"] - r["bigram_ppl_torch"]) for r in rows)),
    }


def e3_torch_fit():
    train, _, _ = data(0)
    counts = bnp.count_sentences(ids_of(train), V)
    P = bnp.bigram_mle(counts)
    t0 = time.perf_counter()
    model, losses = fit_bigram(counts, steps=3000, lr=0.1)
    secs = time.perf_counter() - t0
    T = model.table().detach().numpy()
    seen = counts.sum(1) > 0
    N = counts.sum()
    mle_loss = float(-(counts[P > 0] * np.log(P[P > 0])).sum() / N)
    D = np.where(seen[:, None], np.abs(T - P), 0.0)
    i, j = np.unravel_index(int(D.argmax()), D.shape)
    return {
        "steps": 3000,
        "lr": 0.1,
        "max_abs_diff_seen_rows": float(np.abs(T[seen] - P[seen]).max()),
        # 최대 차이가 어디서 났나: 확률 0인 칸(softmax가 못 내는 값)인지, 덜 수렴한 0 아닌 칸인지
        "max_diff_cell": {"prev": ITOS[i], "next": ITOS[j], "mle_prob": float(P[i, j])},
        "max_abs_diff_zero_cells": float(D[P == 0].max()),
        "max_abs_diff_nonzero_cells": float(D[P > 0].max()),
        "final_loss_nats": losses[-1],
        "mle_loss_nats": mle_loss,
        "loss_at_step": {str(s): losses[s - 1] for s in (1, 10, 100, 1000, 3000)},
        "seconds": secs,
        "rows_never_seen_as_prev": [ITOS[i] for i in np.where(~seen)[0]],
    }


def e4_generate():
    train, _, _ = data(0)
    P = bnp.bigram_mle(bnp.count_sentences(ids_of(train), V))
    rng = np.random.default_rng(123)
    bos, eos = STOI[bnp.BOS], STOI[bnp.EOS]
    toks = bnp.sample(P, bos, 50, rng, eos=eos, bos=bos)
    text = " ".join(ITOS[i] for i in toks)

    # 1000문장을 뽑아 구조별로 센다. 바이그램 표는 구조(the→명사→부사*→동사→</s>)를 강제한다.
    noun_num = {w: i for pair in bnp.NOUNS for i, w in enumerate(pair)}
    verb_num = {w: i for pair in bnp.VERBS for i, w in enumerate(pair)}
    rng = np.random.default_rng(456)
    stats = {"k0": [0, 0], "k_ge1": [0, 0]}  # [오류 수, 문장 수]
    ks, malformed = [], 0
    for _ in range(1000):
        s = [bos]
        while s[-1] != eos and len(s) < 200:
            s.append(int(rng.choice(V, p=P[s[-1]])))
        w = [ITOS[i] for i in s[1:-1]]
        if len(w) < 3 or w[0] != "the" or w[1] not in noun_num or w[-1] not in verb_num:
            malformed += 1
            continue
        k = len(w) - 3
        ks.append(k)
        key = "k0" if k == 0 else "k_ge1"
        stats[key][0] += int(noun_num[w[1]] != verb_num[w[-1]])
        stats[key][1] += 1
    train_ks = [len(t) - 5 for t, *_ in train]
    return {
        "sample_50_tokens": text,
        "agreement_error_rate": {k: {"errors": e, "sentences": n, "rate": e / n} for k, (e, n) in stats.items()},
        "malformed": malformed,
        "generated_k": {"mean": float(np.mean(ks)), "max": int(np.max(ks)), "share_k_gt_8": float(np.mean(np.array(ks) > 8))},
        "train_k": {"mean": float(np.mean(train_ks)), "max": int(np.max(train_ks))},
        # 부사 뒤에 또 부사가 올 확률(부사 12개 행의 평균). 학습 k가 0..8 고르게면 이론값은 약 0.78
        "p_adverb_to_adverb_mean": float(np.mean([sum(P[STOI[b], STOI[a]] for a in bnp.ADVERBS) for b in bnp.ADVERBS])),
    }


def e5_compute():
    train, _, _ = data(0)
    ids = ids_of(train)
    rows = []
    for reps in (1, 4, 16):
        big = ids * reps
        T = int(sum(len(s) for s in big))
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            bnp.count_sentences(big, V)
            times.append(time.perf_counter() - t0)
        rows.append({"tokens": T, "count_ms_median": float(np.median(times) * 1000)})
    counts = bnp.count_sentences(ids, V)
    return {
        "count_time": rows,
        "vocab": V,
        "table_cells": V * V,
        "nonzero_cells": int((counts > 0).sum()),
        "trigram_table_cells": V**3,
    }


def main():
    torch.set_num_threads(1)
    results = {
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "machine": platform.machine(),
            "torch_threads": 1,
        },
        "setup": {"vocab": V, "n_train": N_TRAIN, "n_test_per_k": N_TEST, "train_k": list(TRAIN_KS),
                  "test_k": list(TEST_KS), "seeds": list(SEEDS),
                  "nouns": len(bnp.NOUNS), "verbs": len(bnp.VERBS), "adverbs": len(bnp.ADVERBS)},
        "E0": e0_hand(),
        "E1": e1_agreement(),
        "E2": e2_perplexity(),
        "E3": e3_torch_fit(),
        "E4": e4_generate(),
        "E5": e5_compute(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
