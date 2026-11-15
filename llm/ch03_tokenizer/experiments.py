"""3장 비교 실험. `uv run python llm/ch03_tokenizer/experiments.py` 로 실행한다.

E0 논문 장난감 예제: Sennrich 등 Figure 1 사전에서 'lower'가 'low er</w>'로 나뉘는지,
   Algorithm 1(원문 코드)과 이 구현의 병합 목록 비교
E1 단어 단위의 실패: 어휘 크기 V별 테스트 <unk> 비율, 글자 단위의 시퀀스 길이 배수
E2 BPE 병합 횟수 M별: 어휘 크기, 문장당 토큰 수, <unk> 비율
E3 같은 잣대: 보간 trigram을 각 분할 위에서 학습한 뒤 토큰당 PPL과 글자당 비트(bpc)
E4 계산량: 병합마다 전부 다시 세는 방식 vs 쌍 색인을 갱신하는 방식(같은 병합 목록인지 포함)
E5 처음 보는 글자: ko_like에서 학습에 없던 음절 비율과 UTF-8 바이트 수

데이터 분할 시드 3개(0, 1, 2). 말뭉치 생성 시드는 0으로 고정.
결과는 results/ch03.json 에 저장하고, 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

import bpe_np as B
import ch03_data as D
from ch03_ngram import InterpTrigram, bits_per_char, tune_lambdas
from sennrich_alg1 import algorithm1_verbatim

OUT = Path(__file__).parent / "results" / "ch03.json"
SEEDS = (0, 1, 2)
LANGS = ("en_like", "ko_like")
WORD_V = (1000, 5000, 20000)
MERGES = (0, 500, 2000, 8000)
# 글자 우주 크기 U: <unk>가 가린 글자 하나의 철자 비용 log2(U)에 쓴다. 영어 26, 한글 음절 11,172.
UNIVERSE = {"en_like": 26, "ko_like": D.HANGUL_COUNT}


def e0_toy():
    """논문 장난감 예제(Figure 1, Algorithm 1 예)에서 이 구현과 원문 코드의 병합 목록을 비교한다."""
    fig1 = {"low": 1, "lowest": 1, "newer": 1, "wider": 1}
    merges = B.learn_bpe(fig1, 4)
    tok = B.BPETokenizer(fig1, merges)
    alg1_vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}
    # 논문 원문 코드(동률이면 먼저 센 쌍)와 이 구현(동률이면 사전순)은 동률에서 갈릴 수 있다. 어느 병합에서 갈리는지 아래에서 기록한다.
    ref = algorithm1_verbatim(dict(alg1_vocab), 10)
    toy = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    mine = B.learn_bpe(toy, 10)
    mine_first = B.learn_bpe_naive(toy, 10, tie="first")
    counts = _merge_counts(toy, mine)
    return {
        "figure1_merges_ours": [" ".join(m) for m in merges],
        "figure1_lower": tok.apply_bpe("lower"),
        "algorithm1_merges_paper_code": [" ".join(m) for m in ref],
        "algorithm1_merges_ours": [" ".join(m) for m in mine],
        "algorithm1_same": [tuple(m) for m in ref] == [tuple(m) for m in mine],
        # 두 병합 목록이 처음 달라지는 위치. 완전히 같으면 next()가 StopIteration을 내므로, 이 입력에서 갈린다는 전제의 코드다.
        "algorithm1_first_divergence": next(i for i, (a, b) in enumerate(zip(ref, mine)) if a != b),
        "algorithm1_same_with_first_tie": [tuple(m) for m in ref] == [tuple(m) for m in mine_first],
        "merge_counts_ours": counts,
    }


def _merge_counts(freqs, merges):
    """각 병합 시점에 고른 쌍의 빈도. 동률이 어디서 생기는지 보려고."""
    vocab = {tuple(w) + (B.EOW,): f for w, f in freqs.items()}
    out = []
    for m in merges:
        out.append(sum(f * sum(1 for p in zip(s[:-1], s[1:]) if p == m) for s, f in vocab.items()))
        vocab = {tuple(B._merge_word(list(s), m)): f for s, f in vocab.items()}
    return out


def _fit_and_score(train_seqs, dev_seqs, test_seqs, vocab, n_chars, unk_chars, U):
    """한 토큰화 방식에서 trigram을 학습하고(검증 세트로 람다 선택) 테스트 세트의 bpc와 토큰 PPL을 잰다."""
    lm = InterpTrigram(train_seqs, vocab)
    lams, edges = tune_lambdas(lm, dev_seqs)
    r = bits_per_char(lm, test_seqs, lams, n_chars, unk_chars, U)
    r["lambdas"] = list(lams)
    r["lambda_on_grid_edge"] = edges
    return r


def run_split(lang, seed, sents):
    """언어 하나, 분할 시드 하나에 대해 단어 단위(E1, E3)와 BPE(E2, E3), 처음 보는 글자(E5)를 잰다."""
    tr, dv, te = D.split(sents, seed)
    U = UNIVERSE[lang]
    freqs = B.word_freqs(tr)
    # 모든 방식의 bpc 분모를 같게(원문 글자 수) 맞춘다.
    n_chars = B.chars_per_sentence(te)
    # 평가 문장 안의 단어 수. 토큰 수 대비 단어당 토큰 수를 낼 때의 분모.
    test_words = sum(len(s.split(" ")) for s in te)
    out = {"train_word_tokens": sum(freqs.values()), "train_word_types": len(freqs),
           "test_chars": n_chars, "test_words": test_words}

    ft = B.word_freqs(te)
    # 어휘 제한 없이 "학습에 한 번도 없는 단어"인 테스트 토큰 비율. 단어 단위가 어휘를 아무리 키워도 남는 한계.
    out["test_tokens_unseen_full_vocab"] = sum(c for w, c in ft.items() if w not in freqs) / sum(ft.values())

    # E1 + E3(단어 단위)
    word = {}
    for V in WORD_V:
        vocab = B.word_vocab(freqs, V)
        tok = lambda ss: [B.word_tokenize(s, vocab) for s in ss]
        te_t = tok(te)
        # 단어 단위에서 <unk>가 가린 글자 수. bpc에 철자 비용으로 더한다.
        unk_chars = sum(len(w) for s in te for w in s.split(" ") if w not in vocab)
        r = _fit_and_score(tok(tr), tok(dv), te_t, vocab | {B.UNK}, n_chars, unk_chars, U)
        r["unk_rate"] = B.unk_rate(te, vocab)
        r["vocab_size"] = len(vocab) + 1
        # tokens에는 문장 끝 </s>가 포함돼 있다.
        r["tokens_per_sentence"] = r["tokens"] / len(te)  # </s> 포함
        word[str(V)] = r
    out["word"] = word

    # E2 + E3(BPE)
    t0 = time.perf_counter()
    all_merges = B.learn_bpe(freqs, max(MERGES))
    out["learn_bpe_seconds_8000"] = time.perf_counter() - t0
    bpe = {}
    for M in MERGES:
        # 한 번 8000개까지 학습한 병합 목록의 앞 M개를 잘라 쓴다. BPE 병합은 순서대로 쌓이므로 앞부분이 M회 학습 결과와 같다.
        tk = B.BPETokenizer(freqs, all_merges[:M])
        seqs = lambda ss: [tk.tokenize(s) for s in ss]
        te_t = seqs(te)
        flat = [t for s in te_t for t in s]
        # BPE에서는 <unk>가 항상 글자 하나이므로 개수 = 가린 글자 수.
        unk_chars = sum(t == B.UNK for t in flat)  # BPE에서 <unk>는 글자 하나
        r = _fit_and_score(seqs(tr), seqs(dv), te_t, list(tk.sym2id), n_chars, unk_chars, U)
        r["unk_rate"] = unk_chars / len(flat)
        r["vocab_size"] = tk.vocab_size
        r["tokens_per_sentence"] = r["tokens"] / len(te)
        r["tokens_per_word"] = len(flat) / test_words
        bpe[str(M)] = r
    out["bpe"] = bpe

    # E5 처음 보는 글자
    alpha = {c for w in freqs for c in w}
    test_chars = [c for s in te for c in s if c != " "]
    out["alphabet_size"] = len(alpha)
    out["test_chars_unseen_rate"] = float(np.mean([c not in alpha for c in test_chars]))
    out["utf8_bytes_per_char"] = sum(len(c.encode("utf-8")) for c in test_chars) / len(test_chars)
    return out


def e4_compute(sents, M=200):
    tr, _, _ = D.split(sents, 0)
    freqs = B.word_freqs(tr)
    t0 = time.perf_counter()
    # 시간은 기계 의존. 두 방식의 병합 목록이 같은지도 same_merges로 함께 기록.
    naive = B.learn_bpe_naive(freqs, M)
    t1 = time.perf_counter()
    fast = B.learn_bpe(freqs, M)
    t2 = time.perf_counter()
    return {"merges": M, "naive_seconds": t1 - t0, "fast_seconds": t2 - t1,
            "speedup": (t1 - t0) / (t2 - t1), "same_merges": naive == fast,
            "word_types": len(freqs)}


def _mean_std(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": [float(v) for v in vals]}


def summarize(runs):
    """시드 3개의 평균과 표준편차(모집단 std)를 만든다."""
    s = {}
    for key in ("train_word_types", "test_tokens_unseen_full_vocab", "test_chars_unseen_rate",
                "alphabet_size", "utf8_bytes_per_char", "learn_bpe_seconds_8000"):
        s[key] = _mean_std([r[key] for r in runs])
    for kind, keys in (("word", WORD_V), ("bpe", MERGES)):
        s[kind] = {}
        for k in keys:
            rows = [r[kind][str(k)] for r in runs]
            s[kind][str(k)] = {m: _mean_std([x[m] for x in rows])
                               for m in rows[0] if m not in ("lambdas", "lambda_on_grid_edge")}
            s[kind][str(k)]["lambdas_per_seed"] = [x["lambdas"] for x in rows]
            s[kind][str(k)]["lambda_edges_per_seed"] = [x["lambda_on_grid_edge"] for x in rows]
    return s


def main():
    # torch는 쓰지 않고 버전 기록용으로만 가져온다.
    import torch  # 버전 기록만

    results = {"env": {"python": platform.python_version(), "numpy": np.__version__,
                       "torch": torch.__version__, "machine": platform.machine()}}
    results["E0"] = e0_toy()
    corpora = {lang: D.make_corpus(lang) for lang in LANGS}
    results["corpus"] = {lang: {"sentences": len(s), "example": s[:2]} for lang, s in corpora.items()}
    for lang in LANGS:
        runs = []
        for seed in SEEDS:
            t = time.perf_counter()
            runs.append(run_split(lang, seed, corpora[lang]))
            print(f"{lang} seed {seed}: {time.perf_counter() - t:.1f}s", flush=True)
        results[lang] = {"summary": summarize(runs), "runs": runs}
    results["E4"] = e4_compute(corpora["en_like"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for lang in LANGS:
        print(lang, json.dumps(results[lang]["summary"], indent=1, ensure_ascii=False)[:3000])
    print(json.dumps({"E0": results["E0"], "E4": results["E4"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
