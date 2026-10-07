"""1장 검증. `uv run pytest -q llm/ch01_bigram` 로 실행한다."""

import numpy as np
import torch

import bigram_np as bnp
from bigram_torch import fit_bigram, perplexity_torch


def _grammar(seed=0, n=3000):
    stoi, itos = bnp.grammar_vocab()
    corpus = bnp.make_corpus(n, range(9), np.random.default_rng(seed))
    ids = [np.array([stoi[t] for t in toks]) for toks, *_ in corpus]
    return stoi, itos, corpus, ids


def test_rows_sum_to_one():
    stoi, itos, _, ids = _grammar()
    P = bnp.bigram_mle(bnp.count_sentences(ids, len(itos)))
    row = P.sum(1)
    seen = row > 0
    np.testing.assert_allclose(row[seen], 1.0, atol=1e-12)
    assert [itos[i] for i in np.where(~seen)[0]] == [bnp.EOS]  # </s> 뒤에는 아무것도 오지 않는다


def test_hand_counted_three_sentences():
    # 손으로 센 값: "the cat sat"이 2번, "the dog sat"이 1번
    sents = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "sat"]]
    stoi, itos = bnp.build_vocab([t for s in sents for t in s])
    counts = bnp.count_sentences([[stoi[t] for t in s] for s in sents], len(itos))
    P = bnp.bigram_mle(counts)
    assert counts[stoi["the"], stoi["cat"]] == 2 and counts[stoi["the"], stoi["dog"]] == 1
    assert P[stoi["the"], stoi["cat"]] == 2 / 3
    assert P[stoi["the"], stoi["dog"]] == 1 / 3
    assert P[stoi["cat"], stoi["sat"]] == 1.0
    # 문장 확률 = 곱: P(cat|the) P(sat|cat) = 2/3
    lp = bnp.sequence_log_prob(P, [stoi["the"], stoi["cat"], stoi["sat"]])
    assert np.isclose(np.exp(lp), 2 / 3)


def test_first_screen_example():
    """첫 화면 표: dog runs / dogs run / dog often runs / dogs often run."""
    sents = [["dog", "runs"], ["dogs", "run"], ["dog", "often", "runs"], ["dogs", "often", "run"]]
    stoi, itos = bnp.build_vocab([t for s in sents for t in s])
    P = bnp.bigram_mle(bnp.count_sentences([[stoi[t] for t in s] for s in sents], len(itos)))
    assert P[stoi["dog"], stoi["runs"]] == 0.5 and P[stoi["dog"], stoi["run"]] == 0.0
    assert P[stoi["often"], stoi["runs"]] == P[stoi["often"], stoi["run"]] == 0.5


def test_numpy_mle_matches_torch_training():
    stoi, itos, _, ids = _grammar()
    counts = bnp.count_sentences(ids, len(itos))
    P = bnp.bigram_mle(counts)
    torch.set_num_threads(1)
    model, _ = fit_bigram(counts, steps=3000, lr=0.1)
    T = model.table().detach().numpy()
    seen = counts.sum(1) > 0
    assert np.abs(T[seen] - P[seen]).max() < 1e-3


def test_perplexity_numpy_matches_torch():
    stoi, itos, _, ids = _grammar()
    P = bnp.bigram_mle(bnp.count_sentences(ids, len(itos)))
    assert abs(bnp.perplexity(P, ids) - perplexity_torch(P, ids)) < 1e-9


def test_unseen_bigram_gives_infinite_perplexity():
    stoi, itos, _, ids = _grammar()
    P = bnp.bigram_mle(bnp.count_sentences(ids, len(itos)))
    bad = [np.array([stoi["<s>"], stoi["the"], stoi["dog"], stoi["run"], stoi["</s>"]])]  # dog 뒤에 run은 없다
    assert np.isinf(bnp.perplexity(P, bad))


def test_sample_is_deterministic_with_seed():
    stoi, itos, _, ids = _grammar()
    P = bnp.bigram_mle(bnp.count_sentences(ids, len(itos)))
    a = bnp.sample(P, stoi["<s>"], 30, np.random.default_rng(7), eos=stoi["</s>"], bos=stoi["<s>"])
    b = bnp.sample(P, stoi["<s>"], 30, np.random.default_rng(7), eos=stoi["</s>"], bos=stoi["<s>"])
    assert a == b


def test_bigram_agreement_fails_only_past_distance_one():
    stoi, itos, _, ids = _grammar(n=5000)
    P = bnp.bigram_mle(bnp.count_sentences(ids, len(itos)))
    f = lambda ctx, w: P[stoi[ctx[-1]], stoi[w]]
    rng = np.random.default_rng(99)
    near = bnp.agreement_scores(f, bnp.make_corpus(500, [0], rng), stoi)
    far = bnp.agreement_scores(f, bnp.make_corpus(2000, [3], rng), stoi)
    assert near["acc"] > 0.99  # 못 본 (명사, 동사) 짝이 드물게 동률(0.5)을 만든다
    assert abs(far["acc"] - 0.5) < 0.05  # 우연 수준
