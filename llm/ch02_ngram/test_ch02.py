"""2장 검증. `uv run pytest -q llm/ch02_ngram` 로 실행한다."""

from fractions import Fraction

import numpy as np
import pytest

import ngram_np as ng
import ngram_torch as nt

# 손 계산용 작은 말뭉치: 사전 {</s>=0, a=1, b=2}, <s>=3. 문장 "a b"와 "b".
V_TOY, BOS_TOY = 3, 3
TOY = [[1, 2, 0], [2, 0]]


def _toy(n=2, method="kn", **kw):
    counts = ng.count_ngrams(TOY, n, BOS_TOY)
    cont = ng.continuation_counts(counts, n)
    return ng.NGramModel(counts, n, V_TOY, method=method, cont=cont, **kw)


def test_counts_by_hand():
    counts = ng.count_ngrams(TOY, 2, BOS_TOY)
    assert counts[2] == {(3,): {1: 1, 2: 1}, (1,): {2: 1}, (2,): {0: 2}}
    assert counts[1] == {(): {1: 1, 2: 2, 0: 2}}
    cont = ng.continuation_counts(counts, 2)
    # a 앞: {<s>}, b 앞: {<s>, a}, </s> 앞: {b}
    assert cont[1] == {(): {1: 1, 2: 2, 0: 1}}


def test_kneser_ney_matches_hand_computation():
    """D = 3/4. 손으로 푼 값 (README 대신 여기 적어 둔다).

    유니그램(continuation): N(a)=1, N(b)=2, N(</s>)=1, 합 4, 종류 3
      P1(b) = (2 - 3/4)/4 + (3/4)(3/4)(1/3) = 5/16 + 3/16 = 1/2,  P1(a) = P1(</s>) = 1/4
    바이그램:
      P(b | a)    = (1 - 3/4)/1 + (3/4)(1/1) P1(b)   = 1/4 + 3/8   = 5/8
      P(a | a)    = 0 + (3/4) P1(a)                  = 3/16
      P(</s> | b) = (2 - 3/4)/2 + (3/4)(1/2) P1(</s>) = 5/8 + 3/32 = 23/32
      P(a | <s>)  = (1 - 3/4)/2 + (3/4)(2/2) P1(a)   = 1/8 + 3/16 = 5/16
    """
    m = _toy()
    expect = {((1,), 2): Fraction(5, 8), ((1,), 1): Fraction(3, 16),
              ((2,), 0): Fraction(23, 32), ((3,), 1): Fraction(5, 16)}
    for (h, w), f in expect.items():
        assert m.prob(h, w) == pytest.approx(float(f), abs=1e-15)


def test_add_delta_and_mle_by_hand():
    mle = _toy(method="mle")
    assert mle.prob((2,), 0) == 1.0 and mle.prob((2,), 1) == 0.0
    assert mle.prob((0,), 1) == 0.0  # </s> 뒤 문맥은 본 적이 없다
    add1 = _toy(method="add", delta=1.0)
    assert add1.prob((2,), 0) == pytest.approx(3 / 5)  # (2 + 1) / (2 + 3)
    assert add1.prob((0,), 1) == pytest.approx(1 / 3)  # 본 적 없는 문맥은 균등


@pytest.fixture(scope="module")
def small_corpus():
    words, stoi = ng.grammar_vocab()
    rng = np.random.default_rng(0)
    train = ng.to_ids(ng.make_corpus(2000, range(7), rng), stoi)
    test = ng.to_ids(ng.make_corpus(200, range(7), rng), stoi)
    counts = ng.count_ngrams(train, 4, stoi[ng.BOS])
    cont = ng.continuation_counts(counts, 4)
    return len(words), stoi[ng.BOS], counts, cont, test


@pytest.mark.parametrize("method", ["add", "interp", "kn"])
@pytest.mark.parametrize("n", [1, 2, 4])
def test_smoothed_rows_sum_to_one_and_no_zeros(small_corpus, method, n):
    V, bos, counts, cont, test = small_corpus
    m = ng.NGramModel(counts, n, V, method=method, delta=0.1, lam=0.7, cont=cont)
    for i, (h, _) in enumerate(ng.positions(test, n, bos)):
        if i >= 60:
            break
        d = m.dist(h)
        assert abs(d.sum() - 1.0) < 1e-12
        assert d.min() > 0.0
    # 본 적 없는 문맥(</s>가 네 번 이어진 문맥)에서도 분포가 성립한다
    d = m.dist((0,) * (n - 1))
    assert abs(d.sum() - 1.0) < 1e-12 and d.min() > 0.0


def test_mle_perplexity_is_infinite_when_a_zero_appears(small_corpus):
    V, bos, counts, cont, test = small_corpus
    m = ng.NGramModel(counts, 4, V, method="mle")
    assert ng.perplexity(m, test, bos) == float("inf")


def test_torch_cross_entropy_matches_numpy(small_corpus):
    V, bos, counts, cont, test = small_corpus
    for method in ("add", "interp", "kn"):
        m = ng.NGramModel(counts, 3, V, method=method, delta=0.1, lam=0.7, cont=cont)
        rows, ys = nt.rows_and_targets(m, ng.positions(test, 3, bos))
        assert nt.perplexity_torch(rows, ys) == pytest.approx(ng.perplexity(m, test, bos), rel=1e-9)


def test_same_token_count_for_every_n(small_corpus):
    V, bos, counts, cont, test = small_corpus
    Ts = {n: sum(1 for _ in ng.positions(test, n, bos)) for n in (1, 2, 3, 4)}
    assert len(set(Ts.values())) == 1


def test_continuation_count_of_the_is_one(small_corpus):
    """'the'는 늘 <s> 뒤에만 온다: 횟수는 많지만 앞에 오는 단어 종류는 1개."""
    words, stoi = ng.grammar_vocab()
    V, bos, counts, cont, test = small_corpus
    assert cont[1][()][stoi["the"]] == 1
    assert counts[1][()][stoi["the"]] == 2000
    assert sum(cont[1][()].values()) == ng.nonzero_cells(counts, 2)  # continuation 합 = 바이그램 종류 수
