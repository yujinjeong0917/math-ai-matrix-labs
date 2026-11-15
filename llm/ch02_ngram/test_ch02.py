"""2장 검증. `uv run pytest -q llm/ch02_ngram` 로 실행한다."""

from fractions import Fraction

import numpy as np
import pytest

import ngram_np as ng
import ngram_torch as nt

# 손 계산용 작은 말뭉치: 사전 {</s>=0, a=1, b=2}, <s>=3. 문장 "a b"와 "b".
# TOY 문장은 <s> 뒤에 "a b </s>"와 "b </s>".
V_TOY, BOS_TOY = 3, 3
TOY = [[1, 2, 0], [2, 0]]


def _toy(n=2, method="kn", **kw):
    counts = ng.count_ngrams(TOY, n, BOS_TOY)
    cont = ng.continuation_counts(counts, n)
    return ng.NGramModel(counts, n, V_TOY, method=method, cont=cont, **kw)


def test_counts_by_hand():
    counts = ng.count_ngrams(TOY, 2, BOS_TOY)
    # 바이그램을 손으로 센 값. 문맥 (3,)=<s> 뒤에 a 1번, b 1번. (1,)=a 뒤에 b 1번. (2,)=b 뒤에 </s> 2번.
    assert counts[2] == {(3,): {1: 1, 2: 1}, (1,): {2: 1}, (2,): {0: 2}}
    # 유니그램 카운트: a 1번, b 2번, </s> 2번.
    assert counts[1] == {(): {1: 1, 2: 2, 0: 2}}
    cont = ng.continuation_counts(counts, 2)
    # continuation: 단어 앞에 온 서로 다른 단어 종류 수. a는 <s> 하나, b는 <s>와 a 둘, </s>는 b 하나.
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
    # Fraction으로 쓴 정확한 분수를 float으로 바꿔 부동소수점 오차 1e-15 안에서 비교한다.
    for (h, w), f in expect.items():
        assert m.prob(h, w) == pytest.approx(float(f), abs=1e-15)


def test_add_delta_and_mle_by_hand():
    mle = _toy(method="mle")
    assert mle.prob((2,), 0) == 1.0 and mle.prob((2,), 1) == 0.0
    # MLE는 본 적 없는 문맥이면 0으로 둔다(코드의 선택, 0/0은 정의되지 않음).
    assert mle.prob((0,), 1) == 0.0  # </s> 뒤 문맥은 본 적이 없다
    add1 = _toy(method="add", delta=1.0)
    assert add1.prob((2,), 0) == pytest.approx(3 / 5)  # (2 + 1) / (2 + 3)
    assert add1.prob((0,), 1) == pytest.approx(1 / 3)  # 본 적 없는 문맥은 균등


@pytest.fixture(scope="module")
# 합성 문법 학습 2000문장, 평가 200문장. 시드 0으로 고정한 공용 입력.
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
    # </s> 뒤에 또 </s>가 오는 문맥은 학습 말뭉치에 없다. 평활 모델은 이때도 확률 합 1이고 0이 없어야 한다.
    # 본 적 없는 문맥(</s>가 네 번 이어진 문맥)에서도 분포가 성립한다
    d = m.dist((0,) * (n - 1))
    assert abs(d.sum() - 1.0) < 1e-12 and d.min() > 0.0


def test_mle_perplexity_is_infinite_when_a_zero_appears(small_corpus):
    V, bos, counts, cont, test = small_corpus
    m = ng.NGramModel(counts, 4, V, method="mle")
    # 4-gram MLE는 평가 문장에서 처음 보는 4-gram이 나와 확률 0이 생겨 PPL이 무한대. (이 시드와 크기에서 그렇다는 것을 확인하는 테스트)
    assert ng.perplexity(m, test, bos) == float("inf")


def test_torch_cross_entropy_matches_numpy(small_corpus):
    V, bos, counts, cont, test = small_corpus
    for method in ("add", "interp", "kn"):
        m = ng.NGramModel(counts, 3, V, method=method, delta=0.1, lam=0.7, cont=cont)
        rows, ys = nt.rows_and_targets(m, ng.positions(test, 3, bos))
        # 상대 오차 1e-9. 두 구현이 같은 값을 다른 경로로 계산하는지 본다.
        assert nt.perplexity_torch(rows, ys) == pytest.approx(ng.perplexity(m, test, bos), rel=1e-9)


def test_same_token_count_for_every_n(small_corpus):
    V, bos, counts, cont, test = small_corpus
    Ts = {n: sum(1 for _ in ng.positions(test, n, bos)) for n in (1, 2, 3, 4)}
    # <s> 패딩 덕에 어떤 n이든 예측한 토큰 수가 같다.
    assert len(set(Ts.values())) == 1


def test_continuation_count_of_the_is_one(small_corpus):
    """'the'는 늘 <s> 뒤에만 온다: 횟수는 많지만 앞에 오는 단어 종류는 1개."""
    words, stoi = ng.grammar_vocab()
    V, bos, counts, cont, test = small_corpus
    # the는 항상 <s> 뒤라 앞 단어 종류는 1. 반면 등장 횟수는 문장 수 2000.
    assert cont[1][()][stoi["the"]] == 1
    assert counts[1][()][stoi["the"]] == 2000
    assert sum(cont[1][()].values()) == ng.nonzero_cells(counts, 2)  # continuation 합 = 바이그램 종류 수
