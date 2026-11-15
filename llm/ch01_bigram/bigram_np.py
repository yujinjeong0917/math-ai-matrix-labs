"""1장 NumPy 최소 구현: 바이그램 언어모델.

P(w_1..w_T) = prod_t P(w_t | w_<t) 를 그대로 추정할 수는 없으니,
마르코프 가정 P(w_t | w_<t) ~= P(w_t | w_{t-1}) 로 바로 앞 단어 하나만 본다.
최대우도추정(MLE)은 셈 하나로 끝난다: P(b | a) = c(a, b) / c(a).

설계도(docs/tracks/llm.md)의 함수 이름을 따른다. 파일 이름은 장끼리
pytest 모듈 이름이 겹치지 않도록 bigram_np.py 로 했다(7장의 attention_np.py 관례).
"""

import numpy as np

BOS, EOS = "<s>", "</s>"


def build_vocab(tokens):
    """토큰 목록에서 사전을 만든다. 등장 순서대로 번호를 매겨 결정적이다."""
    itos = []
    stoi = {}
    for tok in tokens:
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)
    return stoi, itos


def count_bigrams(ids, V):
    """ids: 정수 배열(문장 경계 포함). (V, V) 카운트 표 c[a, b] 를 돌려준다.

    한 번 훑으며 칸 하나씩 더하므로 시간은 O(T), 메모리는 표 크기 V^2 이다.
    """
    ids = np.asarray(ids)
    counts = np.zeros((V, V), dtype=np.int64)
    # (앞 단어, 다음 단어) 쌍마다 칸 (a, b)에 1을 더한다. 같은 칸이 반복돼도 누적되도록 counts[idx] += 1 대신 np.add.at을 쓴다.
    # ids[:-1]은 앞 단어 열, ids[1:]은 한 칸 밀린 다음 단어 열이라 길이 T-1개의 쌍이 나온다.
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts


def count_sentences(sentences_ids, V):
    """문장마다 따로 세서 문장과 문장 사이(</s> -> <s>)는 세지 않는다."""
    counts = np.zeros((V, V), dtype=np.int64)
    for ids in sentences_ids:
        counts += count_bigrams(ids, V)
    return counts


def bigram_mle(counts):
    """P[a, b] = c(a, b) / c(a). 한 번도 앞에 나온 적 없는 행(예: </s>)은 0으로 둔다."""
    counts = np.asarray(counts, dtype=np.float64)
    # c(a) = 행 합. (V, 1) 모양이라 (V, V) 카운트를 행별로 나눌 수 있다.
    row = counts.sum(axis=1, keepdims=True)
    # where=row > 0 인 행만 나눈다. 안 나눈 칸은 out의 0으로 남아 0으로 나누기 경고를 피한다.
    return np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)


def unigram_mle(sentences_ids, V, skip_first=True):
    """비교용 유니그램: 앞 단어를 전혀 보지 않는 P(b). 첫 토큰 <s>는 예측 대상이 아니라 뺀다."""
    c = np.zeros(V, dtype=np.float64)
    for ids in sentences_ids:
        # skip_first=True면 첫 토큰(<s>)을 센 대상에서 뺀다. 바이그램 PPL과 예측 토큰 수를 맞추려는 선택.
        np.add.at(c, np.asarray(ids[1:] if skip_first else ids), 1)
    return c / c.sum()


def sequence_log_prob(P, ids):
    """log P(w_2..w_T | w_1) = sum_t log P[w_{t-1}, w_t]. 확률 0인 칸이 있으면 -inf."""
    ids = np.asarray(ids)
    # 각 위치 t의 P[w_{t-1}, w_t]를 한 번에 꺼낸다. 모양 (T-1,).
    p = P[ids[:-1], ids[1:]]
    # log(0) = -inf 경고만 끈다. 값 자체는 -inf로 남아 PPL이 무한대가 된다(테스트 test_unseen_bigram_gives_infinite_perplexity).
    with np.errstate(divide="ignore"):
        return float(np.log(p).sum())


def perplexity(P, sentences_ids):
    """PPL = exp(-(1/N) sum log P). N은 예측한 토큰 수(문장마다 첫 <s> 제외)."""
    total, n = 0.0, 0
    for ids in sentences_ids:
        total += sequence_log_prob(P, ids)
        n += len(ids) - 1
    return float(np.exp(-total / n))


# 유니그램 PPL. perplexity와 같은 식에서 P[a, b] 대신 앞 단어와 무관한 p[b]를 쓴다.
def unigram_perplexity(p, sentences_ids):
    total, n = 0.0, 0
    for ids in sentences_ids:
        total += float(np.log(p[np.asarray(ids[1:])]).sum())
        n += len(ids) - 1
    return float(np.exp(-total / n))


def sample(P, start, n, rng, eos=None, bos=None):
    """start에서 시작해 n개 토큰을 뽑는다. eos를 뽑으면 bos로 돌아가 다음 문장을 잇는다."""
    out = [start]
    cur = start
    for _ in range(n):
        if eos is not None and cur == eos:
            cur = bos
            out.append(cur)
            continue
        cur = int(rng.choice(P.shape[1], p=P[cur]))
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# 실패하는 최소 예제: 주어(단수/복수) + 부사 k개 + 동사(단수/복수형)
# ---------------------------------------------------------------------------

NOUNS = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("child", "children"),
         ("teacher", "teachers"), ("farmer", "farmers"), ("girl", "girls"), ("boy", "boys")]
VERBS = [("runs", "run"), ("sleeps", "sleep"), ("sings", "sing"), ("eats", "eat"),
         ("waits", "wait"), ("laughs", "laugh"), ("reads", "read"), ("swims", "swim")]
ADVERBS = ["often", "still", "really", "never", "always", "quietly",
           "slowly", "usually", "sometimes", "rarely", "happily", "also"]


def grammar_vocab():
    toks = [BOS, EOS, "the"]
    for s, p in NOUNS:
        toks += [s, p]
    for s, p in VERBS:
        toks += [s, p]
    toks += ADVERBS
    return build_vocab(toks)


def make_sentence(k, rng):
    """'<s> the 주어 부사×k 동사 </s>'. 수(단수/복수)는 주어와 동사만 정한다.

    반환: (토큰 목록, 동사 위치, 주어가 복수인지, 동사 원형 번호)
    """
    # 0 = 단수, 1 = 복수. NOUNS/VERBS의 쌍 (단수형, 복수형)을 이 값으로 고른다. 정답 일치의 기준이 된다.
    plural = int(rng.integers(2))
    noun = NOUNS[int(rng.integers(len(NOUNS)))][plural]
    vi = int(rng.integers(len(VERBS)))
    advs = [ADVERBS[int(i)] for i in rng.integers(len(ADVERBS), size=k)]
    toks = [BOS, "the", noun] + advs + [VERBS[vi][plural], EOS]
    # 동사 위치 = <s>, the, 명사 다음부터 부사 k개 뒤이므로 인덱스 3 + k.
    return toks, 3 + k, plural, vi


def make_corpus(n, ks, rng):
    """ks에서 k를 고르게 뽑아 문장 n개를 만든다."""
    ks = list(ks)
    return [make_sentence(ks[int(rng.integers(len(ks)))], rng) for _ in range(n)]


def agreement_scores(prob_fn, corpus, stoi):
    """동사 자리에서 맞는 수의 동사를 더 높게 쳤는지 센다.

    prob_fn(tokens_before_verb, verb_token) -> 확률. 두 형태(runs/run)의 확률을 비교해
    맞는 쪽이 크면 1, 같으면 0.5(동전 던지기), 작으면 0.
    반환: dict(acc=정확도, ties=동률 비율,
              bits=동사가 온다는 걸 알 때 정답 동사의 평균 놀람 -log2 P(정답 | 문맥, 동사 자리))
    bits는 동사 16개 형태 안에서 다시 정규화한다. "언제 문장을 끝내나"의 불확실성을 빼고
    "어느 동사인가"만 보려는 것이다. 원형 8개 중 하나면 3비트, 수까지 모르면 4비트.
    """
    forms = [w for pair in VERBS for w in pair]
    hits, ties, bits = 0.0, 0, 0.0
    for toks, vpos, plural, vi in corpus:
        # 동사 앞까지가 문맥. 모델은 이 문맥만 보고 두 동사 형태의 확률을 낸다.
        ctx = toks[:vpos]
        right = prob_fn(ctx, VERBS[vi][plural])
        wrong = prob_fn(ctx, VERBS[vi][1 - plural])
        if right == wrong:
            ties += 1
        hits += 1.0 if right > wrong else (0.5 if right == wrong else 0.0)
        # 동사 16개 형태의 확률 합 z로 정규화해 "동사 자리" 안에서의 확률만 본다.
        z = sum(prob_fn(ctx, w) for w in forms)
        with np.errstate(divide="ignore", invalid="ignore"):
            bits += -np.log2(right / z) if z > 0 else np.inf
    n = len(corpus)
    return {"acc": hits / n, "ties": ties / n, "bits": float(bits / n)}
