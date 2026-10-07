"""2장 NumPy·표준 라이브러리 최소 구현: n-gram 카운트, MLE, 덧셈 평활, 보간, 보간형 Kneser-Ney.

표기
- 예측 대상 사전: 0..V-1 (문장 끝 </s> 포함). 문장 시작 <s>는 id V로 두고 예측 대상에서 뺀다.
- 모든 n에서 같은 토큰을 예측하도록 문장 앞에 <s>를 (최대 n - 1)개 붙인다. 그래서 n이 달라도
  perplexity의 분모(예측한 토큰 수)가 같다.
- counts[m][h][w]: 길이 m-1 문맥 h 뒤에 w가 나온 횟수 (m = 1..N). h는 튜플이고 m = 1이면 ().

n-gram에는 학습할 파라미터가 없다. 확률표는 셈(count)과 나눗셈으로 바로 정해진다.
"""

from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# 말뭉치: 1장(ch01_bigram/bigram_np.py)의 합성 문법을 그대로 복사했다.
# "<s> the 주어 부사×k 동사 </s>", 수(단수/복수)는 주어와 동사만 정한다.
# ---------------------------------------------------------------------------

BOS, EOS = "<s>", "</s>"
NOUNS = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("child", "children"),
         ("teacher", "teachers"), ("farmer", "farmers"), ("girl", "girls"), ("boy", "boys")]
VERBS = [("runs", "run"), ("sleeps", "sleep"), ("sings", "sing"), ("eats", "eat"),
         ("waits", "wait"), ("laughs", "laugh"), ("reads", "read"), ("swims", "swim")]
ADVERBS = ["often", "still", "really", "never", "always", "quietly",
           "slowly", "usually", "sometimes", "rarely", "happily", "also"]


def grammar_vocab():
    """예측 대상 단어 목록(<s> 제외)과 단어→id 사전. <s>의 id는 len(목록)."""
    words = [EOS, "the"]
    for s, p in NOUNS:
        words += [s, p]
    for s, p in VERBS:
        words += [s, p]
    words += ADVERBS
    stoi = {w: i for i, w in enumerate(words)}
    stoi[BOS] = len(words)
    return words, stoi


def make_sentence(k, rng):
    """반환: (단어 목록(<s>, </s> 없이), 동사 위치, 주어가 복수인지, 동사 번호)."""
    plural = int(rng.integers(2))
    noun = NOUNS[int(rng.integers(len(NOUNS)))][plural]
    vi = int(rng.integers(len(VERBS)))
    advs = [ADVERBS[int(i)] for i in rng.integers(len(ADVERBS), size=k)]
    return ["the", noun] + advs + [VERBS[vi][plural]], 2 + k, plural, vi


def make_corpus(n, ks, rng):
    ks = list(ks)
    return [make_sentence(ks[int(rng.integers(len(ks)))], rng) for _ in range(n)]


def to_ids(corpus, stoi):
    """문장마다 단어 id 목록 + </s>. <s>는 모델이 필요한 만큼 붙인다."""
    return [[stoi[w] for w in toks] + [stoi[EOS]] for toks, *_ in corpus]


# ---------------------------------------------------------------------------
# 카운트
# ---------------------------------------------------------------------------

def count_ngrams(sents, N, bos):
    """counts[m][h][w] (m = 1..N)를 dict로 센다. 본 칸만 저장하는 희소 표다."""
    counts = {m: defaultdict(lambda: defaultdict(int)) for m in range(1, N + 1)}
    for ids in sents:
        seq = [bos] * (N - 1) + ids
        for i in range(N - 1, len(seq)):
            w = seq[i]
            for m in range(1, N + 1):
                counts[m][tuple(seq[i - m + 1 : i])][w] += 1
    return {m: {h: dict(t) for h, t in c.items()} for m, c in counts.items()}


def continuation_counts(counts, N):
    """Kneser-Ney의 낮은 차수용 카운트. cont[m][h][w] = (v, h, w)를 본 서로 다른 v의 수.

    Chen·Goodman(1998) §2.7: 낮은 차수 확률은 단어가 나온 횟수가 아니라
    "그 단어 앞에 온 서로 다른 단어의 수"에 비례하게 둔다.
    """
    cont = {}
    for m in range(1, N):
        c = defaultdict(lambda: defaultdict(int))
        for h_long, table in counts[m + 1].items():
            h = h_long[1:]
            for w in table:
                c[h][w] += 1
        cont[m] = {h: dict(t) for h, t in c.items()}
    return cont


def _with_totals(tables):
    """{h: {w: c}} -> {h: (합계, 종류 수, {w: c})}."""
    return {h: (sum(t.values()), len(t), t) for h, t in tables.items()}


def nonzero_cells(counts, n):
    """n-gram 표에서 실제로 0이 아닌 칸 수."""
    return sum(len(t) for t in counts[n].values())


# ---------------------------------------------------------------------------
# 모델: prob(h, w) -> P(w | h). h는 길이 n-1 튜플(앞쪽은 <s>로 채움)
# ---------------------------------------------------------------------------

class NGramModel:
    """counts는 N차까지 센 것을 공유하고, 모델 차수 n(<= N)과 평활 방법만 바꾼다."""

    def __init__(self, counts, n, V, method="mle", delta=1.0, lam=0.5, D=0.75, cont=None):
        self.n, self.V, self.method = n, V, method
        self.delta, self.lam, self.D = delta, lam, D
        self.raw = {m: _with_totals(counts[m]) for m in range(1, n + 1)}
        if method == "kn":
            assert cont is not None
            self.kn = {m: _with_totals(cont[m]) for m in range(1, n)}
            self.kn[n] = self.raw[n]  # 가장 높은 차수만 실제 카운트를 쓴다

    def _ml(self, m, h, w):
        e = self.raw[m].get(h[len(h) - (m - 1):] if m > 1 else ())
        if e is None:
            return None  # 문맥을 본 적이 없다: MLE는 정의되지 않는다(0/0)
        total, _, t = e
        return t.get(w, 0) / total

    def prob(self, h, w):
        n = self.n
        if self.method == "mle":
            p = self._ml(n, h, w)
            return 0.0 if p is None else p
        if self.method == "add":
            e = self.raw[n].get(h[len(h) - (n - 1):] if n > 1 else ())
            total, c = (0, 0) if e is None else (e[0], e[2].get(w, 0))
            return (c + self.delta) / (total + self.delta * self.V)
        if self.method == "interp":
            return self._interp(n, h, w)
        if self.method == "kn":
            return self._kn(n, h, w)
        raise ValueError(self.method)

    def _interp(self, m, h, w):
        """Jelinek-Mercer식 보간: P_m = λ P_ML,m + (1-λ) P_{m-1}, P_0 = 1/V."""
        if m == 0:
            return 1.0 / self.V
        lower = self._interp(m - 1, h, w)
        p = self._ml(m, h, w)
        if p is None:
            return lower  # 본 적 없는 문맥은 낮은 차수에 전부 맡긴다
        return self.lam * p + (1 - self.lam) * lower

    def _kn(self, m, h, w):
        """보간형 Kneser-Ney, 고정 할인 D (Chen·Goodman 1998 식 (22)의 꼴).

        P_m(w|h) = max(c(h,w) - D, 0) / c(h) + D * N1+(h·) / c(h) * P_{m-1}(w|h')
        가장 높은 차수는 실제 카운트, 낮은 차수는 continuation 카운트. P_0 = 1/V.
        """
        if m == 0:
            return 1.0 / self.V
        lower = self._kn(m - 1, h, w)
        e = self.kn[m].get(h[len(h) - (m - 1):] if m > 1 else ())
        if e is None:
            return lower
        total, types, t = e
        return max(t.get(w, 0) - self.D, 0.0) / total + self.D * types / total * lower

    def dist(self, h):
        """문맥 h 뒤의 전체 분포 (V,). 정규화 검사와 PyTorch 대조에 쓴다."""
        return np.array([self.prob(h, w) for w in range(self.V)])


def positions(sents, n, bos):
    """(문맥 튜플, 정답 id)를 차례로 낸다. 모든 n에서 같은 토큰 수."""
    for ids in sents:
        seq = [bos] * (n - 1) + ids
        for i in range(n - 1, len(seq)):
            yield tuple(seq[i - n + 1 : i]), seq[i]


def perplexity(model, sents, bos):
    """PPL = exp(-(1/T) Σ log P(w_t | 문맥)). 확률 0이 하나라도 있으면 inf."""
    logs, T = 0.0, 0
    for h, w in positions(sents, model.n, bos):
        p = model.prob(h, w)
        if p <= 0.0:
            return float("inf")
        logs += np.log(p)
        T += 1
    return float(np.exp(-logs / T))


def sparsity_stats(counts, n, sents, bos):
    """처음 보는 (n-1)-문맥 비율, 처음 보는 n-gram 비율, 확률 0 토큰이 있는 문장 비율(MLE)."""
    ctx_tab = counts[n]
    T = unseen_ctx = unseen_gram = 0
    zero_sent = 0
    for ids in sents:
        seq = [bos] * (n - 1) + ids
        bad = False
        for i in range(n - 1, len(seq)):
            h, w = tuple(seq[i - n + 1 : i]), seq[i]
            T += 1
            t = ctx_tab.get(h)
            if t is None:
                unseen_ctx += 1
                unseen_gram += 1
                bad = True
            elif w not in t:
                unseen_gram += 1
                bad = True
        zero_sent += bad
    return {"unseen_context_rate": unseen_ctx / T, "unseen_ngram_rate": unseen_gram / T,
            "zero_prob_sentence_rate": zero_sent / len(sents), "tokens": T}


def agreement_accuracy(model, corpus, stoi, bos):
    """동사 자리에서 맞는 수의 동사(runs/run)를 더 높게 쳤는지. k별로 모은다.

    맞는 쪽이 크면 1, 같으면 0.5, 작으면 0. 찍으면 0.5다.
    """
    by_k = defaultdict(list)
    n = model.n
    for toks, vpos, plural, vi in corpus:
        seq = [bos] * (n - 1) + [stoi[x] for x in toks[:vpos]]
        h = tuple(seq[len(seq) - (n - 1):]) if n > 1 else ()
        right = model.prob(h, stoi[VERBS[vi][plural]])
        wrong = model.prob(h, stoi[VERBS[vi][1 - plural]])
        by_k[vpos - 2].append(1.0 if right > wrong else (0.5 if right == wrong else 0.0))
    return {k: float(np.mean(v)) for k, v in sorted(by_k.items())}
