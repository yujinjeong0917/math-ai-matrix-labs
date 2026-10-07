"""3장 최소 구현: 단어 단위 어휘와 BPE(byte pair encoding) 토크나이저.

BPE 학습 규칙은 Sennrich, Haddow, Birch(2016) §3.2를 따른다.
  - 단어를 글자 열 + 단어 끝 기호 '</w>'로 시작한다.
  - 말뭉치에서 가장 잦은 인접 쌍 (A, B)를 새 기호 AB로 바꾸기를 M번 반복한다.
  - 단어 경계를 넘는 쌍은 세지 않으므로, 단어별 빈도 사전 위에서 돌릴 수 있다.
  - 어휘 크기 = 처음 글자 수 + 병합 횟수 M.

동률 규칙(논문에는 없음, 이 구현의 선택): 빈도가 같으면 쌍 (A, B)를 문자열 사전순으로
비교해 가장 앞선 쌍을 고른다. 그래서 같은 입력이면 병합 목록이 항상 같다.
"""

import heapq
from collections import Counter, defaultdict

import numpy as np

EOW = "</w>"
UNK = "<unk>"


# ---------------------------------------------------------------- 단어 단위 어휘

def word_freqs(sentences):
    return Counter(w for s in sentences for w in s.split(" "))


def word_vocab(freqs, V):
    """빈도 상위 V개 단어. 빈도가 같으면 사전순."""
    ranked = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w for w, _ in ranked[:V]}


def word_tokenize(sentence, vocab):
    return [w if w in vocab else UNK for w in sentence.split(" ")]


def unk_rate(sentences, vocab):
    """테스트 토큰 중 <unk>가 된 비율."""
    toks = [t for s in sentences for t in word_tokenize(s, vocab)]
    return float(np.mean(np.array(toks) == UNK))


# ---------------------------------------------------------------- BPE 학습

def _pairs(symbols):
    return zip(symbols[:-1], symbols[1:])


def _merge_word(symbols, pair):
    """왼쪽부터 겹치지 않게 (A, B)를 AB로 바꾼다. 논문 Algorithm 1의 정규식 치환과 같은 동작."""
    a, b = pair
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def learn_bpe_naive(freqs, num_merges, tie="lex"):
    """병합마다 모든 쌍을 처음부터 다시 센다. 원리 확인·대조용(느림).

    tie="lex": 동률이면 사전순(이 구현의 기본 규칙).
    tie="first": 동률이면 처음 센 쌍. 논문 Algorithm 1의 max(pairs, key=pairs.get)과 같은 동작.
    """
    vocab = {tuple(w) + (EOW,): f for w, f in freqs.items()}
    merges = []
    for _ in range(num_merges):
        stats = defaultdict(int)
        for sym, f in vocab.items():
            for p in _pairs(sym):
                stats[p] += f
        if not stats:
            break
        if tie == "first":
            best = max(stats, key=stats.get)
        else:
            best = min(stats, key=lambda p: (-stats[p], p))  # 가장 잦은 쌍, 동률이면 사전순
        merges.append(best)
        vocab = {tuple(_merge_word(list(sym), best)): f for sym, f in vocab.items()}
    return merges


def learn_bpe(freqs, num_merges):
    """learn_bpe_naive와 같은 병합 목록을 내는 빠른 구현.

    논문이 "in practice, we increase efficiency by indexing all pairs, and updating data
    structures incrementally"라고 쓴 방식이다. 쌍마다 등장하는 단어 목록을 들고 있다가
    병합된 쌍이 있는 단어만 다시 센다. 최댓값은 힙에서 꺼내고, 낡은 항목은 버린다.
    """
    words = [list(w) + [EOW] for w in freqs]
    fs = list(freqs.values())
    counts = defaultdict(int)
    where = defaultdict(set)
    for i, (w, f) in enumerate(zip(words, fs)):
        for p in _pairs(w):
            counts[p] += f
            where[p].add(i)
    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) < num_merges and heap:
        negc, best = heapq.heappop(heap)
        if counts.get(best, 0) != -negc or negc == 0:
            continue  # 낡은 항목
        merges.append(best)
        changed = set()
        for i in list(where[best]):
            w, f = words[i], fs[i]
            if best not in set(_pairs(w)):
                continue
            for p in _pairs(w):
                counts[p] -= f
                changed.add(p)
            w2 = _merge_word(w, best)
            for p in _pairs(w2):
                counts[p] += f
                where[p].add(i)
                changed.add(p)
            words[i] = w2
        for p in changed:
            if counts[p] > 0:
                heapq.heappush(heap, (-counts[p], p))
    return merges


# ---------------------------------------------------------------- BPE 적용

class BPETokenizer:
    def __init__(self, train_freqs, merges):
        self.alphabet = sorted({c for w in train_freqs for c in w})
        self.merges = list(merges)
        self.ranks = {p: r for r, p in enumerate(self.merges)}
        symbols = self.alphabet + [EOW] + [a + b for a, b in self.merges] + [UNK]
        self.sym2id = {}
        for s in symbols:  # 병합 결과가 이미 있는 기호와 같으면 한 번만 넣는다
            self.sym2id.setdefault(s, len(self.sym2id))
        self.id2sym = {i: s for s, i in self.sym2id.items()}
        self._alpha = set(self.alphabet)
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.sym2id)

    def apply_bpe(self, word):
        """학습한 병합을 순위가 낮은(먼저 배운) 것부터 적용한다. 처음 보는 글자는 <unk>."""
        if word in self._cache:
            return self._cache[word]
        sym = [c if c in self._alpha else UNK for c in word] + [EOW]
        while len(sym) > 1:
            cand = [(self.ranks.get(p, 1 << 60), p) for p in _pairs(sym)]
            r, p = min(cand)
            if r == 1 << 60:
                break
            sym = _merge_word(sym, p)
        self._cache[word] = sym
        return sym

    def tokenize(self, sentence):
        return [t for w in sentence.split(" ") for t in self.apply_bpe(w)]

    def encode(self, text):
        return [self.sym2id[t] for t in self.tokenize(text)]

    def decode(self, ids):
        return "".join(self.id2sym[i] for i in ids).replace(EOW, " ")[:-1]


def chars_per_sentence(sentences):
    """비교 잣대의 분모: 원문 글자 수(단어 사이 공백 포함) + 문장 끝 1."""
    return sum(len(s) + 1 for s in sentences)
