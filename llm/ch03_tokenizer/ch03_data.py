"""3장 합성 말뭉치. 외부 텍스트 없이 고정 시드로 만든다.

두 가지 '흉내' 언어를 만든다. 둘 다 진짜 언어가 아니라, 단어가 어간 + 접사로
조립되고 빈도가 Zipf 꼴(순위 r의 빈도가 1/r에 비례)을 따르도록 만든 생성기다.

- en_like: 알파벳 26자. 어간 + 영어식 접미사(-s, -ed, -ing ...), 가끔 접두사.
- ko_like: 한글 음절. 어간 + 조사·어미(은/는/에서/에서도 ...)가 한 어절에 붙는다.
  어절의 2%는 전체 한글 음절 11,172자에서 고른 '드문 이름'이라, 학습에 없던
  글자가 테스트에 나올 수 있다.

두 언어의 미등록어 비율 차이는 이 생성기를 그렇게 만들었기 때문이지,
실제 한국어·영어에 대한 증거가 아니다.
"""

import numpy as np

EN_SUFFIX = ["", "s", "ed", "ing", "er", "est", "ly", "ness", "able", "ers", "ings"]
EN_SUFFIX_P = [0.40, 0.14, 0.09, 0.09, 0.06, 0.03, 0.05, 0.03, 0.03, 0.04, 0.04]
EN_PREFIX = ["", "un", "re", "over", "pre"]
EN_PREFIX_P = [0.82, 0.06, 0.06, 0.03, 0.03]

KO_PARTICLE = ["", "은", "는", "이", "가", "을", "를", "에", "에서", "에게", "으로", "로", "와", "과", "도",
               "만", "의", "까지", "부터", "에서도", "에서만", "까지도", "에게는", "으로는", "하고",
               "이다", "입니다", "했다", "하는", "해서", "하지만", "이라고", "에서는", "부터는"]
HANGUL_FIRST, HANGUL_COUNT = 0xAC00, 11172


def _zipf_p(n, s=1.0):
    p = 1.0 / np.arange(1, n + 1) ** s
    return p / p.sum()


def _make_en_stems(rng, n):
    cons, vows = list("bcdfghklmnprstvwz"), list("aeiou")
    stems = set()
    while len(stems) < n:
        k = rng.integers(1, 4)  # 음절 1~3개
        s = "".join(rng.choice(cons) + rng.choice(vows) for _ in range(k))
        if rng.random() < 0.5:
            s += rng.choice(cons)
        stems.add(s)
    return sorted(stems, key=lambda w: (len(w), w))


def _make_ko_stems(rng, n):
    common = [chr(HANGUL_FIRST + int(i)) for i in rng.choice(HANGUL_COUNT, 400, replace=False)]
    cp = _zipf_p(len(common), 0.8)
    stems = set()
    while len(stems) < n:
        k = rng.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        stems.add("".join(rng.choice(common, size=k, p=cp)))
    return sorted(stems)


def make_corpus(lang, n_sentences=30000, n_stems=3000, seed=0):
    """문장 리스트(각 문장은 공백 하나로 이은 단어열)를 돌려준다."""
    rng = np.random.default_rng(seed)
    if lang == "en_like":
        stems = _make_en_stems(rng, n_stems)
    elif lang == "ko_like":
        stems = _make_ko_stems(rng, n_stems)
    else:
        raise ValueError(lang)
    order = rng.permutation(len(stems))  # 빈도 순위를 길이와 무관하게 섞는다
    stems = [stems[i] for i in order]
    sp = _zipf_p(len(stems), 1.0)
    sents = []
    for _ in range(n_sentences):
        m = int(rng.integers(5, 13))
        idx = rng.choice(len(stems), size=m, p=sp)
        words = []
        for i in idx:
            if lang == "en_like":
                pre = EN_PREFIX[rng.choice(len(EN_PREFIX), p=EN_PREFIX_P)]
                suf = EN_SUFFIX[rng.choice(len(EN_SUFFIX), p=EN_SUFFIX_P)]
                words.append(pre + stems[i] + suf)
            else:
                if rng.random() < 0.02:  # 드문 이름: 전체 음절에서 무작위
                    k = int(rng.integers(2, 4))
                    stem = "".join(chr(HANGUL_FIRST + int(c)) for c in rng.integers(0, HANGUL_COUNT, k))
                else:
                    stem = stems[i]
                words.append(stem + KO_PARTICLE[int(rng.integers(len(KO_PARTICLE)))])
        sents.append(" ".join(words))
    return sents


def split(sents, seed, frac=(0.8, 0.1, 0.1)):
    """문장 단위 무작위 분할. 시드가 '데이터 분할 시드'다."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents))
    a = int(frac[0] * len(sents))
    b = a + int(frac[1] * len(sents))
    pick = lambda ii: [sents[i] for i in ii]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])
