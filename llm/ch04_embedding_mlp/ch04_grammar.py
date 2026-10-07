"""4장 합성 문법. 일반화 효과가 크게 드러나도록 일부러 설계한 작은 말뭉치다.

문장 꼴: <s> <s> [때] [주어] [동사] </s>
- 동물 주어 4개는 동물 동사 4개와, 사람 주어 4개는 사람 동사 4개와만 어울린다.
- (주어, 동사) 32쌍 가운데 4쌍은 학습 말뭉치에서 뺀다. 예: "강아지가 잔다".
  빠진 주어도 나머지 동사 3개와는 똑같이 쓰이고, 빠진 동사도 다른 주어 3개와는 쓰인다.
그래서 "강아지가 잔다"는 학습 데이터에 없지만, "강아지가"와 "고양이가"가 같은 자리에서
같은 동사들과 쓰인다는 정보는 학습 데이터에 있다. 이 정보를 n-gram은 쓰지 못하고,
공유 임베딩은 쓸 수 있는지를 재는 것이 이 장의 실험이다.
"""

import numpy as np

BOS, EOS = "<s>", "</s>"
TIMES = ["오늘", "어제", "지금"]
ANIMALS = ["고양이가", "강아지가", "토끼가", "여우가"]
ANIMAL_VERBS = ["잔다", "먹는다", "뛴다", "논다"]
PEOPLE = ["아이가", "학생이", "엄마가", "아빠가"]
PEOPLE_VERBS = ["읽는다", "쓴다", "웃는다", "말한다"]
HELD_OUT = [("강아지가", "잔다"), ("토끼가", "먹는다"), ("학생이", "읽는다"), ("아빠가", "웃는다")]

VOCAB = [BOS, EOS] + TIMES + ANIMALS + ANIMAL_VERBS + PEOPLE + PEOPLE_VERBS
IDX = {w: i for i, w in enumerate(VOCAB)}
CONTEXT = 2  # 앞 두 단어를 본다(트라이그램, Bengio 표기로 n = 3)


def allowed_pairs():
    pairs = [(s, v) for s in ANIMALS for v in ANIMAL_VERBS] + [(s, v) for s in PEOPLE for v in PEOPLE_VERBS]
    return [p for p in pairs if p not in HELD_OUT]


def sentence(t, s, v):
    return [BOS] * CONTEXT + [t, s, v, EOS]


def sample_sentences(k, rng, pairs=None):
    pairs = allowed_pairs() if pairs is None else pairs
    out = []
    for _ in range(k):
        s, v = pairs[rng.integers(len(pairs))]
        out.append(sentence(TIMES[rng.integers(len(TIMES))], s, v))
    return out


def unseen_sentences():
    """빠진 4쌍 × 때 3개 = 12문장. 학습 말뭉치에 한 번도 나오지 않는다."""
    return [sentence(t, s, v) for (s, v) in HELD_OUT for t in TIMES]


def to_examples(sentences):
    """(문맥 id 배열 (N, 2), 다음 단어 id (N,))로 바꾼다. 문장당 예측 4번."""
    ctx, tgt = [], []
    for sent in sentences:
        ids = [IDX[w] for w in sent]
        for i in range(CONTEXT, len(ids)):
            ctx.append(ids[i - CONTEXT : i])
            tgt.append(ids[i])
    return np.array(ctx, dtype=np.int64), np.array(tgt, dtype=np.int64)


def make_splits(seed=0, n_train=600, n_valid=150, n_test=300):
    rng = np.random.default_rng(seed)
    return {
        "train": sample_sentences(n_train, rng),
        "valid": sample_sentences(n_valid, rng),
        "test_seen": sample_sentences(n_test, rng),
        "test_unseen": unseen_sentences(),
    }


def true_ppl():
    """학습 분포(빠진 쌍 제외)를 정확히 아는 모델의 문장 위치당 PPL. 어떤 모델도 기대값으로는 이보다 낮을 수 없다."""
    pairs = allowed_pairs()
    n_verbs = {s: sum(1 for p in pairs if p[0] == s) for s, _ in pairs}
    H = np.log(len(TIMES))  # 때
    H += -sum((c / len(pairs)) * np.log(c / len(pairs)) for c in n_verbs.values())  # 주어
    H += sum((c / len(pairs)) * np.log(c) for c in n_verbs.values())  # 주어가 정해지면 동사는 균등
    return float(np.exp(H / 4))  # 예측 4번(때, 주어, 동사, </s>) 평균. </s>는 확률 1
