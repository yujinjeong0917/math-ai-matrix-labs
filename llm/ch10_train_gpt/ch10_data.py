"""10장 데이터: 작은 한국어 문법 말뭉치.

라이선스 걱정이 없도록 직접 만든 확률 문법에서 문장을 뽑는다. 단어 하나가 토큰 하나다.

    문장 = [시간 부사] 주어 목적어 [방식 부사] 동사 "."

- 시간 부사(어제/오늘/아침에)는 절반 확률로 붙는다.
- 주어가 동물이면 책·편지·신문을 목적어로 쓰지 않는다(사람만 읽고 쓴다).
- 동사는 목적어의 종류가 정한다(음식은 먹었다/샀다/좋아했다, 장소는 갔다/왔다 ...).
- 방식 부사(천천히/빨리)는 30% 확률로 동사 앞에 붙는다.

생성 규칙을 알고 있으니, 매 위치에서 "정답 분포의 엔트로피"를 정확히 계산할 수 있다.
그 평균이 어떤 모델도 넘을 수 없는 손실의 바닥이다(문맥을 다 안다고 할 때).
"""

import math

import numpy as np

TIME_ADV = ["어제", "오늘", "아침에"]
ANIMALS = ["고양이가", "강아지가", "토끼가", "참새가"]
PEOPLE = ["민수가", "지영이가", "선생님이", "아이가"]
SUBJECTS = ANIMALS + PEOPLE
OBJECTS = {
    "food": ["생선을", "사과를", "당근을", "빵을"],
    "text": ["책을", "편지를", "신문을"],
    "ball": ["공을", "돌을"],
    "place": ["공원에", "학교에", "시장에"],
}
VERBS = {
    "food": ["먹었다", "샀다", "좋아했다"],
    "text": ["읽었다", "샀다", "썼다"],
    "ball": ["찼다", "던졌다"],
    "place": ["갔다", "왔다"],
}
MANNER = ["천천히", "빨리"]
END = "."
P_TIME, P_MANNER = 0.5, 0.3

_words = TIME_ADV + SUBJECTS + [w for c in OBJECTS.values() for w in c] + MANNER
_words += [v for c in VERBS.values() for v in c] + [END]
VOCAB = list(dict.fromkeys(_words))  # 순서를 지키며 중복(샀다) 제거
STOI = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
OBJ_CAT = {w: c for c, ws in OBJECTS.items() for w in ws}


def _allowed_objects(subject):
    cats = ["food", "ball", "place"] if subject in ANIMALS else ["food", "text", "ball", "place"]
    return [w for c in cats for w in OBJECTS[c]]


def _entropy(p):
    p = np.asarray([x for x in p if x > 0])
    return float(-(p * np.log(p)).sum())


# 위치별 정답 분포의 엔트로피(자연로그). 생성할 때 같은 규칙으로 함께 기록한다.
H_FIRST = _entropy([P_TIME / 3] * 3 + [(1 - P_TIME) / 8] * 8)  # 문장 첫 토큰: 부사 또는 주어
H_SUBJ = math.log(8)  # 부사 다음 주어


def sample_sentence(rng):
    """문장 하나와, 각 토큰을 맞힐 때의 정답 분포 엔트로피를 함께 돌려준다."""
    toks, ents = [], []
    if rng.random() < P_TIME:
        toks.append(TIME_ADV[rng.integers(3)])
        ents.append(H_FIRST)
        subj = SUBJECTS[rng.integers(8)]
        toks.append(subj)
        ents.append(H_SUBJ)
    else:
        subj = SUBJECTS[rng.integers(8)]
        toks.append(subj)
        ents.append(H_FIRST)
    objs = _allowed_objects(subj)
    obj = objs[rng.integers(len(objs))]
    toks.append(obj)
    ents.append(math.log(len(objs)))
    verbs = VERBS[OBJ_CAT[obj]]
    h_before_verb = _entropy([P_MANNER / 2] * 2 + [(1 - P_MANNER) / len(verbs)] * len(verbs))
    if rng.random() < P_MANNER:
        toks.append(MANNER[rng.integers(2)])
        ents.append(h_before_verb)
        toks.append(verbs[rng.integers(len(verbs))])
        ents.append(math.log(len(verbs)))
    else:
        toks.append(verbs[rng.integers(len(verbs))])
        ents.append(h_before_verb)
    toks.append(END)
    ents.append(0.0)  # 동사 다음은 항상 마침표
    return toks, ents


def make_windows(n_windows, n_ctx, seed):
    """문장 경계에서 시작하는 길이 n_ctx+1 창을 만든다.

    반환: X (n_windows, n_ctx+1) 정수 배열, floor (창 안 예측 n_ctx개의 평균 엔트로피를 다시 평균 낸 값).
    창의 첫 토큰은 예측하지 않으므로, 그 엔트로피는 바닥 계산에서 뺀다.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_windows, n_ctx + 1), dtype=np.int64)
    floors = np.empty(n_windows)
    for i in range(n_windows):
        toks, ents = [], []
        while len(toks) < n_ctx + 1:
            t, e = sample_sentence(rng)
            toks += t
            ents += e
        X[i] = [STOI[w] for w in toks[: n_ctx + 1]]
        floors[i] = float(np.mean(ents[1 : n_ctx + 1]))
    return X, float(floors.mean())


def is_valid_sentence(words):
    """문법을 지킨 문장인지 판정한다(마침표 제외한 단어 목록)."""
    w = list(words)
    if w and w[0] in TIME_ADV:
        w = w[1:]
    if len(w) not in (3, 4) or w[0] not in SUBJECTS:
        return False
    if w[1] not in _allowed_objects(w[0]):
        return False
    if len(w) == 4:
        if w[2] not in MANNER:
            return False
        w = [w[0], w[1], w[3]]
    return w[2] in VERBS[OBJ_CAT[w[1]]]


def split_sentences(ids):
    """토큰 id 열을 마침표 기준 문장 목록으로 자른다(마지막 미완성 문장은 버린다)."""
    out, cur = [], []
    for i in ids:
        if VOCAB[i] == END:
            out.append(tuple(VOCAB[j] for j in cur))
            cur = []
        else:
            cur.append(i)
    return out
