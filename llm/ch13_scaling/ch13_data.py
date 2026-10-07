"""13장 합성 말뭉치: 드물게 나오는 사실일수록 늦게 배우는 '열쇠-값' 언어.

문장은 (열쇠 토큰 두 개, 값 토큰 하나) 묶음 11개, 모두 33토큰이다.
- 열쇠는 V*V = 4,096가지. 자주 나오는 순서로 번호를 매기면 k번째 열쇠의 등장 확률은 k^(-ZIPF)에 비례한다
  (지프 분포). 그래서 몇몇 열쇠는 아주 흔하고, 대부분은 드물다.
- 값은 열쇠마다 정해진 토큰이 P_RULE 확률로 나오고, 나머지는 아무 토큰이다.
- 손실은 값 자리에서만 센다. 모델은 열쇠 4,096개의 값을 외워야 하고, 드문 열쇠를 외우려면
  데이터(그 열쇠를 볼 기회)와 파라미터(외울 자리)가 모두 필요하다.

이 설계는 Michaud 등(2023, arXiv:2303.13506v3)의 '양자화 가설'에서 빌렸다. 그들은 지식이 덩어리(quanta)로
나뉘고 자주 쓰는 순서로 배운다면, 사용 빈도의 거듭제곱 분포가 손실의 거듭제곱 감소를 설명한다고 제안했다.
여기서 열쇠 하나가 덩어리 하나다. 이 말뭉치는 자연어가 아니라, 그 가설이 맞는 상황을 일부러 만든 장난감이다.

데이터는 한 번만 본다(같은 문장을 반복하지 않는다). Hoffmann 등(2022)의 D 항이 한 에폭을 가정하기 때문이다.
"""

import math

import numpy as np

V = 64  # 어휘 크기
N_KEYS = V * V
N_CTX = 32  # 모델 입력 길이(정답까지 33토큰)
GROUPS = (N_CTX + 1) // 3  # 문장 하나의 (열쇠, 열쇠, 값) 묶음 수 = 11
ZIPF = 1.5
P_RULE = 0.9
TEACHER_SEED = 2026


def make_teacher(seed=TEACHER_SEED):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, N_KEYS + 1, dtype=float)
    freq = ranks**-ZIPF
    freq /= freq.sum()
    key_of_rank = rng.permutation(N_KEYS)  # 흔한 열쇠가 특정 토큰에 몰리지 않게 섞는다
    p_key = np.empty(N_KEYS)
    p_key[key_of_rank] = freq
    value = rng.integers(0, V, N_KEYS)  # 열쇠마다 정해진 값
    return p_key, value


P_KEY, VALUE = make_teacher()
CUM_KEY = np.cumsum(P_KEY)


def sample_sequences(n_seq, seed):
    """(n_seq, 33) 정수 배열."""
    rng = np.random.default_rng(seed)
    keys = np.minimum(np.searchsorted(CUM_KEY, rng.random((n_seq, GROUPS))), N_KEYS - 1)
    vals = np.where(rng.random((n_seq, GROUPS)) < P_RULE, VALUE[keys], rng.integers(0, V, (n_seq, GROUPS)))
    X = np.stack([keys // V, keys % V, vals], axis=2).reshape(n_seq, 3 * GROUPS)
    return X.astype(np.int64)


CHUNK = 256  # 학습 데이터는 256문장 묶음 단위로 만든다


def training_stream(n_seq, seed):
    """학습용 문장 n_seq개. 묶음마다 시드를 따로 써서, 같은 seed면 앞부분이 n_seq와 무관하게 같다.

    덕분에 '짧게 학습한 모델'과 '길게 학습한 모델의 중간'이 정확히 같은 토큰을 본다.
    """
    chunks = [sample_sequences(CHUNK, seed=seed * 100_003 + k) for k in range(-(-n_seq // CHUNK))]
    return np.concatenate(chunks)[:n_seq]


def scored_positions():
    """입력 위치 j가 맞히는 토큰은 X[j+1]. 값 자리(인덱스 2, 5, 8, ...)를 맞히는 j만 센다."""
    return np.arange(1, N_CTX, 3)


def entropy_floor():
    """값 자리의 이론 최저 손실(nat). 열쇠를 완벽히 알아도 값의 10%는 아무 토큰이라 남는 엔트로피다."""
    p_hit = P_RULE + (1 - P_RULE) / V
    p_miss = (1 - P_RULE) / V
    floor = -(p_hit * math.log(p_hit) + (V - 1) * p_miss * math.log(p_miss))
    return {"floor": floor, "uniform": math.log(V)}


def key_coverage(n_tokens):
    """n_tokens개 토큰(입력 기준)을 봤을 때 값 자리 하나의 열쇠가 학습 중 한 번 이상 나왔을 확률.

    입력 32토큰에 묶음이 대략 32/3개 들어 있으니 본 열쇠 수 m ≈ n_tokens/3. 열쇠 k를 한 번도 못 볼 확률은
    (1 - p_k)^m. 이걸 p_k로 가중평균하면 '처음 보는 열쇠를 물어볼 확률'이 된다.
    """
    m = n_tokens / 3
    unseen = float(np.sum(P_KEY * (1 - P_KEY) ** m))
    return {"seen_prob": 1 - unseen, "unseen_prob": unseen}
