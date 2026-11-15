"""11장 NumPy 최소 구현: 디코딩(다음 토큰 고르기) 방법과 반복 지표.

model은 `model(ids) -> logits` 형태의 함수다. ids는 (B, L) 정수 배열, 반환은 (B, V) 로짓
(각 행의 마지막 위치 다음 토큰에 대한 점수). 모델 내부가 무엇이든 디코딩 코드는 같다.

- greedy / beam_search: 확률이 가장 높은 열을 찾는 방법(최대화 기반)
- apply_temperature / top_k_filter / top_p_filter / sample_from_logits: 분포를 다듬은 뒤 뽑는 방법
- repetition_stats: 생성문 안의 4-gram 중복 비율과 distinct-2

입력: 로짓 (B, V)(B = 배치, V = 어휘 크기)와 온도 T, top-k의 k, top-p의 p. 출력: 필터링된 로짓, 뽑힌 토큰 id (B,), 생성 열 (B, L0 + n).
필터를 거친 로짓에서 -inf는 '후보에서 제외'를 뜻하고, softmax를 하면 그 칸의 확률이 정확히 0이 된다.
장 본문 숫자와의 대응은 experiments.py 머리말과 test_ch11.py의 손계산 테스트 참고.
"""

import numpy as np


def log_softmax(z, axis=-1):
    # log-sum-exp 안정화: 최댓값을 빼도 log softmax는 같다. 식: log p_i = z_i - log sum_j exp(z_j)
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(z, axis=-1):
    return np.exp(log_softmax(z, axis))


# ---------------------------------------------------------------- 분포 다듬기


def apply_temperature(logits, T):
    """P_T(w) ∝ exp(z_w / T). T=0은 argmax 한 칸만 남기는 극한으로 처리한다."""
    logits = np.asarray(logits, dtype=float)
    if T == 0:
        # T = 0은 1/T가 무한대인 극한. argmax 칸만 0, 나머지 -inf로 두면 softmax가 원-핫이 된다
        out = np.full_like(logits, -np.inf)
        idx = logits.argmax(-1)
        np.put_along_axis(out, idx[..., None], 0.0, axis=-1)
        return out
    # 식: P_T(w) = exp(z_w/T) / sum_v exp(z_v/T). T < 1이면 뾰족해지고 T > 1이면 평평해진다
    return logits / T


def top_k_filter(logits, k):
    """점수 상위 k개만 남기고 나머지를 -inf로. 동점이 없으면 남는 칸은 정확히 k개."""
    logits = np.asarray(logits, dtype=float)
    if k is None or k >= logits.shape[-1]:
        return logits.copy()
    # 오름차순 정렬의 뒤에서 k번째 = k번째로 큰 값, (B, 1)
    kth = np.sort(logits, axis=-1)[..., -k][..., None]  # k번째로 큰 값
    # 주의: 동점이 있으면 k개를 넘겨 남을 수 있다(>= 비교)
    return np.where(logits >= kth, logits, -np.inf)


def top_p_filter(logits, p):
    """Holtzman 등(2020) 식 (2): 누적확률이 p 이상이 되는 가장 작은 집합 V^(p)만 남긴다.

    확률 높은 순으로 더해 가다가, p를 처음 넘기게 만든 토큰까지 포함한다.
    """
    logits = np.asarray(logits, dtype=float)
    if p is None or p >= 1.0:
        return logits.copy()
    # top-p는 확률 기준이라 softmax를 먼저 한다. 온도 처리 후 로짓이 들어오는 것이 정상 순서(shape_logits)
    probs = softmax(logits)
    # 각 행의 확률 내림차순 인덱스 (B, V)
    order = np.argsort(-probs, axis=-1)
    sorted_p = np.take_along_axis(probs, order, axis=-1)
    cum = np.cumsum(sorted_p, axis=-1)
    # 식: 누적 확률(자기 포함) - 자기 확률 = 자기 '앞' 누적이 p 미만이면 남긴다. 이래야 p를 처음 넘기게 만든 토큰이 포함된다
    keep_sorted = (cum - sorted_p) < p  # 이 토큰을 넣기 '전' 누적이 p 미만이면 남긴다
    # 정렬 순서로 계산한 keep_sorted를 원래 토큰 번호 순서로 되돌린다(put_along_axis)
    keep = np.zeros_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)
    return np.where(keep, logits, -np.inf)


def shape_logits(logits, T=1.0, k=None, p=None):
    """온도를 먼저 적용하고 그다음 자른다(Holtzman 등 §3.3: 온도로 모양을 잡은 뒤 top-k)."""
    # 순서가 결과를 바꾼다: 온도 -> top-k -> top-p. (test_temperature_order_changes_nucleus가 확인)
    z = apply_temperature(logits, T)
    z = top_k_filter(z, k)
    return top_p_filter(z, p)


def sample_from_logits(logits, rng, T=1.0, k=None, p=None):
    """다듬은 분포에서 행마다 토큰 하나를 뽑는다. 반환 (B,)."""
    # shape_logits의 -inf 칸은 확률 0
    probs = softmax(shape_logits(np.atleast_2d(logits), T, k, p))
    # 누적분포로 역변환 표집(inverse CDF)
    cdf = probs.cumsum(-1)
    # 균등 난수 u를 cdf의 끝값(약 1)에 곱해 합이 1에서 조금 어긋나도 괜찮게 한다
    u = rng.random((probs.shape[0], 1)) * cdf[:, -1:]  # 반올림 오차로 합이 1보다 조금 작아도 확률 0인 칸은 뽑히지 않게
    # cdf가 u보다 큰 첫 칸의 인덱스. 확률 0 칸은 cdf가 앞 칸과 같아 선택될 수 없다
    return (cdf <= u).sum(-1)


# ---------------------------------------------------------------- 생성


def greedy(model, prompt, n):
    """w_t = argmax_w P(w | w_<t). prompt: (B, L0). 반환 (B, L0 + n)."""
    ids = np.atleast_2d(np.asarray(prompt))
    for _ in range(n):
        # model(ids): (B, V) 로짓. 마지막 위치만 쓴다. 매 스텝 전체 열을 다시 넣는 단순한 구현(KV 캐시는 12장)
        nxt = model(ids).argmax(-1)
        ids = np.concatenate([ids, nxt[:, None]], axis=1)
    return ids


def sample(model, prompt, n, rng, T=1.0, k=None, p=None):
    ids = np.atleast_2d(np.asarray(prompt))
    for _ in range(n):
        nxt = sample_from_logits(model(ids), rng, T, k, p)
        ids = np.concatenate([ids, nxt[:, None]], axis=1)
    return ids


def beam_search(model, prompt, n, beam):
    """sum_t log P(w_t | w_<t)를 최대화하는 열을 빔 크기만큼만 들고 찾는 근사.

    prompt: (L0,) 하나. 길이가 모두 n으로 같으니 길이 정규화는 하지 않는다.
    반환: (가장 높은 점수의 열 (L0 + n,), 그 열의 누적 로그확률).
    """
    # 빔 후보 열들 (b, L). 처음엔 1개
    seqs = np.asarray(prompt)[None, :]
    # 각 빔의 누적 로그확률 (b,)
    scores = np.zeros(1)
    for _ in range(n):
        # (b, V): 빔마다 다음 토큰 로그확률
        lp = log_softmax(model(seqs))  # (b, V)
        # (b, 1) + (b, V) = (b, V): 후보 열의 누적 점수. 식: sum_t log P(w_t | w_<t)
        total = scores[:, None] + lp
        V = total.shape[1]
        # (빔, 토큰) 쌍 전체를 한 줄로 펴서 상위 beam개를 고른다
        flat = total.ravel()
        # stable 정렬이라 동점일 때 결과가 실행마다 같다
        top = np.argsort(-flat, kind="stable")[:beam]  # 동점이면 앞 빔·작은 토큰 번호 먼저
        # 펴진 인덱스를 (어느 빔에서 왔나, 어떤 토큰)으로 되돌린다
        rows, toks = top // V, top % V
        seqs = np.concatenate([seqs[rows], toks[:, None]], axis=1)
        scores = flat[top]
    best = int(np.argmax(scores))
    return seqs[best], float(scores[best])


def sequence_logprob(model, ids, start):
    """ids[:, start:]의 각 토큰을 원래 분포(T=1, 자르지 않음)로 잰 로그확률. 반환 (B, L - start)."""
    ids = np.atleast_2d(ids)
    out = []
    for t in range(start, ids.shape[1]):
        # 위치 t의 토큰은 앞 t개만 보고 예측한 분포로 잰다(off-by-one 주의)
        lp = log_softmax(model(ids[:, :t]))
        # 각 행의 실제 토큰 ids[:, t]에 해당하는 로그확률 (B,)
        out.append(lp[np.arange(len(ids)), ids[:, t]])
    return np.stack(out, axis=1)


# ---------------------------------------------------------------- 지표


def ngrams(tokens, n):
    tokens = list(tokens)
    # 겹치는 n-gram 슬라이딩 창. 토큰이 n개보다 적으면 빈 목록
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repetition_stats(tokens):
    """rep4: 4-gram 가운데 앞에 이미 나온 것의 비율(1 - 고유/전체). distinct2: 고유 2-gram/전체 2-gram."""
    # 주의: 토큰이 4개 미만이면 g4가 비어 아래에서 0으로 나누기 오류
    g4, g2 = ngrams(tokens, 4), ngrams(tokens, 2)
    return {
        # 이미 나온 4-gram의 비율. 높을수록 반복
        "rep4": 1.0 - len(set(g4)) / len(g4),
        "distinct2": len(set(g2)) / len(g2),
    }


# ---------------------------------------------------------------- 실험용 정답 언어(합성)


class SecondOrderSource:
    """앞 두 토큰 (a, b)가 다음 토큰의 분포를 정하는 합성 언어. 정답 확률을 정확히 안다.

    상태마다 허용되는 다음 토큰은 일부뿐이고(support), 나머지의 정답 확률은 0이다.
    support 크기가 상태마다 달라서 어떤 문맥은 후보가 2~3개로 뾰족하고 어떤 문맥은 20개 넘게 평평하다
    (Holtzman 등 Figure 5가 말한 '평평한 분포와 뾰족한 분포'를 작게 흉내 낸 것).
    """

    def __init__(self, V=64, seed=0, min_support=2, max_support=24, alpha=0.8):
        # 시드 고정: 같은 시드면 같은 합성 언어
        rng = np.random.default_rng(seed)
        self.V = V
        # P[a, b, c] = 앞 두 토큰이 (a, b)일 때 다음이 c일 정답 확률. (V, V, V)
        P = np.zeros((V, V, V))
        for a in range(V):
            for b in range(V):
                # 상태마다 support 크기를 min..max에서 무작위로 정한다(뾰족한/평평한 상태가 섞인다)
                s = rng.integers(min_support, max_support + 1)
                sup = rng.choice(V, size=s, replace=False)
                # 디리클레(alpha 0.8, 1보다 작음)는 몇몇 토큰에 확률이 몰리는 분포를 만든다. 합은 1
                P[a, b, sup] = rng.dirichlet(np.full(s, alpha))
        self.P = P

    def sample(self, n_seq, length, rng, start=None):
        X = np.empty((n_seq, length), dtype=np.int64)
        if start is None:
            # 처음 두 토큰은 균등 무작위(정상 분포가 아님: 짧은 앞부분은 엔트로피 계산에서 영향이 작다고 보는 단순화)
            X[:, :2] = rng.integers(0, self.V, (n_seq, 2))
            t0 = 2
        else:
            start = np.atleast_2d(start)
            X[:, : start.shape[1]] = start
            t0 = start.shape[1]
        for t in range(t0, length):
            # (n_seq, V): 상태 (X[t-2], X[t-1])의 확률을 누적. 아래에서 역변환 표집
            cdf = self.P[X[:, t - 2], X[:, t - 1]].cumsum(-1)
            X[:, t] = (cdf <= rng.random((n_seq, 1)) * cdf[:, -1:]).sum(-1)
        return X

    def token_probs(self, ids, start):
        """ids[:, start:]의 각 토큰에 대한 정답 확률. 0이면 이 언어에서 나올 수 없는 토큰이다."""
        ids = np.atleast_2d(ids)
        # 정답 확률 P[a, b, c]. 반환 (n_seq, 길이 - start)
        return np.stack([self.P[ids[:, t - 2], ids[:, t - 1], ids[:, t]] for t in range(start, ids.shape[1])], 1)

    def entropy_rate(self, n=20000, seed=1):
        """정상 상태에서의 토큰당 엔트로피(nat). 긴 표본에서 평균 낸다."""
        # 긴 표본 하나(20000토큰)를 만들어 방문한 상태들의 엔트로피를 평균한다
        X = self.sample(1, n, np.random.default_rng(seed))[0]
        # 각 위치의 상태 (a, b)에 대한 다음 토큰 분포 (n-2, V)
        p = self.P[X[:-2], X[1:-1]]
        # 식: H = -sum p log p. p = 0인 칸은 0으로 취급해 log 0을 피한다
        h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(-1)
        return float(h.mean())
