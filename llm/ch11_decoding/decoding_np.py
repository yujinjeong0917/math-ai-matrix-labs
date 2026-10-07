"""11장 NumPy 최소 구현: 디코딩(다음 토큰 고르기) 방법과 반복 지표.

model은 `model(ids) -> logits` 형태의 함수다. ids는 (B, L) 정수 배열, 반환은 (B, V) 로짓
(각 행의 마지막 위치 다음 토큰에 대한 점수). 모델 내부가 무엇이든 디코딩 코드는 같다.

- greedy / beam_search: 확률이 가장 높은 열을 찾는 방법(최대화 기반)
- apply_temperature / top_k_filter / top_p_filter / sample_from_logits: 분포를 다듬은 뒤 뽑는 방법
- repetition_stats: 생성문 안의 4-gram 중복 비율과 distinct-2
"""

import numpy as np


def log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)  # log-sum-exp 안정화
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(z, axis=-1):
    return np.exp(log_softmax(z, axis))


# ---------------------------------------------------------------- 분포 다듬기


def apply_temperature(logits, T):
    """P_T(w) ∝ exp(z_w / T). T=0은 argmax 한 칸만 남기는 극한으로 처리한다."""
    logits = np.asarray(logits, dtype=float)
    if T == 0:
        out = np.full_like(logits, -np.inf)
        idx = logits.argmax(-1)
        np.put_along_axis(out, idx[..., None], 0.0, axis=-1)
        return out
    return logits / T


def top_k_filter(logits, k):
    """점수 상위 k개만 남기고 나머지를 -inf로. 동점이 없으면 남는 칸은 정확히 k개."""
    logits = np.asarray(logits, dtype=float)
    if k is None or k >= logits.shape[-1]:
        return logits.copy()
    kth = np.sort(logits, axis=-1)[..., -k][..., None]  # k번째로 큰 값
    return np.where(logits >= kth, logits, -np.inf)


def top_p_filter(logits, p):
    """Holtzman 등(2020) 식 (2): 누적확률이 p 이상이 되는 가장 작은 집합 V^(p)만 남긴다.

    확률 높은 순으로 더해 가다가, p를 처음 넘기게 만든 토큰까지 포함한다.
    """
    logits = np.asarray(logits, dtype=float)
    if p is None or p >= 1.0:
        return logits.copy()
    probs = softmax(logits)
    order = np.argsort(-probs, axis=-1)
    sorted_p = np.take_along_axis(probs, order, axis=-1)
    cum = np.cumsum(sorted_p, axis=-1)
    keep_sorted = (cum - sorted_p) < p  # 이 토큰을 넣기 '전' 누적이 p 미만이면 남긴다
    keep = np.zeros_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)
    return np.where(keep, logits, -np.inf)


def shape_logits(logits, T=1.0, k=None, p=None):
    """온도를 먼저 적용하고 그다음 자른다(Holtzman 등 §3.3: 온도로 모양을 잡은 뒤 top-k)."""
    z = apply_temperature(logits, T)
    z = top_k_filter(z, k)
    return top_p_filter(z, p)


def sample_from_logits(logits, rng, T=1.0, k=None, p=None):
    """다듬은 분포에서 행마다 토큰 하나를 뽑는다. 반환 (B,)."""
    probs = softmax(shape_logits(np.atleast_2d(logits), T, k, p))
    cdf = probs.cumsum(-1)
    u = rng.random((probs.shape[0], 1)) * cdf[:, -1:]  # 반올림 오차로 합이 1보다 조금 작아도 확률 0인 칸은 뽑히지 않게
    return (cdf <= u).sum(-1)


# ---------------------------------------------------------------- 생성


def greedy(model, prompt, n):
    """w_t = argmax_w P(w | w_<t). prompt: (B, L0). 반환 (B, L0 + n)."""
    ids = np.atleast_2d(np.asarray(prompt))
    for _ in range(n):
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
    seqs = np.asarray(prompt)[None, :]
    scores = np.zeros(1)
    for _ in range(n):
        lp = log_softmax(model(seqs))  # (b, V)
        total = scores[:, None] + lp
        V = total.shape[1]
        flat = total.ravel()
        top = np.argsort(-flat, kind="stable")[:beam]  # 동점이면 앞 빔·작은 토큰 번호 먼저
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
        lp = log_softmax(model(ids[:, :t]))
        out.append(lp[np.arange(len(ids)), ids[:, t]])
    return np.stack(out, axis=1)


# ---------------------------------------------------------------- 지표


def ngrams(tokens, n):
    tokens = list(tokens)
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def repetition_stats(tokens):
    """rep4: 4-gram 가운데 앞에 이미 나온 것의 비율(1 - 고유/전체). distinct2: 고유 2-gram/전체 2-gram."""
    g4, g2 = ngrams(tokens, 4), ngrams(tokens, 2)
    return {
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
        rng = np.random.default_rng(seed)
        self.V = V
        P = np.zeros((V, V, V))
        for a in range(V):
            for b in range(V):
                s = rng.integers(min_support, max_support + 1)
                sup = rng.choice(V, size=s, replace=False)
                P[a, b, sup] = rng.dirichlet(np.full(s, alpha))
        self.P = P

    def sample(self, n_seq, length, rng, start=None):
        X = np.empty((n_seq, length), dtype=np.int64)
        if start is None:
            X[:, :2] = rng.integers(0, self.V, (n_seq, 2))
            t0 = 2
        else:
            start = np.atleast_2d(start)
            X[:, : start.shape[1]] = start
            t0 = start.shape[1]
        for t in range(t0, length):
            cdf = self.P[X[:, t - 2], X[:, t - 1]].cumsum(-1)
            X[:, t] = (cdf <= rng.random((n_seq, 1)) * cdf[:, -1:]).sum(-1)
        return X

    def token_probs(self, ids, start):
        """ids[:, start:]의 각 토큰에 대한 정답 확률. 0이면 이 언어에서 나올 수 없는 토큰이다."""
        ids = np.atleast_2d(ids)
        return np.stack([self.P[ids[:, t - 2], ids[:, t - 1], ids[:, t]] for t in range(start, ids.shape[1])], 1)

    def entropy_rate(self, n=20000, seed=1):
        """정상 상태에서의 토큰당 엔트로피(nat). 긴 표본에서 평균 낸다."""
        X = self.sample(1, n, np.random.default_rng(seed))[0]
        p = self.P[X[:-2], X[1:-1]]
        h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(-1)
        return float(h.mean())
