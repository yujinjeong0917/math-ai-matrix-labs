"""모델 평가 9장. 클러스터링 평가 지표의 NumPy 최소 구현.

라벨이 있을 때(외부 평가)
  contingency, match_labels_hungarian(Kuhn 1955의 할당 문제), matched_accuracy,
  rand_index, ari, entropy, mutual_info, nmi, expected_mutual_info, ami
라벨이 없을 때(내부 평가)
  silhouette_samples, silhouette (Rousseeuw 1987의 s(i) = (b - a) / max(a, b))
비교 실험용
  kmeans (k-means++ 시작 + Lloyd 반복), greedy_majority_accuracy(흔한 실수)

정보량은 자연로그(nat)로 잰다. NMI·AMI의 정규화 방식 이름은 Vinh·Epps·Bailey(2010) 표 2를 따른다.
  max:        I / max{H(U), H(V)}
  arithmetic: 2I / (H(U) + H(V))         (Vinh의 NMI_sum)
  geometric:  I / sqrt(H(U) H(V))        (Vinh의 NMI_sqrt)
  min:        I / min{H(U), H(V)}
기대 상호정보량 E{I}는 Vinh 식 (2)(순열 모형: 두 분할의 군집 크기를 고정하고 점만 섞는다).
다른 장과 코드를 공유하지 않는다(장끼리 동시에 작성되므로).
"""

from itertools import permutations
from math import comb

import numpy as np


# ---------------------------------------------------------------- 대응표


def _codes(labels):
    """라벨 값이 무엇이든 0..k-1 정수로 바꾼다. 번호는 이름표라 값 자체는 쓰지 않는다."""
    # np.unique의 역인덱스 = 값을 0..k-1로 다시 매긴 번호
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.ravel()


def contingency(u, v):
    """대응표 n[i, j] = U의 i번 군집이면서 V의 j번 군집인 점의 수."""
    a, b = _codes(u), _codes(v)
    if len(a) != len(b):
        raise ValueError("두 라벨의 길이가 달라요")
    # (행 = U 군집 수, 열 = V 군집 수)
    t = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    # 같은 (i, j)가 여러 번 나와도 모두 더해지는 비버퍼 덧셈(t[a, b] += 1은 중복을 한 번만 셈)
    np.add.at(t, (a, b), 1)
    return t


# ---------------------------------------------------------------- 번호 맞추기


def hungarian_max(w):
    """가중치 행렬 w(r x c)에서 행과 열을 하나씩 짝지어 합을 최대로 만드는 짝.

    헝가리안 방법(Kuhn 1955)을 퍼텐셜(이중 변수)로 쓰는 흔한 O(n^3) 구현. 직사각형이면 0으로 채워 정사각형으로 만든다.
    반환: (행 번호 배열, 열 번호 배열) - 원래 크기 안에 있는 짝만.
    """
    w = np.asarray(w, dtype=float)
    r, c = w.shape
    n = max(r, c)
    cost = np.zeros((n, n))
    cost[:r, :c] = -w  # 최대화 = 부호를 바꾼 최소화
    # 아래는 퍼텐셜 방식 헝가리안(1-인덱스 배열). u, v는 행·열 퍼텐셜, p[j]는 열 j에 짝지어진 행, way는 증가 경로를 거슬러 가는 용도
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)  # p[j] = 열 j에 짝지어진 행(1부터, 0은 비어 있음)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, n + 1):
        i = p[j] - 1
        # 채워 넣은 가짜 행·열과 짝지어진 것은 결과에서 제외
        if i < r and j - 1 < c:
            rows.append(i)
            cols.append(j - 1)
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(cols)[order]


def brute_force_max(w):
    """대조용: 모든 순열을 다 본다(작은 행렬만). 최대 합을 돌려준다."""
    w = np.asarray(w, dtype=float)
    r, c = w.shape
    n = max(r, c)
    sq = np.zeros((n, n))
    sq[:r, :c] = w
    # 8! = 40320 순열이라 8 x 8까지만 허용(전수 대조용)
    if n > 8:
        raise ValueError("전수 순열은 8 x 8까지만 해요")
    return max(sum(sq[i, perm[i]] for i in range(n)) for perm in permutations(range(n)))


def match_labels_hungarian(y_true, y_pred):
    """군집 번호 -> 정답 번호 짝(dict)과 맞힌 점 수. 짝이 없는 군집은 어느 정답과도 맞지 않은 것으로 센다."""
    t_vals = np.unique(np.asarray(y_true))
    p_vals = np.unique(np.asarray(y_pred))
    # 정답 i번 모둠 안에 군집 j번 점이 몇 개인지. 짝지을 때 이 개수를 최대로 한다
    table = contingency(y_true, y_pred)  # 행 = 정답, 열 = 군집
    rows, cols = hungarian_max(table)
    # contingency의 행·열 번호(0..)를 원래 라벨 값으로 되돌린다. 짝이 없는 군집은 dict에 없다
    mapping = {p_vals[j].item(): t_vals[i].item() for i, j in zip(rows, cols)}
    return mapping, int(table[rows, cols].sum())


def raw_accuracy(y_true, y_pred):
    """번호를 그대로 비교한 정확도(분류처럼). 군집 평가에서는 쓰면 안 되는 값."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def matched_accuracy(y_true, y_pred):
    """헝가리안으로 번호를 가장 잘 바꿔 읽은 뒤의 정확도."""
    _, hit = match_labels_hungarian(y_true, y_pred)
    return hit / len(np.asarray(y_true))


def greedy_majority_accuracy(y_true, y_pred):
    """흔한 실수: 군집마다 따로 가장 많은 정답으로 읽기(일대일이 아님). 잘게 쪼갤수록 높아진다."""
    t = contingency(y_true, y_pred)
    # 군집(열)마다 최댓값만 취하는 일대일이 아닌 읽기. 군집을 점 하나씩으로 쪼개면 1.0이 된다
    return float(t.max(axis=0).sum() / t.sum())


# ---------------------------------------------------------------- 쌍 세기: RI, ARI


def pair_counts(u, v):
    """점 쌍을 네 가지로 센다. (둘 다 같이, U만 같이, V만 같이, 둘 다 따로)."""
    t = contingency(u, v)
    n = int(t.sum())
    total = comb(n, 2)
    # 같은 U 군집이면서 같은 V 군집인 점 쌍의 수 = 칸마다 C(n_ij, 2)의 합
    same_both = sum(comb(int(x), 2) for x in t.ravel())
    same_u = sum(comb(int(x), 2) for x in t.sum(axis=1))
    same_v = sum(comb(int(x), 2) for x in t.sum(axis=0))
    # (n11, n10, n01, n00). n00은 포함-배제: 전체 - U에서 같은 쌍 - V에서 같은 쌍 + 둘 다 같은 쌍
    return same_both, same_u - same_both, same_v - same_both, total - same_u - same_v + same_both


def rand_index(u, v):
    n11, n10, n01, n00 = pair_counts(u, v)
    return (n11 + n00) / (n11 + n10 + n01 + n00)


def expected_rand_index(u, v):
    """순열 모형에서 RI의 기댓값. 한 쌍이 V에서 같이 묶일 확률은 same_v / total(군집 크기 고정)."""
    t = contingency(u, v)
    total = comb(int(t.sum()), 2)
    # 한 쌍이 U에서 같이 묶일 확률. 순열 모형에서 V의 배치만 섞이므로 V에서 같이 묶일 확률은 pv
    pu = sum(comb(int(x), 2) for x in t.sum(axis=1)) / total
    pv = sum(comb(int(x), 2) for x in t.sum(axis=0)) / total
    # 한 쌍이 두 분할에서 일치(둘 다 같이 또는 둘 다 따로)할 확률 = E[RI]
    return pu * pv + (1 - pu) * (1 - pv)


def ari(u, v):
    """보정 랜드 지수 (RI - E[RI]) / (1 - E[RI]). 쌍 세기로 쓴 흔한 꼴과 같은 값이다.

    두 분할 모두 군집이 하나뿐이거나 모두 한 점씩이면 분모가 0이라, 그때는 두 분할이 같으면 1로 둔다.
    """
    ri, eri = rand_index(u, v), expected_rand_index(u, v)
    # E[RI] = 1이면 분모 0(두 분할 모두 군집 1개이거나 모두 점 하나씩). 같은 분할이면 1, 아니면 0으로 둔다(모형의 선택)
    if abs(1.0 - eri) < 1e-15:
        return 1.0 if ri == 1.0 else 0.0
    return (ri - eri) / (1.0 - eri)


# ---------------------------------------------------------------- 정보량: MI, NMI, E{I}, AMI


def entropy(labels):
    # 군집별 점 수. 엔트로피 H = -sum p log p (자연로그)
    c = np.bincount(_codes(labels)).astype(float)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def mutual_info(u, v):
    """I(U;V) = sum n_ij/N log(N n_ij / (a_i b_j)) (Vinh 2절)."""
    t = contingency(u, v).astype(float)
    n = t.sum()
    a = t.sum(axis=1, keepdims=True)
    b = t.sum(axis=0, keepdims=True)
    # n_ij = 0인 칸은 0 log 0 = 0으로 보고 건너뛴다(log 0 방지)
    nz = t > 0
    # a @ b는 (R, C) 외적 a_i b_j. I = sum n_ij/N log(N n_ij / (a_i b_j))
    return float(np.sum(t[nz] / n * np.log(n * t[nz] / (a @ b)[nz])))


def _norm(hu, hv, average):
    if average == "max":
        return max(hu, hv)
    if average == "arithmetic":
        return (hu + hv) / 2.0
    if average == "geometric":
        return float(np.sqrt(hu * hv))
    if average == "min":
        return min(hu, hv)
    raise ValueError(f"알 수 없는 average: {average}")


def nmi(u, v, average="max"):
    hu, hv = entropy(u), entropy(v)
    # 둘 다 군집이 하나뿐이면 H = 0이라 분모가 0. 같은 분할이므로 1로 둔다
    if hu == 0.0 and hv == 0.0:
        return 1.0  # 둘 다 군집 하나: 같은 분할
    d = _norm(hu, hv, average)
    return mutual_info(u, v) / d if d > 0 else 0.0


def _log_fact_table(n):
    # lf[k] = log(k!). 팩토리얼을 직접 계산하면 넘치므로 로그로 쌓는다
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])


def expected_mutual_info_sizes(a, b):
    """Vinh·Epps·Bailey(2010) 식 (2). a = U의 군집 크기들, b = V의 군집 크기들(합이 같아야 함)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = int(a.sum())
    if n != int(b.sum()):
        raise ValueError("두 분할의 점 수가 달라요")
    lf = _log_fact_table(n)
    # E{I} = sum_i sum_j sum_{n_ij} (n_ij/N) log(N n_ij/(a_i b_j)) * P(n_ij). P는 초기하분포(순열 모형)
    emi = 0.0
    for ai in a:
        for bj in b:
            lo = max(1, ai + bj - n)  # n_ij = 0 항은 0 log 0 = 0이라 1부터
            hi = min(ai, bj)
            if hi < lo:
                continue
            nij = np.arange(lo, hi + 1)
            term1 = nij / n * np.log(n * nij / (ai * bj))
            # 로그 초기하확률 P(n_ij) = a! b! (N-a)! (N-b)! / (N! n! (a-n)! (b-n)! (N-a-b+n)!). nij는 벡터라 한 번에 모든 값
            logp = (lf[ai] + lf[bj] + lf[n - ai] + lf[n - bj] - lf[n]
                    - lf[nij] - lf[ai - nij] - lf[bj - nij] - lf[n - ai - bj + nij])
            emi += float(np.sum(term1 * np.exp(logp)))
    return emi


def expected_mutual_info(u, v):
    t = contingency(u, v)
    return expected_mutual_info_sizes(t.sum(axis=1), t.sum(axis=0))


def ami(u, v, average="max"):
    """(I - E{I}) / (정규화 - E{I}) (Vinh 식 (3)과 표 2)."""
    hu, hv = entropy(u), entropy(v)
    if hu == 0.0 and hv == 0.0:
        return 1.0
    i, e = mutual_info(u, v), expected_mutual_info(u, v)
    # AMI = (I - E{I}) / (정규화값 - E{I})
    d = _norm(hu, hv, average) - e
    # 분모가 0이면 I = E{I}일 때만 1, 아니면 0(퇴화한 경우)
    if abs(d) < 1e-15:
        return 1.0 if abs(i - e) < 1e-15 else 0.0
    return (i - e) / d


# ---------------------------------------------------------------- 라벨 없는 평가: 실루엣


def pairwise_dist(X):
    """유클리드 거리 행렬. 행렬곱 꼴(|x|^2 + |y|^2 - 2xy)은 반올림 오차가 커서 차이를 직접 제곱한다."""
    X = np.asarray(X, dtype=float)
    # (n, 1, d) - (1, n, d) = (n, n, d) -> 제곱합 (n, n). 메모리 n^2 d라 작은 데이터용
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def silhouette_samples(X, labels, D=None):
    """점마다 s(i) = (b(i) - a(i)) / max{a(i), b(i)}.

    a(i): 같은 군집의 다른 점들까지 평균 거리. b(i): 다른 군집마다 평균 거리를 구해 그중 가장 작은 값.
    자기 군집에 점이 하나뿐이면 s(i) = 0 (개념 카드 eval-silhouette의 약속. Rousseeuw 원문은 열지 못함).
    """
    lab = _codes(labels)
    k = lab.max() + 1
    # 실루엣은 군집이 2개 이상, 점 수 - 1개 이하일 때만 정의(군집이 하나면 b가 없고, 모두 한 점씩이면 a가 없다)
    if not 2 <= k <= len(lab) - 1:
        raise ValueError("실루엣은 군집 수가 2 이상, 점 수 - 1 이하일 때만 정의돼요")
    D = pairwise_dist(X) if D is None else D
    # sums[i, c] = 점 i에서 군집 c의 모든 점까지 거리의 합
    sums = np.stack([D[:, lab == c].sum(axis=1) for c in range(k)], axis=1)  # (n, k)
    size = np.bincount(lab, minlength=k).astype(float)
    own = size[lab]
    # a(i): 자기 군집 합을 (크기 - 1)로 나눈 평균(자기 자신 거리 0은 합에 들어 있지만 개수에서 뺀다)
    a = np.where(own > 1, sums[np.arange(len(lab)), lab] / np.maximum(own - 1, 1), 0.0)
    mean_other = sums / size[None, :]
    # 자기 군집은 b 후보에서 제외
    mean_other[np.arange(len(lab)), lab] = np.inf
    b = mean_other.min(axis=1)
    # s = (b - a)/max(a, b). 1e-300은 a = b = 0(중복점)일 때 0 나누기 방지. 한 점짜리 군집은 0
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
    return s


def silhouette(X, labels, D=None):
    return float(np.mean(silhouette_samples(X, labels, D)))


# ---------------------------------------------------------------- 비교용 k-평균


def kmeans(X, k, seed=0, n_init=5, max_iter=300):
    """k-means++로 시작해 Lloyd 반복. n_init번 중 제곱거리 합(inertia)이 가장 작은 결과."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        # k-means++: 첫 중심은 점 하나를 균등하게 뽑는다
        cent = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            # 각 점에서 가장 가까운 기존 중심까지의 제곱거리. 이 값에 비례하는 확률로 다음 중심을 뽑는다
            d2 = np.min(((X[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
            cent.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        cent = np.array(cent)
        lab = None
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
            new = d2.argmin(axis=1)
            # 배정이 바뀌지 않으면 수렴
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for c in range(k):
                # 빈 군집은 중심을 그대로 둔다(재초기화하지 않는 단순화)
                if np.any(lab == c):
                    cent[c] = X[lab == c].mean(axis=0)
        # 군집 내 제곱거리 합. n_init번 중 가장 작은 결과를 채택
        inertia = float(((X - cent[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy())
    return best[1]
