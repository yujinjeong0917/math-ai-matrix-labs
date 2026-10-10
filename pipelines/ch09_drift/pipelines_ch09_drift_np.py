"""9장 NumPy 최소 구현: 기준 분포와 지금 분포를 비교하는 두 검사.

SciPy를 쓰지 않는다(실습 저장소 의존성은 numpy, torch, pytest뿐). 그래서 KS 통계량과 p값을 직접 계산한다.

1) KS(콜모고로프-스미르노프) 두 표본 검정, 열 하나씩
   D = max_x |F_ref(x) - F_cur(x)|, F는 '이 값 이하인 비율'(경험 누적분포).
   p값은 표본이 클 때의 근사(콜모고로프 분포의 급수)를 쓴다. 같은 값이 많은 열(정수 열, 100원 단위 요금)에서는
   이 근사가 실제보다 큰 p값을 줘서 경보가 덜 울린다(보수적). 테스트가 순열 p값과 비교해 확인한다.
   열이 K개면 본페로니 보정: 가장 작은 p값이 alpha/K보다 작을 때만 경보.

2) MMD(최대평균불일치) 두 표본 검정, 열 전체를 한꺼번에
   가우스 커널 k(x,y) = exp(-|x-y|^2 / (2 sigma^2)), sigma는 합친 표본의 쌍별 거리 중앙값(Gretton 등 2012의 실험 설정).
   불편 추정량 MMD^2_u(Gretton 등 2012, 식 3). 문턱은 순열 검정으로 정하고 p = (1 + #{순열 통계량 >= 관측값}) / (1 + 순열 수).
   sigma는 합친 표본으로 정하므로 순열해도 바뀌지 않는다. 그래서 순열마다 커널을 다시 만들 필요가 없다.
"""

import numpy as np


# ---------------- KS ----------------
def ks_stat(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def kolmogorov_q(lam, terms=100):
    """P(K > lam), K는 콜모고로프 분포. 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lam^2)."""
    if lam < 0.2:
        return 1.0
    k = np.arange(1, terms + 1)
    return float(np.clip(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)), 0.0, 1.0))


def ks_pvalue(d, n, m):
    return kolmogorov_q(np.sqrt(n * m / (n + m)) * d)


def ks_perm_pvalue(a, b, n_perm=500, seed=0):
    """비교용: 근사 대신 순열로 구한 p값(같은 값이 많은 열의 실제 오경보율 확인용)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    d0 = ks_stat(a, b)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        hits += ks_stat(p[: len(a)], p[len(a):]) >= d0 - 1e-12
    return (1 + hits) / (1 + n_perm)


def ks_check(ref, cur, alpha=0.05, correction="bonferroni", names=None):
    """ref, cur: (행, 열) 행렬. 열마다 KS. 반환: 열별 D·p, 가장 작은 p, 경보 여부."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    k = ref.shape[1]
    names = names or [f"x{j}" for j in range(k)]
    ds = [ks_stat(ref[:, j], cur[:, j]) for j in range(k)]
    ps = [ks_pvalue(d, len(ref), len(cur)) for d in ds]
    level = alpha / k if correction == "bonferroni" else alpha
    return {
        "D": dict(zip(names, ds)), "p": dict(zip(names, ps)), "min_p": float(min(ps)),
        "level": level, "alarm": bool(min(ps) < level),
        "worst": names[int(np.argmin(ps))],
    }


# ---------------- MMD ----------------
def sq_dists(x, y):
    d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T
    return np.maximum(d, 0.0)


def median_bandwidth(z, max_rows=None):
    z = np.asarray(z, float)
    if max_rows is not None and len(z) > max_rows:
        z = z[:max_rows]
    d = np.sqrt(sq_dists(z, z))
    iu = np.triu_indices(len(z), 1)
    return float(np.median(d[iu]))


def rbf(x, y, sigma):
    return np.exp(-sq_dists(x, y) / (2.0 * sigma * sigma))


def mmd2_unbiased(x, y, sigma):
    """Gretton 등(2012) 식 (3): 같은 쪽끼리는 i != j인 쌍만, 서로 다른 쪽은 모든 쌍."""
    m, n = len(x), len(y)
    kxx, kyy, kxy = rbf(x, x, sigma), rbf(y, y, sigma), rbf(x, y, sigma)
    sxx = kxx.sum() - np.trace(kxx)
    syy = kyy.sum() - np.trace(kyy)
    return float(sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * kxy.sum() / (m * n))


def mmd2_loop(x, y, sigma):
    """테스트용 이중 루프 구현. 식을 한 항씩 그대로 옮겼다."""
    def k(a, b):
        return float(np.exp(-np.sum((a - b) ** 2) / (2.0 * sigma * sigma)))
    m, n = len(x), len(y)
    sxx = sum(k(x[i], x[j]) for i in range(m) for j in range(m) if i != j)
    syy = sum(k(y[i], y[j]) for i in range(n) for j in range(n) if i != j)
    sxy = sum(k(x[i], y[j]) for i in range(m) for j in range(n))
    return sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * sxy / (m * n)


def standardize_like(ref, *others):
    """ref의 평균·표준편차로 열 단위를 맞춘다(시간 측정·대조용)."""
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(a - mu) / sd for a in (ref, *others)]


def standardize_pooled(ref, cur):
    """두 표본을 합친 평균·표준편차로 열 단위를 맞춘 뒤 위아래로 붙인 행렬(앞 len(ref)행이 ref)."""
    z = np.vstack([np.asarray(ref, float), np.asarray(cur, float)])
    sd = z.std(0)
    return (z - z.mean(0)) / np.where(sd > 0, sd, 1.0)


def mmd_permutation_test(ref, cur, n_perm=200, seed=0, sigma=None):
    """반환: MMD^2_u, p값, sigma. 커널 행렬은 한 번만 만들고, 순열 전부를 표시 행렬 하나와의 곱으로 계산한다."""
    z = standardize_pooled(ref, cur)
    # 열 단위 맞추기는 합친 표본의 평균·표준편차로 한다. ref 쪽 통계로만 맞추면 어느 쪽이 ref인지에 따라
    # 변환이 달라져, 순열마다 같은 커널 행렬을 쓴다는 전제(커널이 두 쪽을 합친 표본만의 함수)가 깨진다.
    m, n = len(ref), len(cur)
    sigma = median_bandwidth(z) if sigma is None else sigma
    k = rbf(z, z, sigma)
    np.fill_diagonal(k, 0.0)
    total, row = k.sum(), k.sum(1)

    def stats(a):  # a: (N, P) 0/1 표시 행렬, 1이면 ref 쪽
        ka = k @ a
        sxx = (a * ka).sum(0)
        ar = row @ a
        sxy = ar - sxx
        syy = total - 2.0 * ar + sxx
        return sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * sxy / (m * n)

    a0 = np.zeros((m + n, 1))
    a0[:m] = 1.0
    obs = float(stats(a0)[0])
    rng = np.random.default_rng(seed)
    a = np.zeros((m + n, n_perm))
    for j in range(n_perm):
        a[rng.permutation(m + n)[:m], j] = 1.0
    null = stats(a)
    p = (1 + int(np.sum(null >= obs - 1e-15))) / (1 + n_perm)
    return {"mmd2": obs, "p": float(p), "sigma": sigma, "alarm": None}


def mmd2_blockwise(x, y, sigma, block=2000):
    """큰 n에서 메모리를 아끼는 같은 식. 계산량은 그대로 O((m+n)^2 d)."""
    def ksum(a, b, same):
        s = 0.0
        for i in range(0, len(a), block):
            for j in range(0, len(b), block):
                s += rbf(a[i:i + block], b[j:j + block], sigma).sum()
        return s - (len(a) if same else 0.0)  # 대각선 exp(0)=1 을 뺀다
    m, n = len(x), len(y)
    return float(ksum(x, x, True) / (m * (m - 1)) + ksum(y, y, True) / (n * (n - 1)) - 2.0 * ksum(x, y, False) / (m * n))
