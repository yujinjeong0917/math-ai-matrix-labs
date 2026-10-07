"""13장 NumPy 최소 구현: 파라미터·FLOP 세기, 거듭제곱 법칙 피팅, 계산 예산 배분.

- count_params(cfg): scaling_torch.TinyLM과 같은 구조의 파라미터 수를 식으로 센다.
- flops_per_token(cfg): 학습 한 토큰의 FLOP. Kaplan 등(2020)의 근사 6N과, 행렬곱을 하나씩 센 값을 같이 낸다.
- fit_power_law(N, L): log L = log k - α log N 직선 맞추기(Kaplan 꼴, 바닥 E 없음).
- fit_chinchilla(N, D, L): L = E + A/N^α + B/D^β. α, β를 격자로 고르면 나머지 E, A, B는
  선형 최소제곱(음수 금지)으로 바로 풀린다. 가장 잘 맞는 격자점 주변을 한 번 더 촘촘히 찾는다.
- optimal_allocation(C, fit): 제약 C = 6ND에서 손실을 최소로 하는 N, D (Hoffmann 등 2022, 식 (4)).
"""

import numpy as np

# ---------------------------------------------------------------- 세기


def count_params(cfg, vocab=64, n_ctx=32, ff=4):
    """(N, N_total). N은 토큰·위치 임베딩을 뺀 수(Kaplan 등 Table 1의 'non-embedding').

    블록 하나: 어텐션 W_Q, W_K, W_V, W_O = 4d², MLP = d·ff·d + ff·d + ff·d·d + d, LayerNorm 2개 = 4d.
    마지막 LayerNorm 2d, 출력층 d·V + V. 우리 모델은 출력층을 임베딩과 공유하지 않아서 N에 넣는다.
    """
    d, L = cfg["d"], cfg["L"]
    block = 4 * d * d + (2 * ff * d * d + ff * d + d) + 4 * d
    n = L * block + 2 * d + d * vocab + vocab
    emb = vocab * d + n_ctx * d
    return n, n + emb


def flops_per_token(cfg, vocab=64, n_ctx=32, ff=4):
    """학습 한 토큰의 FLOP 추정. 곱셈·덧셈을 각각 1로 센다(행렬곱 m×k·k×n = 2mkn).

    exact_forward: 행렬곱만 센 순전파. 블록마다 투영 2·4d², MLP 2·2·ff·d², 점수 QKᵀ와 가중합 AV가
    토큰마다 2·n_ctx·d씩(마스크로 가린 칸도 계산하므로 n_ctx 전체), 출력층 2·d·V.
    학습은 역전파가 순전파의 약 2배라 3배로 잡는다(Kaplan 등 §2.1).
    """
    d, L = cfg["d"], cfg["L"]
    n, _ = count_params(cfg, vocab, n_ctx, ff)
    proj = 2 * 4 * d * d
    mlp = 2 * 2 * ff * d * d
    attn_ctx = 2 * 2 * n_ctx * d
    head = 2 * d * vocab
    fwd = L * (proj + mlp + attn_ctx) + head
    return {
        "approx_6N": 6 * n,
        "exact_forward": fwd,
        "exact_train": 3 * fwd,
        "context_share_of_forward": L * attn_ctx / fwd,
        "ratio_exact_over_6N": 3 * fwd / (6 * n),
    }


# ---------------------------------------------------------------- 피팅


def fit_power_law(N, L):
    """log L = c - α log N 를 최소제곱으로 맞춘다. L = (N_c/N)^α 꼴이고 N_c = e^{c/α}."""
    x, y = np.log(np.asarray(N, float)), np.log(np.asarray(L, float))
    slope, c = np.polyfit(x, y, 1)
    alpha = -slope
    resid = y - (c + slope * x)
    return {"alpha": float(alpha), "Nc": float(np.exp(c / alpha)), "rmse_log": float(np.sqrt(np.mean(resid**2)))}


def _nnls_small(X, y):
    """열이 2~3개뿐인 음수 금지 최소제곱. 쓰는 열의 조합을 다 풀어 보고 가장 좋은 해를 고른다."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if np.all(coef >= 0):  # 제약 없는 해가 이미 음수가 아니면 그게 답이다
        return float(np.sum((X @ coef - y) ** 2)), coef
    best = None
    k = X.shape[1]
    for mask in range(1, 2**k - 1):  # 열 일부만 쓰는 경우(나머지 계수 0)
        cols = [j for j in range(k) if mask >> j & 1]
        coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        if np.any(coef < 0):
            continue
        full = np.zeros(k)
        full[cols] = coef
        sse = float(np.sum((X @ full - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, full)
    return best


def chinchilla_loss(N, D, p):
    return p["E"] + p["A"] / np.asarray(N, float) ** p["alpha"] + p["B"] / np.asarray(D, float) ** p["beta"]


def fit_chinchilla(N, D, L, alphas=None, betas=None, refine=2, E_fixed=None):
    """L = E + A/N^α + B/D^β 를 제곱오차로 맞춘다.

    α, β를 고정하면 식이 E, A, B에 대해 일차식(1, N^-α, D^-β의 가중합)이라 최소제곱 한 번으로 풀린다.
    그래서 α, β만 격자로 훑고, 가장 좋은 점 주변을 격자 간격의 1/10로 refine번 더 훑은 뒤
    레벤버그-마쿼트로 다섯 값을 함께 다듬는다.
    E_fixed를 주면 바닥 E를 그 값으로 고정하고 A, B만 푼다(이론 최저 손실을 아는 장난감 과제에서 쓴다).
    """
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    alphas = np.linspace(0.02, 1.5, 75) if alphas is None else np.asarray(alphas)
    betas = np.linspace(0.02, 1.5, 75) if betas is None else np.asarray(betas)
    best = None
    for _ in range(refine + 1):
        for a in alphas:
            for b in betas:
                if E_fixed is None:
                    X = np.stack([np.ones_like(N), N**-a, D**-b], axis=1)
                    sse, (E, A, B) = _nnls_small(X, L)
                else:
                    X = np.stack([N**-a, D**-b], axis=1)
                    sse, (A, B) = _nnls_small(X, L - E_fixed)
                    E = E_fixed
                if best is None or sse < best[0]:
                    best = (sse, dict(E=E, A=A, B=B, alpha=float(a), beta=float(b)))
        p = best[1]
        sa = (alphas[1] - alphas[0]) if len(alphas) > 1 else 0.0
        sb = (betas[1] - betas[0]) if len(betas) > 1 else 0.0
        alphas = np.linspace(max(1e-3, p["alpha"] - sa), p["alpha"] + sa, 21)
        betas = np.linspace(max(1e-3, p["beta"] - sb), p["beta"] + sb, 21)
    p = _lm_polish(N, D, L, best[1], E_fixed)
    p = {k: float(v) for k, v in p.items()}
    p["rmse"] = float(np.sqrt(np.mean((chinchilla_loss(N, D, p) - L) ** 2)))
    return p


def _lm_polish(N, D, L, p, E_fixed=None, iters=200):
    """격자 해에서 출발해 레벤버그-마쿼트로 다섯(또는 네) 값을 함께 다듬는다.

    격자는 국소해에 빠지지 않게 출발점을 고르는 역할만 하고, 마지막 자릿수는 이 단계가 맞춘다.
    A, B는 로그로 다뤄 양수를 유지하고, E는 음수가 되면 0으로 자른다.
    """
    lN, lD = np.log(N), np.log(D)
    th = np.array([p["E"], np.log(max(p["A"], 1e-12)), np.log(max(p["B"], 1e-12)), p["alpha"], p["beta"]])
    free = np.array([E_fixed is None, True, True, True, True])

    def resid(t):
        return t[0] + np.exp(t[1] - t[3] * lN) + np.exp(t[2] - t[4] * lD) - L

    def jac(t):
        a, b = np.exp(t[1] - t[3] * lN), np.exp(t[2] - t[4] * lD)
        return np.stack([np.ones_like(L), a, b, -lN * a, -lD * b], axis=1)[:, free]

    lam, r = 1e-3, resid(th)
    for _ in range(iters):
        J = jac(th)
        H = J.T @ J
        step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -J.T @ r)
        cand = th.copy()
        cand[free] += step
        cand[0] = max(cand[0], 0.0)
        rc = resid(cand)
        if rc @ rc < r @ r:
            th, r, lam = cand, rc, lam / 3
            if np.max(np.abs(step)) < 1e-12:
                break
        else:
            lam *= 4
            if lam > 1e12:
                break
    return {"E": th[0], "A": np.exp(th[1]), "B": np.exp(th[2]), "alpha": th[3], "beta": th[4]}


def optimal_allocation(C, p):
    """Hoffmann 등(2022) 식 (4): N_opt = G (C/6)^a, D_opt = G^{-1} (C/6)^b,
    G = (αA / (βB))^{1/(α+β)}, a = β/(α+β), b = α/(α+β)."""
    al, be = p["alpha"], p["beta"]
    G = (al * p["A"] / (be * p["B"])) ** (1.0 / (al + be))
    a, b = be / (al + be), al / (al + be)
    C = np.asarray(C, float)
    return {"N_opt": G * (C / 6) ** a, "D_opt": (C / 6) ** b / G, "a": a, "b": b, "G": G}
