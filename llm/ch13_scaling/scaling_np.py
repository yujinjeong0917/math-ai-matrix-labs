"""13장 NumPy 최소 구현: 파라미터·FLOP 세기, 거듭제곱 법칙 피팅, 계산 예산 배분.

- count_params(cfg): scaling_torch.TinyLM과 같은 구조의 파라미터 수를 식으로 센다.
- flops_per_token(cfg): 학습 한 토큰의 FLOP. Kaplan 등(2020)의 근사 6N과, 행렬곱을 하나씩 센 값을 같이 낸다.
- fit_power_law(N, L): log L = log k - α log N 직선 맞추기(Kaplan 꼴, 바닥 E 없음).
- fit_chinchilla(N, D, L): L = E + A/N^α + B/D^β. α, β를 격자로 고르면 나머지 E, A, B는
  선형 최소제곱(음수 금지)으로 바로 풀린다. 가장 잘 맞는 격자점 주변을 한 번 더 촘촘히 찾는다.
- optimal_allocation(C, fit): 제약 C = 6ND에서 손실을 최소로 하는 N, D (Hoffmann 등 2022, 식 (4)).

입력: 모델 설정 cfg(d = 모델 차원, L = 층 수, h = 헤드 수), 측정된 (N, D, L) 점들. 출력: 파라미터 수, FLOP, 피팅된 지수와 계수, 최적 (N, D).
기호: N = 파라미터 수, D = 학습 토큰 수, L(또는 loss) = 검증 손실(nat), C = 계산량(FLOP). 장 본문 숫자는 experiments.py의 결과와 대응한다.
"""

import numpy as np

# ---------------------------------------------------------------- 세기


def count_params(cfg, vocab=64, n_ctx=32, ff=4):
    """(N, N_total). N은 토큰·위치 임베딩을 뺀 수(Kaplan 등 Table 1의 'non-embedding').

    블록 하나: 어텐션 W_Q, W_K, W_V, W_O = 4d², MLP = d·ff·d + ff·d + ff·d·d + d, LayerNorm 2개 = 4d.
    마지막 LayerNorm 2d, 출력층 d·V + V. 우리 모델은 출력층을 임베딩과 공유하지 않아서 N에 넣는다.
    """
    d, L = cfg["d"], cfg["L"]
    # 블록 하나: 어텐션 4d^2 + MLP(W1 + b1 + W2 + b2) + LayerNorm 2개(gamma, beta 각 d)
    block = 4 * d * d + (2 * ff * d * d + ff * d + d) + 4 * d
    # 블록 L개 + 마지막 LN 2d + 출력층 d*V + V. 임베딩은 뺀다
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
    # Q, K, V, O 네 투영, 토큰당 2*d*d씩
    proj = 2 * 4 * d * d
    # MLP 두 행렬곱, 토큰당 각 2 * d * (ff*d)
    mlp = 2 * 2 * ff * d * d
    # QK^T와 AV: 토큰당 각 2 * n_ctx * d. 마스크로 가려진 칸도 계산하므로 n_ctx 전체(구현에 따른 것)
    attn_ctx = 2 * 2 * n_ctx * d
    head = 2 * d * vocab
    # 출력층은 층 수와 무관하게 한 번
    fwd = L * (proj + mlp + attn_ctx) + head
    return {
        "approx_6N": 6 * n,
        "exact_forward": fwd,
        # 학습 = 순전파 + 역전파(순전파의 약 2배) = 3배(Kaplan 등 근사)
        "exact_train": 3 * fwd,
        "context_share_of_forward": L * attn_ctx / fwd,
        "ratio_exact_over_6N": 3 * fwd / (6 * n),
    }


# ---------------------------------------------------------------- 피팅


def fit_power_law(N, L):
    """log L = c - α log N 를 최소제곱으로 맞춘다. L = (N_c/N)^α 꼴이고 N_c = e^{c/α}."""
    # 양변에 로그: log L = log N_c * alpha - alpha log N 꼴의 직선
    x, y = np.log(np.asarray(N, float)), np.log(np.asarray(L, float))
    # 1차 다항식 최소제곱: slope = -alpha, c = alpha * log N_c
    slope, c = np.polyfit(x, y, 1)
    alpha = -slope
    resid = y - (c + slope * x)
    return {"alpha": float(alpha), "Nc": float(np.exp(c / alpha)), "rmse_log": float(np.sqrt(np.mean(resid**2)))}


def _nnls_small(X, y):
    """열이 2~3개뿐인 음수 금지 최소제곱. 쓰는 열의 조합을 다 풀어 보고 가장 좋은 해를 고른다."""
    # 먼저 제약 없는 최소제곱. 계수가 모두 음수가 아니면 그대로 답
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if np.all(coef >= 0):  # 제약 없는 해가 이미 음수가 아니면 그게 답이다
        return float(np.sum((X @ coef - y) ** 2)), coef
    best = None
    k = X.shape[1]
    # 2^k - 2개의 비어 있지 않은 부분집합(전체 제외: 전체는 이미 위에서 풀었다). 비트마스크로 열 선택
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
    # 식: L(N, D) = E + A/N^alpha + B/D^beta
    return p["E"] + p["A"] / np.asarray(N, float) ** p["alpha"] + p["B"] / np.asarray(D, float) ** p["beta"]


def fit_chinchilla(N, D, L, alphas=None, betas=None, refine=2, E_fixed=None):
    """L = E + A/N^α + B/D^β 를 제곱오차로 맞춘다.

    α, β를 고정하면 식이 E, A, B에 대해 일차식(1, N^-α, D^-β의 가중합)이라 최소제곱 한 번으로 풀린다.
    그래서 α, β만 격자로 훑고, 가장 좋은 점 주변을 격자 간격의 1/10로 refine번 더 훑은 뒤
    레벤버그-마쿼트로 다섯 값을 함께 다듬는다.
    E_fixed를 주면 바닥 E를 그 값으로 고정하고 A, B만 푼다(이론 최저 손실을 아는 장난감 과제에서 쓴다).
    """
    # float 배열로 통일
    N, D, L = (np.asarray(v, float) for v in (N, D, L))
    # alpha, beta의 초기 격자 75점 * 75점
    alphas = np.linspace(0.02, 1.5, 75) if alphas is None else np.asarray(alphas)
    betas = np.linspace(0.02, 1.5, 75) if betas is None else np.asarray(betas)
    best = None
    for _ in range(refine + 1):
        for a in alphas:
            for b in betas:
                if E_fixed is None:
                    # alpha, beta를 고정하면 L = E*1 + A*N^-a + B*D^-b: E, A, B에 대한 선형식. 열 3개
                    X = np.stack([np.ones_like(N), N**-a, D**-b], axis=1)
                    sse, (E, A, B) = _nnls_small(X, L)
                else:
                    X = np.stack([N**-a, D**-b], axis=1)
                    sse, (A, B) = _nnls_small(X, L - E_fixed)
                    E = E_fixed
                if best is None or sse < best[0]:
                    best = (sse, dict(E=E, A=A, B=B, alpha=float(a), beta=float(b)))
        p = best[1]
        # 격자 간격. 다음 격자는 현재 최적점 +- 한 간격의 21점(1/10로 세밀화)
        sa = (alphas[1] - alphas[0]) if len(alphas) > 1 else 0.0
        sb = (betas[1] - betas[0]) if len(betas) > 1 else 0.0
        alphas = np.linspace(max(1e-3, p["alpha"] - sa), p["alpha"] + sa, 21)
        betas = np.linspace(max(1e-3, p["beta"] - sb), p["beta"] + sb, 21)
    # 격자 해에서 출발해 연속 최적화로 마무리
    p = _lm_polish(N, D, L, best[1], E_fixed)
    p = {k: float(v) for k, v in p.items()}
    # 최종 잔차의 RMSE(손실 단위 그대로, 로그 아님)
    p["rmse"] = float(np.sqrt(np.mean((chinchilla_loss(N, D, p) - L) ** 2)))
    return p


def _lm_polish(N, D, L, p, E_fixed=None, iters=200):
    """격자 해에서 출발해 레벤버그-마쿼트로 다섯(또는 네) 값을 함께 다듬는다.

    격자는 국소해에 빠지지 않게 출발점을 고르는 역할만 하고, 마지막 자릿수는 이 단계가 맞춘다.
    A, B는 로그로 다뤄 양수를 유지하고, E는 음수가 되면 0으로 자른다.
    """
    lN, lD = np.log(N), np.log(D)
    # 매개변수 벡터 [E, log A, log B, alpha, beta]. A, B는 로그로 다뤄 양수를 보장
    th = np.array([p["E"], np.log(max(p["A"], 1e-12)), np.log(max(p["B"], 1e-12)), p["alpha"], p["beta"]])
    free = np.array([E_fixed is None, True, True, True, True])

    def resid(t):
        # 식: 예측 - 관측 = E + A N^-alpha + B D^-beta - L. A = exp(t1): exp(t1 - alpha log N) = A / N^alpha
        return t[0] + np.exp(t[1] - t[3] * lN) + np.exp(t[2] - t[4] * lD) - L

    def jac(t):
        # a = A N^-alpha, b = B D^-beta (각 항의 값)
        a, b = np.exp(t[1] - t[3] * lN), np.exp(t[2] - t[4] * lD)
        # 야코비안 열: dE = 1, d(logA) = a, d(logB) = b, d(alpha) = -log N * a, d(beta) = -log D * b. free로 고정한 변수의 열을 뺀다
        return np.stack([np.ones_like(L), a, b, -lN * a, -lD * b], axis=1)[:, free]

    lam, r = 1e-3, resid(th)
    for _ in range(iters):
        J = jac(th)
        H = J.T @ J
        # 레벤버그-마쿼트 단계: (J^T J + lam diag(J^T J)) step = -J^T r. lam이 작으면 가우스-뉴턴, 크면 경사하강에 가깝다
        step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -J.T @ r)
        cand = th.copy()
        cand[free] += step
        # E는 음수가 되면 0으로 자른다(손실의 바닥은 음수가 아니라는 가정)
        cand[0] = max(cand[0], 0.0)
        rc = resid(cand)
        # 제곱오차가 줄면 채택하고 lam을 줄이고, 아니면 거절하고 lam을 키운다
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
    # 식 (4) 유도: 제약 C = 6ND에서 A N^-alpha + B D^-beta를 최소화
    al, be = p["alpha"], p["beta"]
    # G = (alpha A / (beta B))^(1/(alpha + beta))
    G = (al * p["A"] / (be * p["B"])) ** (1.0 / (al + be))
    a, b = be / (al + be), al / (al + be)
    C = np.asarray(C, float)
    # N_opt = G (C/6)^a, D_opt = (C/6)^b / G. 따라서 6 * N_opt * D_opt = C
    return {"N_opt": G * (C / 6) ** a, "D_opt": (C / 6) ** b / G, "a": a, "b": b, "G": G}
