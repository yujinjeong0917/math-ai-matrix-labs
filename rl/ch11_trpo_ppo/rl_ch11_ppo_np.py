"""강화학습 11장 NumPy 최소 구현: 바닐라 정책경사, TRPO, PPO(클리핑)와 시드 통계.

정책은 cart-pole 상태 4개에 상수 1을 붙인 특징 phi = [x, x_dot, theta, theta_dot, 1]의 선형 결합을
시그모이드에 넣은 것이다. p(오른쪽 | s) = sigmoid(w . phi), 매개변수 w는 5개.
(TRPO 논문 §8.1도 cart-pole 기준선에 "a linear policy with six parameters"를 썼다.)

세 방법은 같은 판 묶음, 같은 이득 추정(할인 리턴을 묶음 안에서 평균 0, 표준편차 1로 맞춤)을 쓰고
"한 번 고칠 때 얼마나 가나"만 다르다.
- 바닐라: w <- w + lr * g. 보폭을 매개변수 공간의 거리로 정한다.
- TRPO: 평균 KL <= delta 안에서 대리 목적을 키운다. 켤레기울기로 F^{-1} g를 풀고 선탐색.
- PPO: 비율 r을 [1-eps, 1+eps] 밖으로 밀어도 이득이 없게 잘라 낸 목적을 Adam으로 여러 번 오른다.
"""

import math
import time

import numpy as np

import rl_ch11_cartpole as cp

# 할인율 gamma = 0.99(모형의 선택). 보상이 매 걸음 +1이라 G는 남은 걸음 수의 할인 합이 된다
GAMMA = 0.99


# ---------- 정책 ----------
def features(states):
    return np.concatenate([states, np.ones((states.shape[0], 1))], axis=1)


def sigmoid(z):
    # sigma(z) = 1/(1+e^-z)와 항등. tanh 꼴이라 exp 넘침이 없다
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # 큰 |z|에서도 넘치지 않는 꼴


def probs(w, phi):
    """(N, 2): [왼쪽 확률, 오른쪽 확률]."""
    p1 = sigmoid(phi @ w)
    return np.stack([1.0 - p1, p1], axis=1)


def log_prob(w, phi, a):
    z = phi @ w
    # a=1이면 log p, a=0이면 log(1-p). logaddexp로 p가 0/1에 붙어도 -inf가 되지 않는다
    # log sigmoid(z) = -log(1 + e^{-z}), log(1 - sigmoid(z)) = -log(1 + e^{z})
    return np.where(a == 1, -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z))


def log_probs_all(w, phi):
    """(N, 2): [log 왼쪽 확률, log 오른쪽 확률]. 확률이 0이나 1로 뭉개져도 유한하게 남는다."""
    z = phi @ w
    return np.stack([-np.logaddexp(0.0, z), -np.logaddexp(0.0, -z)], axis=1)


def kl_from_logp(logp_old_all, logp_new_all):
    """행마다 KL(old || new) = sum_a p_old(a) (log p_old(a) - log p_new(a))."""
    return (np.exp(logp_old_all) * (logp_old_all - logp_new_all)).sum(axis=1)


def grad_log_prob(w, phi, a):
    """d log pi(a|s) / dw = (a - p) phi. (N, 5)"""
    # 식: grad log pi(a|s) = (a - p) phi. (N,) -> (N,1) 곱하기 (N,5) = (N, 5)
    return (a - sigmoid(phi @ w))[:, None] * phi


def kl_categorical(p, q):
    """행마다 KL(p || q) = sum_a p_a log(p_a / q_a). 0 log 0 = 0."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return t.sum(axis=-1)


# ---------- 판 모으기 ----------
def collect(w, n_episodes, rng):
    """현재 정책으로 판 n_episodes개를 끝까지(최대 200걸음) 동시에 둔다."""
    s = cp.reset(n_episodes, rng)
    alive = np.ones(n_episodes, bool)
    PHI, A, EP, T = [], [], [], []
    lengths = np.zeros(n_episodes, int)
    for t in range(cp.MAX_STEPS):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        phi = features(s[idx])
        # 행동 표집: 오른쪽 확률 p = sigmoid(w.phi). 확률적 정책이 탐험 역할(epsilon 없음)
        a = (rng.random(idx.size) < sigmoid(phi @ w)).astype(int)
        PHI.append(phi), A.append(a), EP.append(idx), T.append(np.full(idx.size, t))
        nxt, done = cp.step(s[idx], a)
        s[idx] = nxt
        lengths[idx] += 1
        # 끝난 판은 빠지고 살아 있는 판만 계속 진행(판들을 동시에 벡터화해 둔다)
        alive[idx[done]] = False
    phi, a, ep, t = (np.concatenate(v) for v in (PHI, A, EP, T))
    # lexsort는 마지막 키(ep)가 1순위: 판 번호 먼저, 같은 판 안에서 t 순(함정: 키 순서가 인자 순서의 역)
    order = np.lexsort((t, ep))  # 판 순서, 그 안에서 시간 순서로 정렬
    phi, a, ep, t = phi[order], a[order], ep[order], t[order]
    # 식: G_t = sum_{k=0}^{L-t-1} gamma^k = (1 - gamma^(L-t))/(1-gamma) (등비급수). 끝나는 걸음의 보상도 포함하므로 남은 걸음 수는 L - t
    # 보상은 매 걸음 1. 판 길이 L에서 t걸음째의 할인 리턴 G_t = (1 - gamma^(L - t)) / (1 - gamma)
    G = (1.0 - GAMMA ** (lengths[ep] - t)) / (1.0 - GAMMA)
    return {"phi": phi, "a": a, "ep": ep, "t": t, "G": G, "lengths": lengths,
            "logp_old": log_prob(w, phi, a), "logp_old_all": log_probs_all(w, phi)}


def advantages(batch):
    G = batch["G"]
    # 이득 추정 A = 표준화한 G(가치 기준선 없이 묶음 평균 하나를 빼는 단순 기준선). 1e-8은 0 나눔 방지
    return (G - G.mean()) / (G.std() + 1e-8)


# ---------- 바닐라 정책경사 ----------
def pg_gradient(w, batch, adv):
    """g = 평균[ grad log pi(a|s) * A ]  (PPO 논문 식 (1))."""
    # 식: g = E[grad log pi(a|s) A] (표본 평균). (N,5)*(N,1) 평균 -> (5,)
    return (grad_log_prob(w, batch["phi"], batch["a"]) * adv[:, None]).mean(axis=0)


# ---------- PPO ----------
def ppo_clip_objective(logp_new, logp_old, adv, eps):
    """L^CLIP = 평균[ min(r A, clip(r, 1-eps, 1+eps) A) ],  r = exp(logp_new - logp_old)."""
    # 확률비 r = pi_new(a|s) / pi_old(a|s) = exp(log 차). 로그로 빼서 안정하게
    r = np.exp(logp_new - logp_old)
    # 식: L^CLIP = E[min(r A, clip(r, 1-eps, 1+eps) A)]. eps = 클리핑 폭(이 코드의 기본 0.2). A>0이면 r이 1+eps를 넘는 이득을, A<0이면 1-eps 아래로 내리는 이득을 막는다
    return np.minimum(r * adv, np.clip(r, 1 - eps, 1 + eps) * adv).mean()


def ppo_clip_loss(logp_new, logp_old, adv, eps):
    """최소화용 손실(-L^CLIP)과 logp_new에 대한 기울기."""
    r = np.exp(logp_new - logp_old)
    unclipped, clipped = r * adv, np.clip(r, 1 - eps, 1 + eps) * adv
    loss = -np.minimum(unclipped, clipped).mean()
    # min이 잘리지 않은 쪽을 고른 칸(또는 r이 구간 안이라 둘이 같은 칸)에서만 기울기가 흐른다
    # min이 '잘리지 않은 항'을 고른 곳과 r이 구간 안인 곳만 기울기를 통과(잘린 항은 상수라 기울기 0). 경계값에서의 처리는 torch.min/clamp의 규약과 맞춰 둔 것으로 보인다(확인 필요)
    live = (unclipped <= clipped) | ((r >= 1 - eps) & (r <= 1 + eps))
    # d(-r A)/d logp = -r A (r = exp(logp)이므로). 평균이라 1/N. 반환은 logp_new에 대한 기울기 (N,)
    dlogp = -np.where(live, r * adv, 0.0) / r.size
    return loss, dlogp


def ppo_loss_grad_w(w, phi, a, logp_old, adv, eps):
    loss, dlogp = ppo_clip_loss(log_prob(w, phi, a), logp_old, adv, eps)
    # 연쇄법칙: dloss/dw = sum_i dloss/dlogp_i * dlogp_i/dw. (N,)@(N,5) -> (5,)
    return loss, dlogp @ grad_log_prob(w, phi, a)


class Adam:
    """torch.optim.Adam(기본값 betas=(0.9, 0.999), eps=1e-8)과 같은 순서로 계산한다."""

    def __init__(self, n, lr):
        self.lr, self.m, self.v, self.k = lr, np.zeros(n), np.zeros(n), 0

    def step(self, w, grad):
        self.k += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        bc1, bc2 = 1 - 0.9**self.k, 1 - 0.999**self.k
        denom = np.sqrt(self.v) / math.sqrt(bc2) + 1e-8
        # Adam 갱신(편향 보정 bc1, bc2 포함). torch.optim.Adam과 계산 순서를 맞춰 1e-9 수준으로 일치시킨다. 손실 최소화라 w - ...(내려감)
        return w - (self.lr / bc1) * self.m / denom


def ppo_update(w, batch, adv, eps, epochs, n_minibatch, opt, rng):
    """같은 묶음을 epochs번 돌며 미니배치마다 Adam 한 걸음. eps=inf면 잘라 내기 없음."""
    N = adv.size
    clip_frac = 0.0
    for _ in range(epochs):
        # epoch마다 데이터를 새로 섞어 미니배치로 나눈다. 같은 묶음(옛 정책으로 모은 데이터)을 epochs번 재사용하는 것이 PPO의 핵심
        perm = rng.permutation(N)
        for mb in np.array_split(perm, n_minibatch):
            _, g = ppo_loss_grad_w(w, batch["phi"][mb], batch["a"][mb], batch["logp_old"][mb], adv[mb], eps)
            w = opt.step(w, g)
    r = np.exp(log_prob(w, batch["phi"], batch["a"]) - batch["logp_old"])
    if np.isfinite(eps):
        # 갱신 후 비율이 클리핑 구간 밖인 표본의 비율(진단값)
        clip_frac = float(np.mean((r < 1 - eps) | (r > 1 + eps)))
    return w, {"minibatch_steps": epochs * n_minibatch, "clip_frac": clip_frac,
               "ratio_min": float(r.min()), "ratio_max": float(r.max())}


# ---------- TRPO ----------
def fisher_matrix(w, phi):
    """평균 KL(pi_old || pi_w)의 헤세 행렬(w = w_old에서). 베르누이 정책이면 평균[p(1-p) phi phi^T]."""
    p = sigmoid(phi @ w)
    # 식: 베르누이 정책의 피셔 행렬 F = E[p(1-p) phi phi^T] (= 평균 KL의 헤세 행렬, w = w_old에서). (5,N)@(N,5) = (5,5)
    return (phi * (p * (1 - p))[:, None]).T @ phi / phi.shape[0]


def fisher_vector_product(w, phi, v):
    """행렬을 만들지 않고 F v = J^T M J v (TRPO 부록 C.1)."""
    p = sigmoid(phi @ w)
    # F v를 (N,)에서 (5,)로 줄이며 계산: phi @ v (N,) -> 가중 p(1-p) -> phi^T @ (.) (5,). 5x5라 행렬을 만들어도 되지만 TRPO 방식(행렬 없이)을 그대로 보이려는 것
    return phi.T @ (p * (1 - p) * (phi @ v)) / phi.shape[0]


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    """A x = b를 A v 곱만으로 푼다. 반환: x, 실제 반복 수."""
    x = np.zeros_like(b)
    r = b.copy()
    d = r.copy()
    rr = r @ r
    for k in range(iters):
        Ad = Avp(d)
        # 켤레기울기(CG): 걸음 크기 alpha = r.r / d.A d
        alpha = rr / (d @ Ad)
        x += alpha * d
        r -= alpha * Ad
        rr_new = r @ r
        # 잔차 제곱합이 tol(1e-10) 미만이면 조기 종료. 반환 반복 수는 k+1
        if rr_new < tol:
            return x, k + 1
        # 새 탐색 방향 = 잔차 + (이전 대비 잔차 비) * 이전 방향 (Fletcher-Reeves식 계수)
        d = r + (rr_new / rr) * d
        rr = rr_new
    return x, iters


def surrogate(w, batch, adv):
    """L(w) = 평균[ pi_w(a|s) / pi_old(a|s) * A ]."""
    # 식: 대리 목적 L(w) = E[pi_w/pi_old A]. w = w_old에서 값 = E[A] = 0에 가깝고 기울기는 정책경사
    return np.mean(np.exp(log_prob(w, batch["phi"], batch["a"]) - batch["logp_old"]) * adv)


def mean_kl(logp_old_all, w, phi):
    """묶음의 상태들에서 평균 KL(pi_old || pi_w). TRPO의 제약식이 재는 양이다."""
    return float(kl_from_logp(logp_old_all, log_probs_all(w, phi)).mean())


def trpo_step(w, batch, adv, delta, line_search=True, max_backtracks=10, cg_iters=10):
    # TRPO 1단계: 대리 목적 기울기 g
    g = pg_gradient(w, batch, adv)  # 대리 목적의 기울기(w = w_old에서 정책경사와 같다)
    phi = batch["phi"]
    # 2단계: F s = g를 CG로 풀어 s = F^-1 g(자연 기울기 방향). cg_iters = CG 반복 상한
    s, k = conjugate_gradient(lambda v: fisher_vector_product(w, phi, v), g, iters=cg_iters)
    sAs = s @ fisher_vector_product(w, phi, s)
    # 식: 평균 KL 약 (1/2) (beta s)^T F (beta s) = delta를 풀면 beta = sqrt(2 delta / s^T F s). delta = KL 신뢰영역 반경(기본 0.01)
    beta = math.sqrt(2 * delta / sAs) if sAs > 0 else 0.0  # 부록 C: beta = sqrt(2 delta / s^T A s)
    full = beta * s
    L0 = surrogate(w, batch, adv)
    if not line_search:
        w_new = w + full
        return w_new, {"cg_iters": k, "backtracks": 0, "accepted": True}
    for j in range(max_backtracks):
        # 선탐색(line search): 전체 걸음에서 시작해 절반씩 줄이며(0.5^j) 제약을 맞춘다
        w_new = w + (0.5**j) * full
        # 수락 조건: 실제 평균 KL <= delta(2차 근사가 틀릴 수 있어 실제 값으로 재확인) 그리고 대리 목적이 개선됨. 끝까지 실패하면 w를 그대로 둔다(accepted=False)
        if mean_kl(batch["logp_old_all"], w_new, phi) <= delta and surrogate(w_new, batch, adv) > L0:
            return w_new, {"cg_iters": k, "backtracks": j, "accepted": True}
    return w, {"cg_iters": k, "backtracks": max_backtracks, "accepted": False}


# ---------- 학습 루프 ----------
def train(algo, seed, n_iters=60, n_episodes=16, lr=0.01, eps=0.2, delta=0.01, epochs=10,
          n_minibatch=4, adam_lr=0.01, line_search=True, log_every=None):
    """algo: "pg" | "trpo" | "ppo". 갱신마다 리턴(그 묶음의 평균 판 길이), 누적 걸음, 갱신 전후 평균 KL을 남긴다."""
    # 난수 용도 분리: 환경/행동용(seed)과 PPO 미니배치용([seed,1]). 방법이 달라도 같은 seed에서 같은 판을 만들게 한다
    rng = np.random.default_rng(seed)  # 판 시작 상태와 행동 표본
    rng_mb = np.random.default_rng([seed, 1])  # PPO 미니배치 섞기
    w = np.zeros(5)
    opt = Adam(5, adam_lr) if algo == "ppo" else None
    hist = {"ret": [], "steps": [], "kl": [], "update_ms": [], "w_norm": [], "w": []}
    extra = []
    total = 0
    for it in range(n_iters):
        batch = collect(w, n_episodes, rng)
        total += batch["a"].size
        adv = advantages(batch)
        t0 = time.perf_counter()
        if algo == "pg":
            # 바닐라: w <- w + lr g (매개변수 공간 보폭 lr). 식 (1)의 경사 상승
            w_new = w + lr * pg_gradient(w, batch, adv)
            info = {}
        elif algo == "trpo":
            w_new, info = trpo_step(w, batch, adv, delta, line_search=line_search)
        elif algo == "ppo":
            w_new, info = ppo_update(w, batch, adv, eps, epochs, n_minibatch, opt, rng_mb)
        else:
            raise ValueError(algo)
        # 갱신 한 번의 시간(ms). 환경 의존 값
        ms = 1000 * (time.perf_counter() - t0)
        hist["ret"].append(float(batch["lengths"].mean()))
        hist["steps"].append(int(total))
        # 갱신 전후 평균 KL(old||new): 방법마다 '한 번에 정책을 얼마나 움직였나'의 공통 잣대
        hist["kl"].append(mean_kl(batch["logp_old_all"], w_new, batch["phi"]))
        hist["update_ms"].append(ms)
        extra.append(info)
        w = w_new
        hist["w_norm"].append(float(np.linalg.norm(w)))
        hist["w"].append(w.tolist())
    hist["final_w"] = w.tolist()
    hist["info"] = extra
    return hist


def drop_events(ret, frac=0.5, min_best=50.0):
    """리턴이 그때까지 최고치(50 이상)의 50% 아래로 떨어진 갱신 수. 회복 여부는 따지지 않는다."""
    # 붕괴 판정: 그때까지 최고 리턴(best)의 50% 아래로 떨어진 횟수/시점. 최고가 50 미만인 초반 흔들림은 제외(min_best)
    best, n = -np.inf, 0
    for i, r in enumerate(ret):
        if i > 0 and best >= min_best and r < frac * best:
            n += 1
        best = max(best, r)
    return n


def stuck_diagnosis(w, seed=123, n_episodes=16):
    """정책이 한쪽 행동으로 굳었는지: 고른 행동의 평균 확률과, 그 정책으로 모은 묶음에서 잰 정책경사의 크기."""
    batch = collect(np.asarray(w, float), n_episodes, np.random.default_rng(seed))
    p_chosen = np.exp(batch["logp_old"])
    g = pg_gradient(np.asarray(w, float), batch, advantages(batch))
    return {"mean_len": float(batch["lengths"].mean()), "mean_p_chosen": float(p_chosen.mean()),
            "frac_right": float(batch["a"].mean()), "grad_norm": float(np.linalg.norm(g))}


def collapsed(ret, steps, frac=0.5, window=10_000, min_best=50.0):
    """설계도의 붕괴 기준: 리턴이 그때까지 최고치의 50% 아래로 떨어진 뒤 1만 걸음 안에 50%로 돌아오지 못함.
    최고치가 50 미만인 초반 흔들림은 세지 않는다. 떨어진 뒤 남은 갱신이 하나도 없으면 판정하지 않는다.
    반환: 처음 붕괴한 갱신 번호(없으면 None)."""
    best = -np.inf
    for i in range(len(ret)):
        if i > 0 and best >= min_best and ret[i] < frac * best:
            # 떨어진 시점 이후 window(1만 걸음) 안의 갱신들. 하나라도 50% 이상으로 돌아오면 붕괴로 세지 않는다
            later = [j for j in range(i + 1, len(ret)) if steps[j] <= steps[i] + window]
            if later and all(ret[j] < frac * best for j in later):
                return i
        best = max(best, ret[i])
    return None


# ---------- 시드 통계 (scipy 없이) ----------
# scipy 없이 t분포 p값을 얻기 위한 불완전 베타 함수 구현(Numerical Recipes 방식). 통계 보조 코드라 RL 식과는 무관
def _betacf(a, b, x, iters=300, eps=3e-16):
    """정규화된 불완전 베타 함수의 연분수(Numerical Recipes betacf와 같은 Lentz 방식)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    """자유도 df인 t분포에서 |T| >= |t|일 확률(양측)."""
    return betainc(df / 2, 0.5, df / (df + t * t))


def welch_t(x, y):
    """Welch t 검정(두 묶음 분산이 같다고 가정하지 않음). 반환: t, 자유도, 양측 p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    if vx + vy == 0:
        return 0.0, float(x.size + y.size - 2), 1.0
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy)
    # Welch-Satterthwaite 자유도 근사
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1))
    return float(t), float(df), float(t_two_sided_p(t, df))


def bootstrap_ci(x, n_boot=10_000, seed=0, level=0.95):
    """평균의 부트스트랩 구간(백분위 방법). Henderson 등은 pivotal 방법을 썼다(Table 3)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    # 복원추출로 평균을 n_boot번 다시 구해 분위수로 구간을 만든다. (n_boot, n) 인덱스 -> (n_boot,) 평균
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(x.mean()), float(lo), float(hi)
