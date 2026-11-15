"""강화학습 연결 장 NumPy 최소 구현: 비교 데이터로 보상 모델 만들기(Bradley-Terry), KL 정규 정책경사(RLHF),
DPO, 검증기 보상(RLVR).

장난감 "문장" 공간
- 단어 6개(0~5), 길이 8. 가능한 문장은 6^8 = 1,679,616개라 전부 열거해서 기댓값과 KL을 정확히 계산할 수 있다.
- 숨은 참 보상 r*(y): 단어 v의 품질 W[v]를 두 번까지만 쳐 주고, 같은 단어를 세 번째부터 쓰면 한 번마다 PENALTY만큼 깎는다.
    r*(y) = sum_v W[v] * min(c_v, 2) - PENALTY * sum_v max(c_v - 2, 0)      (c_v = 문장 속 단어 v의 개수)
  "좋은 말도 같은 말만 반복하면 싫다"는 취향을 숫자로 적은 것이다. 학습 쪽은 이 식을 모르고, 둘 중 어느 쪽이 나은지만 본다.
- 보상 모델 r_hat(y) = u . c(y): 단어 개수의 선형식(단어마다 점수 하나). 반복 감점을 표현할 수 없다.

정책(작은 자기회귀 언어모델)
- 위치 t, 바로 앞 단어 prev(맨 앞은 BOS = 6), 다음 단어 v마다 로짓 theta[t, prev, v]. 매개변수 8 x 7 x 6 = 336개.
- theta = 0이면 모든 위치에서 6개 단어가 1/6씩, 즉 균등 분포다. 기준 정책 pi_ref를 이것으로 둔다.

함수와 실험의 대응
- 입력: 비교 쌍(이긴 문장 Yw, 진 문장 Yl) 각 (n, 8)과 시드. 출력: 보상 모델 점수 u (6,), 정책 theta (8, 7, 6), 학습 곡선(걸음별 참 보상, KL 등).
- bt_*: 보상 모델 적합, kl_regularized_pg: RLHF(보상 모델 + KL 벌점 정책경사), dpo_*/train_dpo*: DPO, rlvr_*: 검증기 보상.
- 이 파일의 숫자는 experiments.py의 E0~E8 결과(results/chbridge.json)와 test_rl_chbridge_rlhf_dpo.py의 기대값으로 이어진다.
- 단순화: 6^8개 문장을 전부 열거해 기댓값과 KL을 표본 오차 없이 계산한다(실제 언어모델은 불가능). 기준 정책은 균등분포로 고정.
"""

import math

import numpy as np

# 상수: V = 단어 수, L = 문장 길이, BOS = 문장 시작 표시(단어 번호 6, 실제 단어와 겹치지 않는 값)
# W = 단어별 품질, PENALTY = 같은 단어를 세 번째부터 쓸 때 한 번당 깎는 점수(모형의 선택)
V = 6
L = 8
BOS = V
W = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
PENALTY = 1.0
WORDS = ["가", "나", "다", "라", "마", "바"]  # 화면에 보일 때 쓰는 이름(품질 0.0 ~ 1.0 순)


# ---------- 문장 공간 전수 열거 ----------
_CACHE = {}


def all_sequences():
    """(6^8, 8) uint8. 사전순."""
    if "Y" not in _CACHE:
        idx = np.arange(V**L, dtype=np.int64)
        Y = np.empty((V**L, L), np.uint8)
        for t in range(L - 1, -1, -1):
            # 문장 번호를 6진법으로 풀어 마지막 위치부터 채운다. 결과 행 순서는 사전순(가가가...부터 바바바...까지)
            Y[:, t] = idx % V
            idx //= V
        _CACHE["Y"] = Y
    return _CACHE["Y"]


def counts(Y):
    """(N, 6): 문장마다 단어별 개수."""
    Y = np.asarray(Y)
    return np.stack([(Y == v).sum(axis=1) for v in range(V)], axis=1).astype(np.float64)


def true_reward(Y):
    c = counts(Y)
    # 머리말 식: r*(y) = sum_v W[v] min(c_v,2) - PENALTY sum_v max(c_v-2,0)
    # 앞 항은 두 번까지만 인정, 뒤 항은 세 번째부터의 초과분 벌점. c는 (N, 6), 반환은 (N,)
    return (W * np.minimum(c, 2)).sum(axis=1) - PENALTY * np.maximum(c - 2, 0).sum(axis=1)


def n_unique(Y):
    return (counts(Y) > 0).sum(axis=1)


def _flat_index(Y):
    """theta를 (8*7*6,)로 폈을 때 문장의 각 위치가 가리키는 칸. (N, 8)"""
    Y = np.asarray(Y, np.int64)
    # 각 위치의 '바로 앞 단어': 한 칸 밀어 넣고 맨 앞은 BOS. (N, 8)
    prev = np.concatenate([np.full((Y.shape[0], 1), BOS), Y[:, :-1]], axis=1)
    # theta[t, prev, v]를 1차원으로 폈을 때의 번호 = t*(7*6) + prev*6 + v (행 우선 배치)
    return np.arange(L)[None, :] * (V + 1) * V + prev * V + Y


def _enum_tables():
    if "tab" not in _CACHE:
        Y = all_sequences()
        _CACHE["tab"] = {
            "flat": _flat_index(Y).astype(np.int32),
            "r_true": true_reward(Y),
            "counts": counts(Y),
            "uniq": n_unique(Y).astype(np.float64),
        }
    return _CACHE["tab"]


# ---------- 정책 ----------
def zeros_theta():
    return np.zeros((L, V + 1, V))


def log_softmax(theta):
    # 수치 안정화: 로짓에서 최댓값을 빼도 softmax는 같고 exp 넘침을 막는다. 마지막 축이 '다음 단어 v'
    m = theta.max(axis=-1, keepdims=True)
    return theta - m - np.log(np.exp(theta - m).sum(axis=-1, keepdims=True))


def logprob(theta, Y):
    """log pi_theta(y) = sum_t log pi(y_t | t, y_{t-1})."""
    ls = log_softmax(theta).reshape(-1)
    return ls[_flat_index(Y)].sum(axis=1)


def logprob_ref(Y):
    # 기준 정책은 균등분포라 어느 문장이든 log pi_ref(y) = L * log(1/V) = -L log V (상수)
    return np.full(np.asarray(Y).shape[0], -L * math.log(V))


def grad_logprob(theta, Y, weights):
    """sum_i weights_i * d log pi(y_i) / d theta. (8, 7, 6)"""
    # 정책 확률 P[t, prev, v] = pi(v | t, prev). (8, 7, 6)
    P = np.exp(log_softmax(theta))
    Y = np.asarray(Y, np.int64)
    prev = np.concatenate([np.full((Y.shape[0], 1), BOS), Y[:, :-1]], axis=1)
    g = np.zeros_like(theta)
    for t in range(L):
        # 식: 위 줄의 미분은 onehot(y) - p. 아래 첫 줄이 onehot 항, 둘째 줄이 -p 항
        # weights_i가 문장별 가중(REINFORCE에서는 (보상-기준선)/묶음 크기). np.add.at은 같은 칸에 여러 문장이 겹쳐도 모두 더한다
        # d log softmax(theta[t, prev])[y] / d theta[t, prev, :] = onehot(y) - p
        np.add.at(g[t], (prev[:, t], Y[:, t]), weights)
        np.add.at(g[t], prev[:, t], -weights[:, None] * P[t, prev[:, t]])
    return g


def sample(theta, n, rng):
    P = np.exp(log_softmax(theta))
    Y = np.empty((n, L), np.uint8)
    prev = np.full(n, BOS)
    for t in range(L):
        # 역변환 표집: 누적확률 cum 중 u보다 작은 칸의 수가 곧 표집된 단어 번호. (n, 6)
        cum = P[t, prev].cumsum(axis=1)
        u = rng.random(n)[:, None]
        y = (u > cum).sum(axis=1)
        # 부동소수점 누적합이 1에 못 미쳐 u > cum이 전부 참이 되는 경우를 막는다(함정)
        y = np.minimum(y, V - 1)
        Y[:, t] = y
        prev = y
    return Y


def exact_stats(theta, u=None):
    """전체 문장 공간에서 정확히: E[r*], E[r_hat], KL(pi || pi_ref), 고유 단어 수 기댓값, 가장 흔한 문장의 확률."""
    tab = _enum_tables()
    ls = log_softmax(theta).reshape(-1)
    lp = ls[tab["flat"]].sum(axis=1)
    p = np.exp(lp)
    out = {
        "r_true": float(p @ tab["r_true"]),
        # KL(pi || pi_ref) = sum_y p(y) (log p(y) - log pi_ref(y)) = sum p (lp + L log V). 6^8개 문장 전체를 합하므로 표본 오차 없이 정확
        "kl": float(p @ (lp + L * math.log(V))),
        "uniq": float(p @ tab["uniq"]),
        "p_top": float(p.max()),
        "mass": float(p.sum()),
        "exp_counts": (p @ tab["counts"]).tolist(),
    }
    if u is not None:
        out["r_hat"] = float(p @ (tab["counts"] @ u))
    return out


def tilt_stats(score, beta):
    """균등 기준 위의 정확한 KL 정규 최적해 pi(y) ∝ pi_ref(y) exp(score(y)/beta)의 통계.
    score는 전체 문장 공간 길이의 배열."""
    tab = _enum_tables()
    z = score / beta
    # 최댓값을 빼 exp 넘침을 막는다(정규화되므로 결과는 같다)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    lp = np.log(np.maximum(p, 1e-300))
    return {
        "r_true": float(p @ tab["r_true"]),
        "r_score": float(p @ score),
        "kl": float(p @ (lp + L * math.log(V))),
        "uniq": float(p @ tab["uniq"]),
    }


def tilt_theta_linear(u, beta):
    """r_hat = u . c(y)가 위치별 합이라 균등 기준의 최적해는 위치마다 독립인 softmax(u / beta)다.
    같은 분포를 이 장의 정책 꼴(theta)로 적은 것."""
    theta = np.zeros((L, V + 1, V))
    # 균등 기준에서 최적해 pi ∝ pi_ref exp(r_hat/beta). r_hat이 단어별 합이므로 위치·앞 단어와 무관한 로짓 u/beta를 모든 칸에 둔다
    theta[:, :, :] = u / beta
    return theta


# ---------- Bradley-Terry 보상 모델 ----------
def sigmoid(z):
    # sigma(z) = 1/(1+e^-z)와 같은 값. tanh 꼴은 z가 크게 음수여도 exp 넘침이 없다
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def log_sigmoid(z):
    # log sigma(z) = -log(1 + e^-z). logaddexp로 안정하게 계산
    return -np.logaddexp(0.0, -z)


def make_pairs(n, rng, theta=None):
    """기준 정책(또는 theta)에서 문장 두 개씩 뽑고, 참 보상으로 BT 확률을 만들어 이긴 쪽을 정한다.
    돌려주는 값: (이긴 문장 Yw, 진 문장 Yl), 둘 다 (n, 8)."""
    th = zeros_theta() if theta is None else theta
    A = sample(th, n, rng)
    B = sample(th, n, rng)
    # Bradley-Terry(BT) 모형: A가 이길 확률 = sigma(r*(A) - r*(B)). 학습 쪽은 r*를 모르고 이 결과(이긴 쪽)만 본다
    pa = sigmoid(true_reward(A) - true_reward(B))
    a_wins = rng.random(n) < pa
    Yw = np.where(a_wins[:, None], A, B)
    Yl = np.where(a_wins[:, None], B, A)
    return Yw, Yl


def bt_loss(u, Yw, Yl, l2=1e-3):
    """-mean log sigma(r_hat(yw) - r_hat(yl)) + l2/2 |u|^2 와 u에 대한 기울기."""
    # r_hat = u . c이므로 r_hat(yw) - r_hat(yl) = (c(yw) - c(yl)) . u. d: (n, 6)
    d = counts(Yw) - counts(Yl)
    z = d @ u
    loss = -log_sigmoid(z).mean() + 0.5 * l2 * (u @ u)
    # d/du[-log sigma(z)] = -(1 - sigma(z)) d. 여기에 L2 항의 기울기 l2*u를 더한다
    grad = -(d * (1 - sigmoid(z))[:, None]).mean(axis=0) + l2 * u
    return float(loss), grad


def bt_reward_model_fit(Yw, Yl, l2=1e-3, iters=50):
    """뉴턴법. 단어 개수의 합이 늘 8이라 u에 상수를 더해도 손실이 같다. l2가 그중 크기가 가장 작은 해를 고른다."""
    d = counts(Yw) - counts(Yl)
    u = np.zeros(V)
    for _ in range(iters):
        z = d @ u
        s = sigmoid(z)
        g = -(d * (1 - s)[:, None]).mean(axis=0) + l2 * u
        # 헤세 행렬 = mean[sigma(1-sigma) d d^T] + l2 I. l2 I 덕에 가역(d는 모든 행의 합이 0이라 상수 방향이 퇴화)
        H = (d * (s * (1 - s))[:, None]).T @ d / d.shape[0] + l2 * np.eye(V)
        # 뉴턴 걸음: u <- u - H^-1 g
        step = np.linalg.solve(H, g)
        u = u - step
        if np.abs(step).max() < 1e-12:
            break
    return u


def rm_accuracy(u, A, B):
    """r_hat가 고른 쪽이 r*로도 더 나은 비율(r*가 같은 쌍은 뺀다)."""
    rt = true_reward(A) - true_reward(B)
    rh = counts(A) @ u - counts(B) @ u
    # r*가 같은 쌍은 정답이 없으므로 제외. 부동소수점 비교라 0 대신 허용 오차 1e-12
    keep = np.abs(rt) > 1e-12
    return float((np.sign(rh[keep]) == np.sign(rt[keep])).mean())


# ---------- RLHF: 보상 모델 + KL 정규 정책경사 ----------
class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, theta, grad_ascent):
        self.t += 1
        # Adam: 기울기의 1차 모멘트(m)와 2차 모멘트(v)의 지수이동평균
        self.m = self.b1 * self.m + (1 - self.b1) * grad_ascent
        self.v = self.b2 * self.v + (1 - self.b2) * grad_ascent**2
        # 초기에 0으로 시작한 이동평균의 치우침을 바로잡는 보정. 반환은 theta + lr * mh / (sqrt(vh)+eps)로 '오르는' 방향(경사 상승)
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + self.eps)


def kl_regularized_pg(u, beta, seed, steps=300, batch=64, lr=0.05, eval_every=10, log_samples=None):
    """문장마다 보상 R(y) = r_hat(y) - beta * (log pi(y) - log pi_ref(y))를 주고 REINFORCE(기준선 = 묶음의
    나머지 평균)로 오른다. InstructGPT식 (2)의 KL 항과 같은 꼴이다(사전학습 항 gamma는 0).
    매 eval_every 걸음마다 전체 문장 공간에서 정확한 통계를 잰다."""
    # 시드와 용도 번호 2를 함께 넣어, 같은 seed여도 다른 용도(BT 쌍 등)와 난수열이 겹치지 않게 한다
    rng = np.random.default_rng([seed, 2])
    theta = zeros_theta()
    opt = Adam(theta.shape, lr)
    curve = []
    for k in range(steps + 1):
        if k % eval_every == 0:
            st = exact_stats(theta, u)
            st["step"] = k
            curve.append(st)
            if log_samples is not None:
                log_samples(k, theta, rng)
        if k == steps:
            break
        Y = sample(theta, batch, rng)
        # 식: R(y) = r_hat(y) - beta (log pi(y) - log pi_ref(y)). beta가 KL 벌점 세기. 문장 하나가 곧 한 번의 행동(문장 단위 보상)
        R = counts(Y) @ u - beta * (logprob(theta, Y) - logprob_ref(Y))
        # 기준선 b_i = 자기를 뺀 나머지 평균(leave-one-out). 분산을 줄이면서 자기 보상에 의존하지 않아 기울기가 치우치지 않는다
        b = (R.sum() - R) / (batch - 1)
        # REINFORCE 추정: (1/batch) sum_i (R_i - b_i) grad log pi(y_i)
        g = grad_logprob(theta, Y, (R - b) / batch)
        # g는 목적(기대 보상)을 키우는 방향이라 그대로 상승 방향으로 넣는다
        theta = opt.step(theta, g)
    return theta, curve


# ---------- DPO ----------
def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    """L = -mean log sigma(beta (logp_w - ref_w) - beta (logp_l - ref_l)).
    logp_w, logp_l에 대한 기울기도 돌려준다."""
    # 식: z = beta[log(pi/pi_ref)(yw) - log(pi/pi_ref)(yl)]. 암묵 보상의 차이. 손실 = -mean log sigma(z)
    z = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    loss = float(-log_sigmoid(z).mean())
    s = sigmoid(-z)  # 틀리게 줄 세운 정도: 암묵 보상이 진 쪽을 높게 볼수록 1에 가깝다
    n = z.shape[0]
    # d loss/d logp_w = -beta sigma(-z)/n, d loss/d logp_l = +beta sigma(-z)/n (부호가 반대: 이긴 쪽 확률은 올리고 진 쪽은 내린다)
    return loss, -beta * s / n, beta * s / n


def dpo_grad_theta(theta, Yw, Yl, beta):
    lw, ll = logprob(theta, Yw), logprob(theta, Yl)
    loss, dw, dl = dpo_loss(lw, ll, logprob_ref(Yw), logprob_ref(Yl), beta)
    # 연쇄법칙: 손실의 theta 기울기 = sum (d loss/d logp) * d logp/d theta. 이긴 문장과 진 문장 몫을 더한다
    g = grad_logprob(theta, Yw, dw) + grad_logprob(theta, Yl, dl)
    return loss, g


def train_dpo(Yw, Yl, beta, steps=300, lr=0.05, eval_every=10, u_for_stats=None):
    """같은 선호 쌍 전체로 한 걸음씩(전체 묶음) Adam. 표본 생성 없음."""
    theta = zeros_theta()
    opt = Adam(theta.shape, lr)
    curve = []
    for k in range(steps + 1):
        if k % eval_every == 0:
            st = exact_stats(theta, u_for_stats)
            st["step"] = k
            st["dpo_loss"] = dpo_grad_theta(theta, Yw, Yl, beta)[0]
            curve.append(st)
        if k == steps:
            break
        _, g = dpo_grad_theta(theta, Yw, Yl, beta)
        # opt.step은 경사 상승이므로 손실 기울기에 -를 붙여 내려간다
        theta = opt.step(theta, -g)
    return theta, curve


def implicit_reward(theta, Y, beta):
    return beta * (logprob(theta, Y) - logprob_ref(Y))


# ---------- RLVR: 검증기 보상 ----------
def rlvr_reward(Y):
    """장난감 산수 검증기: 첫 단어(문제)를 숫자 a(가=0, ..., 바=5)로 읽고, 두 번째 단어(답)가 (a + 2) mod 6이면 1, 아니면 0.
    문제는 밖에서 주어진다고 보고, rlvr_pg는 첫 위치의 로짓을 고치지 않는다(문제를 쉬운 쪽으로 고르는 길을 막는다)."""
    Y = np.asarray(Y, np.int64)
    # 첫 단어를 문제 a, 둘째 단어를 답으로 읽는다(산수의 '문제'는 외부에서 정해진 값이라는 모형의 단순화)
    # 검증기 보상: 정답이면 1, 아니면 0. 보상 모델 없이 규칙으로 바로 채점(RLVR = 검증 가능한 보상으로 하는 강화학습)
    return ((Y[:, 0] + 2) % V == Y[:, 1]).astype(np.float64)


def rlvr_pg(seed, steps=200, batch=64, lr=0.05, beta=0.0, eval_every=10):
    rng = np.random.default_rng([seed, 3])
    theta = zeros_theta()
    opt = Adam(theta.shape, lr)
    tab = _enum_tables()
    Y_all = all_sequences()
    if "rv" not in _CACHE:
        _CACHE["rv"] = rlvr_reward(Y_all)
    rv = _CACHE["rv"]
    curve = []
    for k in range(steps + 1):
        if k % eval_every == 0:
            lp = log_softmax(theta).reshape(-1)[tab["flat"]].sum(axis=1)
            p = np.exp(lp)
            curve.append({"step": k, "success": float(p @ rv), "kl": float(p @ (lp + L * math.log(V))),
                          "r_true": float(p @ tab["r_true"])})
        if k == steps:
            break
        Y = sample(theta, batch, rng)
        # 보상은 검증기 점수. 기본값 beta=0이면 KL 벌점이 없다
        R = rlvr_reward(Y) - beta * (logprob(theta, Y) - logprob_ref(Y))
        b = (R.sum() - R) / (batch - 1)
        g = grad_logprob(theta, Y, (R - b) / batch)
        # theta[0]은 t=0(첫 위치)의 로짓 전체. 0으로 두면 Adam 갱신도 0이 되어 균등 분포가 유지된다
        g[0] = 0.0  # 문제(첫 단어)의 분포는 고정
        theta = opt.step(theta, g)
    return theta, curve


# ---------- 같은 표현력 비교용 ----------
def kl_between(theta_p, theta_q):
    """전체 문장 공간에서 정확한 KL(pi_p || pi_q)."""
    tab = _enum_tables()
    lp = log_softmax(theta_p).reshape(-1)[tab["flat"]].sum(axis=1)
    lq = log_softmax(theta_q).reshape(-1)[tab["flat"]].sum(axis=1)
    return float(np.exp(lp) @ (lp - lq))


def train_dpo_shared(Yw, Yl, beta, steps=3000, lr=0.05):
    """정책 꼴을 '위치·앞 단어와 상관없이 단어마다 로짓 하나(a_v)'로 묶은 DPO.
    이때 암묵 보상 beta * log(pi/pi_ref)는 beta * a . c(y) + 상수라서 단어 개수 보상 모델과 표현력이 같다."""
    a = np.zeros(V)
    opt = Adam(a.shape, lr)
    for _ in range(steps):
        # 모든 (위치, 앞 단어) 칸에 같은 로짓 a를 복사. broadcast_to는 읽기 전용이라 copy 필요
        theta = np.broadcast_to(a, (L, V + 1, V)).copy()
        _, g = dpo_grad_theta(theta, Yw, Yl, beta)
        # theta가 a의 복사이므로 a의 기울기 = theta 기울기를 위치와 앞 단어 축으로 합한 것. 손실을 줄이는 방향이라 -g
        a = opt.step(a, -g.sum(axis=(0, 1)))
    return np.broadcast_to(a, (L, V + 1, V)).copy(), a


def edge_features(Y):
    """(N, 336): 문장이 지나간 (위치, 앞 단어, 단어) 칸의 개수. DPO가 쓰는 정책 꼴과 같은 칸이다."""
    fl = _flat_index(Y)
    F = np.zeros((fl.shape[0], L * (V + 1) * V))
    rows = np.arange(fl.shape[0])
    for t in range(L):
        np.add.at(F, (rows, fl[:, t]), 1.0)
    return F


def bt_edge_fit(Yw, Yl, l2=1e-2, iters=100):
    """앞 단어까지 보는 보상 모델(336개 점수)을 BT 손실 + L2로 뉴턴법 적합."""
    D = edge_features(Yw) - edge_features(Yl)
    phi = np.zeros(D.shape[1])
    for _ in range(iters):
        s = sigmoid(D @ phi)
        g = -(D * (1 - s)[:, None]).mean(axis=0) + l2 * phi
        H = (D * (s * (1 - s))[:, None]).T @ D / D.shape[0] + l2 * np.eye(D.shape[1])
        step = np.linalg.solve(H, g)
        phi -= step
        if np.abs(step).max() < 1e-12:
            break
    return phi


def edge_score_all(phi):
    return phi[_enum_tables()["flat"].astype(np.int64)].sum(axis=1)


def pair_accuracy(score_diff, A, B):
    rt = true_reward(A) - true_reward(B)
    keep = np.abs(rt) > 1e-12
    return float((np.sign(score_diff[keep]) == np.sign(rt[keep])).mean())
