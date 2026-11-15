"""5장 실습: AIPW(증강 역확률가중)와 이중강건, 교차적합. 표준 라이브러리만 사용합니다.

결과모형(μ)과 성향점수모형(e)을 각각 맞게/틀리게 만들어 네 가지 조합을 돌립니다.
진짜 효과는 데이터를 만들 때 정해 두었어요.

무엇을 재는가: 평균 처치 효과(ATE)를 세 가지로 추정한다. (1) 결과모형만(regression), (2) 가중치만(IPW), (3) 둘을 합친 AIPW
(Augmented IPW, 증강 역확률가중). AIPW는 둘 중 하나만 맞아도 불편(이중강건)이라는 점을 네 조합(맞음/틀림)으로 확인한다.
또 모형이 데이터에 과적합될 때 같은 행으로 맞추고 평가하는 것과 교차적합(cross-fitting)의 차이를 본다.
입력: make_data(n, seed)의 행 목록(x: 공변량, t: 쿠폰, y: 주문액 천 원, 가상). 출력: 조합별 추정값, 반복 요약표.
테스트의 기대값이 장 본문의 4조합 표와 교차적합 표에 대응한다. 진짜 효과 TAU는 8.0.
"""
import bisect
import math
import random


def normal(rng):
    """Box-Muller 표준정규 한 개."""
    u1, u2 = rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def sigmoid(z):
    """로지스틱 함수."""
    return 1.0 / (1.0 + math.exp(-z))


def solve(a, b):
    """연립방정식 a z = b를 부분 피벗팅 가우스 소거법으로 푼다(작은 크기 전용)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(i + 1, n):
            f = m[r][i] / m[i][i]
            for c in range(i, n + 1):
                m[r][c] -= f * m[i][c]
    z = [0.0] * n
    for i in range(n - 1, -1, -1):
        z[i] = (m[i][n] - sum(m[i][c] * z[c] for c in range(i + 1, n))) / m[i][i]
    return z


def fit_logit(features, labels, iters=30):
    """로지스틱 회귀를 뉴턴 방법으로 맞춘다. g = X'(y - p), H = X' diag(p(1-p)) X, beta <- beta + H^-1 g."""
    k = len(features[0])
    beta = [0.0] * k
    for _ in range(iters):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for f, y in zip(features, labels):
            p = sigmoid(sum(b * v for b, v in zip(beta, f)))
            w = p * (1 - p)
            for i in range(k):
                grad[i] += (y - p) * f[i]
                for j in range(k):
                    hess[i][j] += w * f[i] * f[j]
        step = solve(hess, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return beta


def fit_ols(features, ys):
    """최소제곱 계수를 정규방정식 (X'X) b = X'y로 푼다."""
    k = len(features[0])
    xtx = [[sum(f[i] * f[j] for f in features) for j in range(k)] for i in range(k)]
    xty = [sum(f[i] * y for f, y in zip(features, ys)) for i in range(k)]
    return solve(xtx, xty)


def basis(x, degree):
    """다항 기저 [1, x, x^2, ..., x^degree]. 차수 2가 데이터 생성 규칙과 같다(맞는 모형), 차수 1은 틀린 모형."""
    return [x ** d for d in range(degree + 1)]


# ---------- 손계산: 고객 4명 ----------
# (쿠폰 여부 T, 다음 달 주문액 Y, 천 원 가상). 쿠폰 확률은 모두 0.5로 알고 있다고 하자.
HAND = [(1, 58.0), (1, 60.0), (0, 50.0), (0, 52.0)]


def hand_aipw(mu1, mu0, e=0.5, table=HAND):
    """고객별 점수 psi_i = mu1 - mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e) 와 그 평균.
    mu1, mu0는 결과모형의 예측(상수로 단순화), e는 쿠폰 확률. 뒤의 두 항이 결과모형 오차를 가중치로 보정하는 항이다."""
    psi = []
    for t, y in table:
        psi.append(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e))
    return psi, sum(psi) / len(psi)


# 진짜 데이터 규칙: 배정 로짓과 결과 모두 x의 제곱항을 가진다.
TAU = 8.0
CLIP = 0.02  # 교차적합 실험에서만 성향점수를 0.02~0.98로 자른다


def make_data(n, seed):
    """x ~ 표준정규. 쿠폰 확률과 y0가 모두 x의 이차식이라 선형 모형은 곡률을 놓친다. 효과는 모두에게 TAU(상수)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = normal(rng)
        # 진짜 로짓에 x^2 항(0.9)이 있다. (x*x - 1)은 E[x^2] = 1을 빼 평균 수준을 맞추기 위한 중심화
        e = sigmoid(-0.4 + 0.5 * x + 0.9 * (x * x - 1.0))
        t = 1 if rng.random() < e else 0
        y0 = 50.0 + 6.0 * x + 8.0 * x * x + 5.0 * normal(rng)  # 쿠폰이 없을 때의 y(x의 이차식 + 잡음)
        rows.append({"x": x, "t": t, "y": y0 + TAU * t})
    return rows


def naive_diff(rows):
    """단순 비교. x가 극단인 고객이 쿠폰을 더 받고 y도 높아서(x^2 항) 진짜 8보다 훨씬 크게 나온다."""
    t = [r["y"] for r in rows if r["t"] == 1]
    c = [r["y"] for r in rows if r["t"] == 0]
    return sum(t) / len(t) - sum(c) / len(c)


# ---------- 두 가지 모형: 맞는 것(차수 2)과 틀린 것(차수 1) ----------
def fit_models(train, out_degree, ps_degree):
    """훈련 행으로 결과모형 mu1, mu0와 성향점수 e를 맞춰 예측 함수를 돌려준다.
    out_degree: 결과모형 다항 차수, ps_degree: 성향점수 로짓 다항 차수. mu1은 처치군만, mu0는 비처치군만으로 각각 맞춘다."""
    b1 = fit_ols([basis(r["x"], out_degree) for r in train if r["t"] == 1], [r["y"] for r in train if r["t"] == 1])
    b0 = fit_ols([basis(r["x"], out_degree) for r in train if r["t"] == 0], [r["y"] for r in train if r["t"] == 0])
    bp = fit_logit([basis(r["x"], ps_degree) for r in train], [r["t"] for r in train])

    def mu1(x):
        return sum(c * v for c, v in zip(b1, basis(x, out_degree)))

    def mu0(x):
        return sum(c * v for c, v in zip(b0, basis(x, out_degree)))

    def e(x):
        return sigmoid(sum(c * v for c, v in zip(bp, basis(x, ps_degree))))

    return mu1, mu0, e


def three_estimates(rows, mu1, mu0, e):
    """결과모형만(regression), 가중치만(IPW, Hajek), AIPW를 한꺼번에 계산. 반환은 이 순서의 튜플.
    성향점수는 여기서 자르지 않는다(자르기는 교차적합 실험의 aipw_flexible에서만)."""
    n = len(rows)
    reg = sum(mu1(r["x"]) - mu0(r["x"]) for r in rows) / n
    n1 = d1 = n0 = d0 = 0.0
    psi = 0.0
    for r in rows:
        x, t, y = r["x"], r["t"], r["y"]
        ex = e(x)
        if t == 1:
            n1 += y / ex
            d1 += 1 / ex
        else:
            n0 += y / (1 - ex)
            d0 += 1 / (1 - ex)
        # AIPW 점수: 결과모형 예측 차이 + 가중 잔차 보정. 결과모형이 맞으면 보정항 평균이 0, 성향점수가 맞으면 모형 편향을 보정한다
        psi += mu1(x) - mu0(x) + t * (y - mu1(x)) / ex - (1 - t) * (y - mu0(x)) / (1 - ex)
    return reg, n1 / d1 - n0 / d0, psi / n


# (이름, 결과모형 차수, 성향점수모형 차수). 2는 데이터 생성 규칙과 같아 "맞음", 1은 곡률을 놓쳐 "틀림"
COMBOS = (
    ("둘 다 맞음", 2, 2),
    ("결과모형만 틀림", 1, 2),
    ("성향점수모형만 틀림", 2, 1),
    ("둘 다 틀림", 1, 1),
)


def run_combo(rows, out_degree, ps_degree):
    """한 조합으로 모형을 맞추고 (결과모형만, IPW, AIPW) 추정값을 돌려준다."""
    mu1, mu0, e = fit_models(rows, out_degree, ps_degree)
    return three_estimates(rows, mu1, mu0, e)


# ---------- 교차적합(cross-fitting) ----------
def knn_model(train, k=1):
    """x가 가장 가까운 훈련 행 k개의 결과 평균. 일부러 데이터에 바짝 붙는 유연한 모형.
    k=1이면 훈련 행 자신에게 적용할 때 예측이 정확히 자기 y라서 잔차가 0이 된다(과적합)."""
    pts = sorted((r["x"], r["y"]) for r in train)
    xs = [p[0] for p in pts]

    def f(x):
        i = bisect.bisect_left(xs, x)
        # 삽입 위치 i 좌우로 k개씩을 후보 창으로 잡고, 그중 x가 가장 가까운 k개를 쓴다(경계에서는 창이 잘림)
        lo, hi = max(0, i - k), min(len(xs), i + k)
        near = sorted(pts[lo:hi], key=lambda p: abs(p[0] - x))[:k]
        return sum(p[1] for p in near) / len(near)

    return f


def aipw_flexible(rows, folds=1, k=1, ps_degree=2, seed=0):
    """folds=1: 같은 행으로 맞추고 같은 행에 적용. folds=2: 반으로 나눠 서로에게 적용.
    교차적합은 모형을 맞추는 행과 점수를 계산하는 행을 분리해, 과적합된 잔차가 0으로 사라지는 것을 막는다."""
    n = len(rows)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)  # 이 장에서 유일하게 shuffle을 쓰는 곳. 접기(fold) 배정을 seed로 재현한다
    parts = [idx] if folds == 1 else [idx[i::folds] for i in range(folds)]
    psi = 0.0
    for part in parts:
        test = [rows[i] for i in part]
        # folds=2에서는 평가 접기(part)를 뺀 나머지 행으로만 모형을 맞춘다
        train = rows if folds == 1 else [rows[i] for i in idx if i not in set(part)]
        m1 = knn_model([r for r in train if r["t"] == 1], k)
        m0 = knn_model([r for r in train if r["t"] == 0], k)
        bp = fit_logit([basis(r["x"], ps_degree) for r in train], [r["t"] for r in train])
        for r in test:
            x, t, y = r["x"], r["t"], r["y"]
            # 성향점수를 [CLIP, 1-CLIP]으로 잘라 극단적 가중치(1/e)가 점수를 지배하지 못하게 한다
            ex = min(max(sigmoid(sum(c * v for c, v in zip(bp, basis(x, ps_degree)))), CLIP), 1 - CLIP)
            psi += m1(x) - m0(x) + t * (y - m1(x)) / ex - (1 - t) * (y - m0(x)) / (1 - ex)
    return psi / n


def mean(v):
    """산술평균."""
    return sum(v) / len(v)


def sd(v):
    """표본표준편차(분모 n - 1)."""
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def rmse(v):
    """진짜 효과 TAU 기준 제곱평균제곱근 오차. 편향과 흔들림이 합쳐진 총 오차다."""
    return math.sqrt(sum((a - TAU) ** 2 for a in v) / len(v))


def cross_fit_runs(reps=100, n=400, seed0=900):
    """같은 데이터에 같은 행 적용(ins)과 2-겹 교차적합(cf)을 reps번 비교한다. 반환: 두 추정값 리스트."""
    ins, cf = [], []
    for s in range(reps):
        r = make_data(n, seed0 + s)
        ins.append(aipw_flexible(r, folds=1))
        cf.append(aipw_flexible(r, folds=2, seed=s))
    return ins, cf


def replicate(reps, n, seed0):
    """네 조합 x 세 추정량을 reps번 반복한 추정값 목록. table[조합이름] = (회귀 목록, IPW 목록, AIPW 목록)."""
    table = {name: ([], [], []) for name, _, _ in COMBOS}
    for k in range(reps):
        rows = make_data(n, seed0 + k)
        for name, od, pd in COMBOS:
            reg, ipw, aipw = run_combo(rows, od, pd)
            table[name][0].append(reg)
            table[name][1].append(ipw)
            table[name][2].append(aipw)
    return table


def main():
    print("[손계산: 고객 4명]")
    for label, m1, m0, e in (("결과모형이 10만큼 벌어진다고 예측(실제는 8)", 60.0, 50.0, 0.5),
                             ("결과모형이 맞음(예측 59, 51)", 59.0, 51.0, 0.5),
                             ("결과모형이 맞음, 확률은 0.2로 잘못 넣음", 59.0, 51.0, 0.2)):
        psi, avg = hand_aipw(m1, m0, e)
        print(f"{label}: 고객별 점수 {[round(p, 1) for p in psi]}, 평균 {avg:.1f}")
    print()
    rows = make_data(2000, 3)
    print(f"[한 번 돌린 결과] n=2000, seed=3, 진짜 효과 {TAU:.1f} (천 원, 가상), 쿠폰 {sum(r['t'] for r in rows)}명")
    print(f"단순 비교 {naive_diff(rows):.2f}")
    for name, od, pd in COMBOS:
        reg, ipw, aipw = run_combo(rows, od, pd)
        print(f"{name}: 결과모형만 {reg:.2f}, IPW {ipw:.2f}, AIPW {aipw:.2f}")

    print("\n[100번 반복] n=1000, seed 500부터. 진짜 값 대비 평균 오차 / 표준편차")
    tab = replicate(100, 1000, 500)
    for name, _, _ in COMBOS:
        reg, ipw, aipw = tab[name]
        print(f"{name}: 결과모형만 {mean(reg) - TAU:+.2f}, IPW {mean(ipw) - TAU:+.2f}, AIPW {mean(aipw) - TAU:+.2f} "
              f"(AIPW 표준편차 {sd(aipw):.2f})")

    print("\n[교차적합] 1-최근접 결과모형(일부러 과적합), 성향점수는 0.02~0.98로 자름, n=400, 100번 반복")
    ins, cf = cross_fit_runs()
    for name, v in (("같은 행으로 맞춤", ins), ("2-겹 교차적합", cf)):
        print(f"{name}: 평균 오차 {mean(v) - TAU:+.2f}, 표준편차 {sd(v):.2f}, 제곱평균제곱근 오차 {rmse(v):.2f}")

if __name__ == "__main__":
    main()
