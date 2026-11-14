"""4장 실습: 성향점수, 매칭, 역확률가중(IPW). 표준 라이브러리만 사용합니다.

가상 온라인 마켓: 쿠폰(T)은 구매 활동이 많은 고객(x)에게 더 자주 갔고,
활동이 많은 고객은 쿠폰이 없어도 더 많이 삽니다. 진짜 효과는 코드 안에서 알고 있어요.
"""
import math
import random


# ---------- 난수 (random() 하나만 쓰고 정규분포는 직접 만든다) ----------
def normal(rng):
    u1, u2 = rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ---------- 손계산 표 (고객 두 층) ----------
HAND = {
    "heavy": {"n": 100, "treated": 80, "y1": 70.0, "y0": 62.0},
    "light": {"n": 100, "treated": 20, "y1": 38.0, "y0": 30.0},
}


def hand_naive(table=HAND):
    nt = sum(s["treated"] for s in table.values())
    nc = sum(s["n"] - s["treated"] for s in table.values())
    m1 = sum(s["treated"] * s["y1"] for s in table.values()) / nt
    m0 = sum((s["n"] - s["treated"]) * s["y0"] for s in table.values()) / nc
    return m1, m0, m1 - m0


def hand_stratified(table=HAND):
    """층마다 차이를 구하고 층 크기로 평균 (ATE)."""
    n = sum(s["n"] for s in table.values())
    return sum(s["n"] * (s["y1"] - s["y0"]) for s in table.values()) / n


def hand_ipw(table=HAND):
    """층 안의 처치 비율을 성향점수로 보고 가중 평균 (Hajek 형태)."""
    num1 = den1 = num0 = den0 = 0.0
    for s in table.values():
        e = s["treated"] / s["n"]
        nt, nc = s["treated"], s["n"] - s["treated"]
        num1 += nt * (1 / e) * s["y1"]
        den1 += nt * (1 / e)
        num0 += nc * (1 / (1 - e)) * s["y0"]
        den0 += nc * (1 / (1 - e))
    return num1 / den1, num0 / den0, num1 / den1 - num0 / den0


# ---------- 장난감 데이터 ----------
def make_data(n, seed, slope=1.0, intercept=-0.2, tau0=8.0, tau1=0.0, noise=5.0):
    """x: 구매 활동 점수(표준화). T: 쿠폰. Y: 다음 달 주문액(천 원, 가상).
    Y0 = 50 + 10x + 잡음, 효과 tau(x) = tau0 + tau1*x."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = normal(rng)
        e = sigmoid(intercept + slope * x)
        t = 1 if rng.random() < e else 0
        tau = tau0 + tau1 * x
        y0 = 50.0 + 10.0 * x + noise * normal(rng)
        y = y0 + tau * t
        rows.append({"x": x, "t": t, "y": y, "tau": tau, "e_true": e})
    return rows


def true_ate(rows):
    return sum(r["tau"] for r in rows) / len(rows)


def true_att(rows):
    tr = [r["tau"] for r in rows if r["t"] == 1]
    return sum(tr) / len(tr)


def naive_diff(rows):
    t = [r["y"] for r in rows if r["t"] == 1]
    c = [r["y"] for r in rows if r["t"] == 0]
    return sum(t) / len(t) - sum(c) / len(c)


# ---------- 로지스틱 회귀 (뉴턴 방법) ----------
def solve(a, b):
    """작은 연립방정식 a z = b (가우스 소거법)."""
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


def fit_logit(features, labels, iters=25):
    """features: 각 행이 [1, x1, x2, ...]. 최대가능도 로지스틱 회귀."""
    k = len(features[0])
    beta = [0.0] * k
    for _ in range(iters):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for f, y in zip(features, labels):
            p = sigmoid(sum(b * v for b, v in zip(beta, f)))
            for i in range(k):
                grad[i] += (y - p) * f[i]
                w = p * (1 - p)
                for j in range(k):
                    hess[i][j] += w * f[i] * f[j]
        step = solve(hess, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    return beta


def propensity_scores(rows):
    beta = fit_logit([[1.0, r["x"]] for r in rows], [r["t"] for r in rows])
    return [sigmoid(beta[0] + beta[1] * r["x"]) for r in rows], beta


# ---------- 추정량 ----------
def ipw_ate(rows, ps):
    """Hajek(정규화) IPW. 추정 대상은 ATE."""
    n1 = d1 = n0 = d0 = 0.0
    for r, e in zip(rows, ps):
        if r["t"] == 1:
            n1 += r["y"] / e
            d1 += 1 / e
        else:
            n0 += r["y"] / (1 - e)
            d0 += 1 / (1 - e)
    return n1 / d1 - n0 / d0


def weights(rows, ps):
    return [1 / e if r["t"] == 1 else 1 / (1 - e) for r, e in zip(rows, ps)]


def effective_sample_size(w):
    return sum(w) ** 2 / sum(v * v for v in w)


def match_att(rows, ps, caliper=None):
    """처치 고객마다 성향점수가 가장 가까운 비처치 고객 1명(복원 허용). 추정 대상은 ATT."""
    ctrl = sorted((e, r["y"]) for r, e in zip(rows, ps) if r["t"] == 0)
    keys = [c[0] for c in ctrl]
    import bisect
    diffs, used, dropped = [], set(), 0
    for r, e in zip(rows, ps):
        if r["t"] != 1:
            continue
        i = bisect.bisect_left(keys, e)
        cand = [j for j in (i - 1, i) if 0 <= j < len(ctrl)]
        j = min(cand, key=lambda j: abs(keys[j] - e))
        if caliper is not None and abs(keys[j] - e) > caliper:
            dropped += 1
            continue
        used.add(j)
        diffs.append(r["y"] - ctrl[j][1])
    return sum(diffs) / len(diffs), len(diffs), len(used), dropped


def smd(rows, w=None):
    """x의 표준화 평균 차이(처치-비처치). w가 있으면 가중."""
    if w is None:
        w = [1.0] * len(rows)

    def stats(t):
        ws = [wi for r, wi in zip(rows, w) if r["t"] == t]
        xs = [r["x"] for r in rows if r["t"] == t]
        m = sum(a * b for a, b in zip(ws, xs)) / sum(ws)
        v = sum(a * (b - m) ** 2 for a, b in zip(ws, xs)) / sum(ws)
        return m, v

    m1, v1 = stats(1)
    m0, v0 = stats(0)
    return (m1 - m0) / math.sqrt((v1 + v0) / 2)


def trimmed_ipw(rows, ps, lo=0.1, hi=0.9):
    keep = [(r, e) for r, e in zip(rows, ps) if lo <= e <= hi]
    sub = [k[0] for k in keep]
    return ipw_ate(sub, [k[1] for k in keep]), len(sub), sum(r["tau"] for r in sub) / len(sub)


def replicate(reps, n, seed0, **kw):
    out = []
    for k in range(reps):
        rows = make_data(n, seed0 + k, **kw)
        ps, _ = propensity_scores(rows)
        out.append((naive_diff(rows), ipw_ate(rows, ps), true_ate(rows), max(weights(rows, ps)),
                    effective_sample_size(weights(rows, ps))))
    return out


def weighted_mean_x(pairs):
    """(행, 가중치) 쌍들의 활동 점수 x 가중 평균."""
    return sum(r["x"] * wi for r, wi in pairs) / sum(wi for _, wi in pairs)


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def main():
    print("[손계산 표]")
    m1, m0, d = hand_naive()
    print(f"단순 비교: 쿠폰 {m1:.1f}, 비쿠폰 {m0:.1f}, 차이 {d:.1f}")
    print(f"층별 비교 후 평균(ATE): {hand_stratified():.1f}")
    a, b, c = hand_ipw()
    print(f"IPW: 처치 {a:.1f}, 비처치 {b:.1f}, 차이 {c:.1f}")

    print("\n[겹침이 충분한 장면] n=2000, 진짜 효과 +8.0(천 원, 가상), seed=7")
    rows = make_data(2000, 7)
    ps, beta = propensity_scores(rows)
    print(f"쿠폰 받은 고객 {sum(r['t'] for r in rows)}명, 진짜 ATE {true_ate(rows):.2f}, 진짜 ATT {true_att(rows):.2f}")
    print(f"로지스틱 계수 절편 {beta[0]:.2f}, 기울기 {beta[1]:.2f}")
    print(f"단순 비교 {naive_diff(rows):.2f}")
    print(f"IPW(ATE) {ipw_ate(rows, ps):.2f}")
    att, nm, nu, dr = match_att(rows, ps)
    print(f"1:1 매칭(ATT) {att:.2f}, 처치 {nm}명, 사용된 비처치 {nu}명")
    w = weights(rows, ps)
    print(f"x 표준화 평균 차이: 전 {smd(rows):.2f}, IPW 후 {smd(rows, w):.2f}")
    print(f"최대 가중치 {max(w):.1f}, 유효 표본 크기 {effective_sample_size(w):.0f}")

    print("\n[겹침이 부족한 장면] 쿠폰이 활동 점수 상위자에게 거의 몰림, 효과 = 8 + 4x")
    kw = dict(slope=4.0, intercept=-5.0, tau0=8.0, tau1=4.0)
    rows2 = make_data(2000, 7, **kw)
    ps2, beta2 = propensity_scores(rows2)
    w2 = weights(rows2, ps2)
    print(f"쿠폰 받은 고객 {sum(r['t'] for r in rows2)}명, 진짜 ATE {true_ate(rows2):.2f}, 진짜 ATT {true_att(rows2):.2f}")
    print(f"단순 비교 {naive_diff(rows2):.2f}")
    print(f"IPW(ATE) {ipw_ate(rows2, ps2):.2f}")
    att2, nm2, nu2, dr2 = match_att(rows2, ps2)
    print(f"1:1 매칭(ATT) {att2:.2f}, 사용된 비처치 {nu2}명")
    t2, n2, tau2 = trimmed_ipw(rows2, ps2)
    print(f"성향점수 0.1~0.9만 남긴 IPW {t2:.2f} (남은 {n2}명, 이 집단의 진짜 평균 효과 {tau2:.2f})")
    print(f"최대 가중치 {max(w2):.1f}, 유효 표본 크기 {effective_sample_size(w2):.0f}")
    wt = [(r, wi) for r, wi in zip(rows2, w2) if r["t"] == 1]
    wc = [(r, wi) for r, wi in zip(rows2, w2) if r["t"] == 0]
    print(f"가중 평균 활동 점수: 쿠폰 쪽 {weighted_mean_x(wt):.2f}, 못 받은 쪽 {weighted_mean_x(wc):.2f}, 전체 고객 {mean([r['x'] for r in rows2]):.2f}")
    top = max(wt, key=lambda a: a[1])
    print(f"가장 큰 가중치({top[1]:.1f})를 받은 쿠폰 고객: 활동 점수 {top[0]['x']:.2f}, 쿠폰 쪽 가중치 합의 {top[1] / sum(wi for _, wi in wt):.0%}")
    low = [r for r in rows2 if r["x"] < 0]
    print(f"활동 점수 0 미만 고객 {len(low)}명 중 쿠폰을 받은 고객 {sum(r['t'] for r in low)}명")

    print("\n[같은 설정을 200번 반복] n=2000, seed 100부터")
    for name, kws in (("겹침 충분", {}), ("겹침 부족", kw)):
        res = replicate(200, 2000, 100, **kws)
        ipw = [r[1] for r in res]
        tr = [r[2] for r in res]
        err = [a - b for a, b in zip(ipw, tr)]
        print(f"{name}: IPW 평균 오차 {mean(err):+.2f}, 표준편차 {sd(ipw):.2f}, "
              f"최대 가중치 평균 {mean([r[3] for r in res]):.1f}, 유효 표본 평균 {mean([r[4] for r in res]):.0f}")


if __name__ == "__main__":
    main()
