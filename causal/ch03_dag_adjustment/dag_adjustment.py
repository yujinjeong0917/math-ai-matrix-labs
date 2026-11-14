"""3장 실습: 무엇을 통제해야 하나. 교란·매개·충돌변수. 모든 숫자는 가상(예시용)이다.

구조(화살표는 '원인이다'):
  Z(충성도) -> T(쿠폰), Z -> Y(월 구매액)
  T -> M(장바구니 크기) -> Y,  T -> Y(직접)
  T -> C(후기 점수) <- Y
쿠폰이 구매액에 주는 전체 효과 = 직접 0.5 + 매개 1.5*1.0 = 2.0 (진짜 값).
난수는 random.Random(seed).random()만 쓴다.
"""
import math
import random

TRUE_TOTAL = 2.0
DIRECT = 0.5
T_TO_M = 1.5
M_TO_Y = 1.0


def normal(rng, sd=1.0):
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_data(n=20000, seed=31, randomize=False):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = normal(rng)
        if randomize:
            t = 1.0 if rng.random() < 0.5 else 0.0
        else:
            t = 1.0 if (0.8 * z + normal(rng)) > 0 else 0.0
        m = T_TO_M * t + normal(rng)
        y = 10.0 + 3.0 * z + DIRECT * t + M_TO_Y * m + normal(rng, 2.0)
        c = 1.0 * t + 1.0 * y + normal(rng)
        rows.append({"z": z, "t": t, "m": m, "y": y, "c": c})
    return rows


def ols(rows, y_name, x_names):
    """정규방정식 (X'X) b = X'y 를 가우스 소거법으로 푼다. 계수 dict(절편 포함)를 돌려준다."""
    cols = ["1"] + list(x_names)
    k = len(cols)
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for r in rows:
        x = [1.0] + [r[nm] for nm in x_names]
        for i in range(k):
            xty[i] += x[i] * r[y_name]
            for j in range(k):
                xtx[i][j] += x[i] * x[j]
    a = [xtx[i][:] + [xty[i]] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r_: abs(a[r_][i]))
        a[i], a[p] = a[p], a[i]
        for r_ in range(i + 1, k):
            f = a[r_][i] / a[i][i]
            for c_ in range(i, k + 1):
                a[r_][c_] -= f * a[i][c_]
    b = [0.0] * k
    for i in range(k - 1, -1, -1):
        b[i] = (a[i][k] - sum(a[i][j] * b[j] for j in range(i + 1, k))) / a[i][i]
    return dict(zip(cols, b))


# ---------- 손계산 표: 충성도 층별로 나눠 비교한다(가상 1,000명) ----------
# (층, 쿠폰 받음 여부, 인원, 평균 월 구매액(만원))
HAND_CELLS = [
    ("높음", 1, 600, 32.0), ("높음", 0, 100, 30.0),
    ("낮음", 1, 100, 14.0), ("낮음", 0, 200, 12.0),
]


def hand_stratified():
    def pooled(t):
        cells = [c for c in HAND_CELLS if c[1] == t]
        return sum(c[2] * c[3] for c in cells) / sum(c[2] for c in cells)
    naive = pooled(1) - pooled(0)
    within, weights = {}, {}
    total = sum(c[2] for c in HAND_CELLS)
    for layer in ("높음", "낮음"):
        a = next(c for c in HAND_CELLS if c[0] == layer and c[1] == 1)
        b = next(c for c in HAND_CELLS if c[0] == layer and c[1] == 0)
        within[layer] = a[3] - b[3]
        weights[layer] = (a[2] + b[2]) / total
    adjusted = sum(within[k] * weights[k] for k in within)
    return {"treated_mean": pooled(1), "control_mean": pooled(0), "naive": naive,
            "within": within, "weights": weights, "adjusted": adjusted}


ADJUSTMENT_SETS = [
    ("아무것도 안 넣음", []),
    ("충성도 Z", ["z"]),
    ("Z + 장바구니 M", ["z", "m"]),
    ("Z + 후기 점수 C", ["z", "c"]),
    ("전부 다(Z, M, C)", ["z", "m", "c"]),
]


def adjustment_table(randomize=False):
    rows = make_data(randomize=randomize)
    return [(label, ols(rows, "y", ["t"] + s)["t"]) for label, s in ADJUSTMENT_SETS]


def seed_spread(seeds=range(40, 52)):
    """시드만 바꿔 같은 장난감을 12번 만들 때 쿠폰 계수가 얼마나 흔들리는지(표본 흔들림) 본다."""
    out = {}
    for randomize in (False, True):
        for label, names in (("아무것도 안 넣음", []), ("Z + 장바구니 M", ["z", "m"])):
            vals = [ols(make_data(seed=sd, randomize=randomize), "y", ["t"] + names)["t"] for sd in seeds]
            m = sum(vals) / len(vals)
            sd_ = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            out[(randomize, label)] = (m, sd_)
    return out


if __name__ == "__main__":
    h = hand_stratified()
    print("손계산: 받은 사람 평균 %.2f / 못 받은 사람 평균 %.2f / 단순 차이 %.2f / 층 안 차이 %s / 층 가중 평균 %.2f" % (
        h["treated_mean"], h["control_mean"], h["naive"], h["within"], h["adjusted"]))
    print("진짜 전체 효과 %.1f (직접 %.1f + 매개 %.1f*%.1f)" % (TRUE_TOTAL, DIRECT, T_TO_M, M_TO_Y))
    print("쿠폰을 충성도 높은 사람이 더 받는 관찰 데이터:")
    for label, b in adjustment_table():
        print("  %-18s T 계수 %.2f" % (label, b))
    print("쿠폰을 동전으로 배정한 데이터:")
    for label, b in adjustment_table(randomize=True):
        print("  %-18s T 계수 %.2f" % (label, b))
    print("시드를 바꿔 12번 만든 평균(표준편차):")
    for (rnd, label), (m, sd_) in seed_spread().items():
        print("  %-8s %-18s %.2f (%.2f)" % ("동전 배정" if rnd else "관찰", label, m, sd_))
