"""3장 실습: 무엇을 통제해야 하나. 교란·매개·충돌변수. 모든 숫자는 가상(예시용)이다.

구조(화살표는 '원인이다'):
  Z(충성도) -> T(쿠폰), Z -> Y(월 구매액)
  T -> M(장바구니 크기) -> Y,  T -> Y(직접)
  T -> C(후기 점수) <- Y
쿠폰이 구매액에 주는 전체 효과 = 직접 0.5 + 매개 1.5*1.0 = 2.0 (진짜 값).
난수는 random.Random(seed).random()만 쓴다.

무엇을 재는가: 같은 데이터에 회귀로 "무엇을 통제 변수로 넣느냐"에 따라 쿠폰(T) 계수가 어떻게 달라지는지 본다.
교란(Z)은 넣어야 하고, 매개(M)를 넣으면 전체 효과 중 매개 경로가 사라지며, 충돌변수(C)를 넣으면 없던 연관이 생긴다.
입력: make_data(n, seed, randomize)가 만드는 행 목록(dict: z, t, m, y, c). 출력: 통제 조합별 T 계수 표.
테스트의 기대값(5.00, 1.99, 0.52, -0.51, -0.72)이 장 본문의 통제 변수 표와 대응한다. 변수 단위는 모두 가상의 임의 단위다.
"""
import math
import random

# 진짜 값: 전체 효과 = 직접 효과 + 매개 경로(T->M 1.5 x M->Y 1.0)
TRUE_TOTAL = 2.0
DIRECT = 0.5
T_TO_M = 1.5
M_TO_Y = 1.0


def normal(rng, sd=1.0):
    """Box-Muller 표준정규 x sd."""
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_data(n=20000, seed=31, randomize=False):
    """구조 방정식대로 z -> t -> m -> y -> c 순서로 값을 만든다. randomize=True면 t를 z와 무관한 동전으로 정한다."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = normal(rng)
        if randomize:
            # 동전 배정이면 Z -> T 화살표가 끊겨 Z를 통제하지 않아도 교란이 없다
            t = 1.0 if rng.random() < 0.5 else 0.0
        else:
            # 충성도 z가 높을수록 쿠폰을 받는다(Z -> T). 이 화살표가 교란 경로를 연다
            t = 1.0 if (0.8 * z + normal(rng)) > 0 else 0.0
        m = T_TO_M * t + normal(rng)  # 매개: 쿠폰이 장바구니 크기를 키운다
        # Z가 y를 직접 올리는 계수 3.0이 교란의 크기를 정한다
        y = 10.0 + 3.0 * z + DIRECT * t + M_TO_Y * m + normal(rng, 2.0)
        # 충돌변수: T와 Y가 모두 C의 원인이다(결과의 결과). C를 조건에 넣으면 T와 Y 사이에 가짜 연관이 생긴다
        c = 1.0 * t + 1.0 * y + normal(rng)
        rows.append({"z": z, "t": t, "m": m, "y": y, "c": c})
    return rows


def ols(rows, y_name, x_names):
    """정규방정식 (X'X) b = X'y 를 가우스 소거법으로 푼다. 계수 dict(절편 포함)를 돌려준다."""
    cols = ["1"] + list(x_names)  # "1"은 절편 열
    k = len(cols)
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for r in rows:
        x = [1.0] + [r[nm] for nm in x_names]
        for i in range(k):
            xty[i] += x[i] * r[y_name]
            for j in range(k):
                xtx[i][j] += x[i] * x[j]
    a = [xtx[i][:] + [xty[i]] for i in range(k)]  # 첨가행렬 [X'X | X'y]
    for i in range(k):
        # 부분 피벗팅: 열에서 절댓값이 가장 큰 행을 위로 올려 0 나누기와 오차 확대를 줄인다
        p = max(range(i, k), key=lambda r_: abs(a[r_][i]))
        a[i], a[p] = a[p], a[i]
        for r_ in range(i + 1, k):
            f = a[r_][i] / a[i][i]
            for c_ in range(i, k + 1):
                a[r_][c_] -= f * a[i][c_]
    b = [0.0] * k
    for i in range(k - 1, -1, -1):  # 후진 대입. 위 반복의 r_, c_는 행 변수 r과 겹치지 않게 붙인 이름
        b[i] = (a[i][k] - sum(a[i][j] * b[j] for j in range(i + 1, k))) / a[i][i]
    return dict(zip(cols, b))


# ---------- 손계산 표: 충성도 층별로 나눠 비교한다(가상 1,000명) ----------
# (층, 쿠폰 받음 여부, 인원, 평균 월 구매액(만원))
HAND_CELLS = [
    ("높음", 1, 600, 32.0), ("높음", 0, 100, 30.0),
    ("낮음", 1, 100, 14.0), ("낮음", 0, 200, 12.0),
]


def hand_stratified():
    """손계산 표에서 단순 차이와 층별 차이의 층 크기 가중 평균(조정 후 효과)을 구한다."""
    def pooled(t):
        cells = [c for c in HAND_CELLS if c[1] == t]
        return sum(c[2] * c[3] for c in cells) / sum(c[2] for c in cells)
    naive = pooled(1) - pooled(0)  # 층을 무시한 단순 차이: 충성도 높은 사람이 쿠폰을 더 받아 부풀려진다
    within, weights = {}, {}
    total = sum(c[2] for c in HAND_CELLS)
    for layer in ("높음", "낮음"):
        a = next(c for c in HAND_CELLS if c[0] == layer and c[1] == 1)
        b = next(c for c in HAND_CELLS if c[0] == layer and c[1] == 0)
        within[layer] = a[3] - b[3]
        weights[layer] = (a[2] + b[2]) / total
    # 층 안에서는 어디서나 차이가 2.0이라 어떤 가중을 써도 2.0이다(층별 효과가 같다는 손계산 설정)
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
    """통제 조합마다 y ~ t + 통제변수 회귀를 돌려 T 계수만 모은다."""
    rows = make_data(randomize=randomize)
    return [(label, ols(rows, "y", ["t"] + s)["t"]) for label, s in ADJUSTMENT_SETS]


def seed_spread(seeds=range(40, 52)):
    """시드만 바꿔 같은 장난감을 12번 만들 때 쿠폰 계수가 얼마나 흔들리는지(표본 흔들림) 본다."""
    out = {}
    for randomize in (False, True):
        for label, names in (("아무것도 안 넣음", []), ("Z + 장바구니 M", ["z", "m"])):
            # 주의: 지역 변수 sd는 이 함수 안에서만 쓰는 시드 값이다(통계 함수 sd가 아님)
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
