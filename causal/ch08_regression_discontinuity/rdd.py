"""8장 실습: 회귀불연속(RDD). 표준 라이브러리만 사용합니다.

가상 온라인 마켓: 전월 구매액(x, 만 원)이 10 이상이면 쿠폰을 줍니다.
다음 달 주문액(Y, 천 원, 가상)에 쿠폰이 만드는 진짜 효과는 +10.0으로 정해 두었어요.
"""
import math
import random

CUT = 10.0
TAU = 10.0
XMEAN = 8.0  # 대부분의 고객은 기준선보다 아래
CURV = 0.3   # 쿠폰이 없을 때의 곡선 정도


def normal(rng):
    u1, u2 = rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


# ---------- 손계산: 구간 평균 4개 ----------
# 구간 가운데 x와 다음 달 구매액 평균(천 원, 가상)
HAND_BELOW = [(8.5, 44.0), (9.5, 48.0)]
HAND_ABOVE = [(10.5, 62.0), (11.5, 66.0)]


def line_through(points):
    (x0, y0), (x1, y1) = points
    slope = (y1 - y0) / (x1 - x0)
    return slope, y0 - slope * x0


def hand_rdd():
    sb, ib = line_through(HAND_BELOW)
    sa, ia = line_through(HAND_ABOVE)
    below, above = ib + sb * CUT, ia + sa * CUT
    adjacent = HAND_ABOVE[0][1] - HAND_BELOW[1][1]
    return {"slope_below": sb, "slope_above": sa, "below_at_cut": below, "above_at_cut": above,
            "jump": above - below, "adjacent_bin_gap": adjacent}


# ---------- 데이터 ----------
def make_data(n, seed, manipulate=False):
    """x ~ N(8, 3^2). Y0 = 40 + 2.5(x-10) + 0.3(x-10)^2 + 4u + 잡음. u는 눈에 안 보이는 구매 의욕.
    manipulate=True: x가 9.0~10 미만이고 의욕이 높은(u>0.5) 고객 중 80%가 10.0~10.3으로 채워서 올린다."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = XMEAN + 3.0 * normal(rng)
        u = normal(rng)
        noise = 6.0 * normal(rng)
        coin, jitter = rng.random(), rng.random()  # 올리지 않는 경우에도 항상 뽑아 두 판이 같은 고객을 쓴다
        moved = False
        if manipulate and 9.0 <= x < CUT and u > 0.5 and coin < 0.8:
            x = CUT + 0.3 * jitter
            moved = True
        d = 1 if x >= CUT else 0
        z = x - CUT
        y = 40.0 + 2.5 * z + CURV * z * z + 4.0 * u + noise + TAU * d
        rows.append({"x": x, "d": d, "y": y, "moved": moved})
    return rows


# ---------- 추정 ----------
def wls_line(points):
    """최소제곱 직선. points: (x, y). 반환 (절편, 기울기) 단, x는 이미 cutoff 기준으로 옮겨 둔다."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, slope


def naive_gap(rows):
    a = [r["y"] for r in rows if r["d"] == 1]
    b = [r["y"] for r in rows if r["d"] == 0]
    return sum(a) / len(a) - sum(b) / len(b)


def local_linear(rows, h):
    """기준선 양쪽 h 안의 고객만 써서 양쪽에 따로 직선을 맞추고 기준선에서의 차이를 잰다."""
    left = [(r["x"] - CUT, r["y"]) for r in rows if CUT - h <= r["x"] < CUT]
    right = [(r["x"] - CUT, r["y"]) for r in rows if CUT <= r["x"] <= CUT + h]
    a0, _ = wls_line(left)
    a1, _ = wls_line(right)
    return a1 - a0, len(left), len(right)


def global_linear_common_slope(rows):
    """전체 고객에 기울기 하나를 공유하는 직선 + 쿠폰 점프. 곡선을 못 따라간다."""
    n = len(rows)
    # y = a + b z + c d  의 정규방정식
    s = {k: 0.0 for k in ("1", "z", "d", "zz", "zd", "dd", "y", "zy", "dy")}
    for r in rows:
        z, d, y = r["x"] - CUT, r["d"], r["y"]
        s["1"] += 1
        s["z"] += z
        s["d"] += d
        s["zz"] += z * z
        s["zd"] += z * d
        s["dd"] += d * d
        s["y"] += y
        s["zy"] += z * y
        s["dy"] += d * y
    a = [[s["1"], s["z"], s["d"]], [s["z"], s["zz"], s["zd"]], [s["d"], s["zd"], s["dd"]]]
    b = [s["y"], s["zy"], s["dy"]]
    return solve3(a, b)[2]


def solve3(a, b):
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 3
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


def bin_counts(rows, lo=8.0, hi=12.0, width=0.5):
    k = int(round((hi - lo) / width))
    cnt = [0] * k
    for r in rows:
        if lo <= r["x"] < hi:
            cnt[int((r["x"] - lo) // width)] += 1
    return [(lo + i * width, cnt[i]) for i in range(k)]


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def replicate(reps, n, seed0, manipulate=False):
    res = {h: [] for h in BANDWIDTHS}
    for k in range(reps):
        rows = make_data(n, seed0 + k, manipulate)
        for h in BANDWIDTHS:
            res[h].append(local_linear(rows, h)[0])
    return res


def main():
    h = hand_rdd()
    print("[손계산: 구간 평균 4개]")
    print(f"기준선 아래 기울기 {h['slope_below']:.1f}/만 원, 위 기울기 {h['slope_above']:.1f}/만 원")
    print(f"기준선(10)에서 아래쪽 직선 {h['below_at_cut']:.1f}, 위쪽 직선 {h['above_at_cut']:.1f}, 점프 {h['jump']:.1f}")
    print(f"바로 붙은 두 구간 평균 차이 {h['adjacent_bin_gap']:.1f}")

    rows = make_data(6000, 21)
    print(f"\n[한 번 돌린 결과] n=6000, seed=21, 쿠폰 {sum(r['d'] for r in rows)}명, 진짜 효과 {TAU:.1f}")
    print(f"단순 비교(위 평균 - 아래 평균) {naive_gap(rows):.2f}")
    print(f"전체에 직선 하나(기울기 공유) {global_linear_common_slope(rows):.2f}")
    for bw in BANDWIDTHS:
        est, nl, nr = local_linear(rows, bw)
        print(f"대역폭 {bw}: 국소 선형 {est:.2f} (아래 {nl}명, 위 {nr}명)")

    print("\n[200번 반복] n=6000, seed 1000부터. 진짜 값 대비 평균 오차 / 표준편차")
    res = replicate(200, 6000, 1000)
    for bw in BANDWIDTHS:
        print(f"대역폭 {bw}: 평균 오차 {mean(res[bw]) - TAU:+.2f}, 표준편차 {sd(res[bw]):.2f}, "
              f"합친 크기 {math.hypot(mean(res[bw]) - TAU, sd(res[bw])):.2f}")

    print("\n[기준선 직전에 채워서 올린 고객이 있을 때] 같은 설정, seed=21")
    rm = make_data(6000, 21, manipulate=True)
    moved = sum(1 for r in rm if r["moved"])
    print(f"올려서 넘어간 고객 {moved}명")
    print("0.5 만 원 폭 구간별 고객 수 (왼쪽 끝 x부터):")
    cm, c0 = bin_counts(rm), bin_counts(rows)
    for (x, a), (_, b) in zip(c0, cm):
        print(f"  [{x:.1f}, {x + 0.5:.1f}) 원래 {a}명 -> 올린 뒤 {b}명")
    for bw in (0.5, 1.0, 2.0):
        est, nl, nr = local_linear(rm, bw)
        print(f"대역폭 {bw}: 국소 선형 {est:.2f} (아래 {nl}명, 위 {nr}명)")
    resm = replicate(200, 6000, 1000, manipulate=True)
    for bw in (0.5, 1.0, 2.0):
        print(f"올린 경우 200번 반복, 대역폭 {bw}: 평균 오차 {mean(resm[bw]) - TAU:+.2f}, 표준편차 {sd(resm[bw]):.2f}")


if __name__ == "__main__":
    main()
