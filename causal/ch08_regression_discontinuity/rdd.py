"""8장 실습: 회귀불연속(RDD). 표준 라이브러리만 사용합니다.

가상 온라인 마켓: 전월 구매액(x, 만 원)이 10 이상이면 쿠폰을 줍니다.
다음 달 주문액(Y, 천 원, 가상)에 쿠폰이 만드는 진짜 효과는 +10.0으로 정해 두었어요.

무엇을 재는가: 기준선(CUT = 10) 바로 위·아래 고객은 거의 같다는 가정으로, 기준선에서 결과가 뛰는 크기(점프)를 효과로 읽는다.
단순 비교, 전체 직선 하나, 기준선 양쪽의 국소 선형(대역폭 h 안의 고객만 사용)을 비교하고, 대역폭에 따른 편향-분산 균형을 반복 실험으로 본다.
또 기준선 직전 고객이 x를 올려 넘어가는 조작(manipulate=True)이 있을 때 추정이 어떻게 틀어지는지 본다.
입력: make_data(n, seed, manipulate)의 행(x: 전월 구매액 만 원, d: 쿠폰, y: 다음 달 주문액 천 원, moved: 조작 여부). 출력: 점프 추정값과 구간별 인원.
테스트의 기대값(손계산 점프 10, 단순 비교 19.44, 대역폭별 추정 등)이 장 본문의 표와 대응한다.
"""
import math
import random

CUT = 10.0
TAU = 10.0
XMEAN = 8.0  # 대부분의 고객은 기준선보다 아래
CURV = 0.3   # 쿠폰이 없을 때의 곡선 정도


def normal(rng):
    """Box-Muller 표준정규 한 개."""
    u1, u2 = rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


# ---------- 손계산: 구간 평균 4개 ----------
# 기준선 양쪽에서 직선을 기준선(x=10)까지 외삽해 점프를 읽는다. 두 구간의 직접 차이(14)는 기울기 몫 4가 섞여 부풀려진다
# 구간 가운데 x와 다음 달 구매액 평균(천 원, 가상)
HAND_BELOW = [(8.5, 44.0), (9.5, 48.0)]
HAND_ABOVE = [(10.5, 62.0), (11.5, 66.0)]


def line_through(points):
    """두 점을 지나는 직선의 (기울기, 절편)."""
    (x0, y0), (x1, y1) = points
    slope = (y1 - y0) / (x1 - x0)
    return slope, y0 - slope * x0


def hand_rdd():
    """구간 평균 4개로 아래·위 직선을 각각 세워 기준선에서의 값 차이(점프)와 인접 구간 평균 차이를 비교한다."""
    sb, ib = line_through(HAND_BELOW)
    sa, ia = line_through(HAND_ABOVE)
    below, above = ib + sb * CUT, ia + sa * CUT
    adjacent = HAND_ABOVE[0][1] - HAND_BELOW[1][1]  # 10.5 구간 62 - 9.5 구간 48 = 14 = 점프 10 + 기울기 4 x 구간 중심 간격 1
    return {"slope_below": sb, "slope_above": sa, "below_at_cut": below, "above_at_cut": above,
            "jump": above - below, "adjacent_bin_gap": adjacent}


# ---------- 데이터 ----------
def make_data(n, seed, manipulate=False):
    """x ~ N(8, 3^2). Y0 = 40 + 2.5(x-10) + 0.3(x-10)^2 + 4u + 잡음. u는 눈에 안 보이는 구매 의욕.
    manipulate=True: x가 9.0~10 미만이고 의욕이 높은(u>0.5) 고객 중 80%가 10.0~10.3으로 채워서 올린다.
    조작은 기준선 위쪽에 의욕 높은 고객을 몰아 넣어 "기준선 근처 고객은 비슷하다"는 가정을 깨뜨린다."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = XMEAN + 3.0 * normal(rng)
        u = normal(rng)  # 관찰되지 않는 구매 의욕. y에 4u로 들어가 조작 시 위쪽 평균 y를 올린다
        noise = 6.0 * normal(rng)
        coin, jitter = rng.random(), rng.random()  # 올리지 않는 경우에도 항상 뽑아 두 판이 같은 고객을 쓴다
        moved = False
        if manipulate and 9.0 <= x < CUT and u > 0.5 and coin < 0.8:
            x = CUT + 0.3 * jitter
            moved = True
        d = 1 if x >= CUT else 0  # 기준선 이상이면 쿠폰(정확히 10.0은 쿠폰 쪽)
        z = x - CUT  # 기준선을 0으로 옮긴 값. 이렇게 해야 직선의 절편이 곧 기준선에서의 값
        # 쿠폰이 없을 때의 곡선(2.5z + 0.3z^2)은 양쪽에서 이어지고, 불연속은 TAU * d 하나뿐이다
        y = 40.0 + 2.5 * z + CURV * z * z + 4.0 * u + noise + TAU * d
        rows.append({"x": x, "d": d, "y": y, "moved": moved})
    return rows


# ---------- 추정 ----------
def wls_line(points):
    """최소제곱 직선. points: (x, y). 반환 (절편, 기울기) 단, x는 이미 cutoff 기준으로 옮겨 둔다.
    함수 이름은 wls지만 가중치 없는 보통 최소제곱이다(사각 커널: 대역폭 안은 모두 같은 가중치)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)  # 단순회귀 닫힌 해: Cov(x, y) / Var(x)의 합계 형태
    return (sy - slope * sx) / n, slope


def naive_gap(rows):
    """기준선 위 평균 - 아래 평균. 멀리 있는 고객(구매액이 크게 다른)까지 섞여 효과 10을 크게 넘는다."""
    a = [r["y"] for r in rows if r["d"] == 1]
    b = [r["y"] for r in rows if r["d"] == 0]
    return sum(a) / len(a) - sum(b) / len(b)


def local_linear(rows, h):
    """기준선 양쪽 h 안의 고객만 써서 양쪽에 따로 직선을 맞추고 기준선에서의 차이를 잰다."""
    # 구간 경계: 왼쪽은 [CUT-h, CUT), 오른쪽은 [CUT, CUT+h]. x == CUT는 오른쪽(쿠폰 쪽)에 들어간다
    left = [(r["x"] - CUT, r["y"]) for r in rows if CUT - h <= r["x"] < CUT]
    right = [(r["x"] - CUT, r["y"]) for r in rows if CUT <= r["x"] <= CUT + h]
    a0, _ = wls_line(left)   # 절편 = 기준선(z=0)에서 아래쪽 직선의 값
    a1, _ = wls_line(right)  # 기울기는 양쪽 따로 맞추므로 쓰지 않는다
    # 반환: (점프 추정, 왼쪽 인원, 오른쪽 인원). h가 작을수록 편향은 줄지만 인원이 줄어 흔들림이 커진다
    return a1 - a0, len(left), len(right)


def global_linear_common_slope(rows):
    """전체 고객에 기울기 하나를 공유하는 직선 + 쿠폰 점프. 곡선을 못 따라간다."""
    n = len(rows)  # (사용하지 않음)
    # y = a + b z + c d  의 정규방정식. c가 점프. 곡선(z^2 항)을 직선으로 근사하므로 기준선 양쪽 곡률이 점프에 섞인다
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
    return solve3(a, b)[2]  # 계수 순서 [a, b, c] 중 점프 c


def solve3(a, b):
    """3x3 연립방정식을 부분 피벗팅 가우스 소거법으로 푼다."""
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
    """[lo, hi)를 width 폭으로 나눠 (구간 왼쪽 끝, 고객 수)를 센다. 기준선 위쪽만 불쑥 높으면 조작(bunching) 의심 신호."""
    k = int(round((hi - lo) / width))
    cnt = [0] * k
    for r in rows:
        if lo <= r["x"] < hi:
            cnt[int((r["x"] - lo) // width)] += 1  # 부동소수점 경계값(예: 정확히 10.0)은 오른쪽 구간으로 들어간다
    return [(lo + i * width, cnt[i]) for i in range(k)]


def mean(v):
    """산술평균."""
    return sum(v) / len(v)


def sd(v):
    """표본표준편차(분모 n - 1)."""
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


# 비교할 대역폭(만 원). 너무 좁으면 인원이 적어 흔들리고, 너무 넓으면 곡선 편향이 커진다
BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def replicate(reps, n, seed0, manipulate=False):
    """시드 seed0부터 reps번 새 표본으로 대역폭별 점프 추정값 목록을 모은다."""
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
