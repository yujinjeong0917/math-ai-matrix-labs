"""4장: 국소 최적과 전역 최적. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용.

운동화 한 켤레 가격 p(천원)를 고른다. 수요는 두 고객 묶음의 합이다.
 - 가격에 덜 민감한 30명: 100천원 근처까지 산다
 - 가격에 민감한 70명: 45천원 근처까지 산다
원가는 20천원. 이익 f(p) = (p - 20) * 수요(p). 천원 단위, 한 주 기준 가상값.

재는 것: 이익 곡선의 봉우리가 둘이라 시작점에 따라 경사 상승이 다른 곳에 멈추는 현상.
입력: 시작 가격(천원). 출력: 멈춘 가격, 이익, 반복 횟수.
대응(README 기대 출력): 국소 최대 (41.58, 1707.22)와 (86.16, 1805.19), 골짜기 56.26.
"""
import math

COST = 20.0


def sigmoid(z):
    # 가격이 오를수록 구매자가 줄도록 (경계가격 - p)를 넣어 쓴다. 분모 6, 4는 민감도(모형의 선택)
    return 1.0 / (1.0 + math.exp(-z))


def demand(p):
    # 30명 x (100천원 근처까지 구매) + 70명 x (45천원 근처까지 구매). 수요는 연속값(명)으로 둔 단순화
    return 30 * sigmoid((100 - p) / 6) + 70 * sigmoid((45 - p) / 4)


def profit(p):
    return (p - COST) * demand(p)


def grad(p, h=1e-5):
    # 중심차분 미분. 해석 미분 대신 수치 미분을 써 모형을 바꿔도 코드를 고치지 않는다
    return (profit(p + h) - profit(p - h)) / (2 * h)


def curvature(p, h=1e-3):
    # 2계 중심차분. 음수면 국소 최대, 양수면 국소 최소. h가 너무 작으면 반올림 오차가 커져 1e-3으로 둔다
    return (profit(p + h) - 2 * profit(p) + profit(p - h)) / h ** 2


def valley(lo=45.0, hi=80.0, step=0.01):
    """두 봉우리 사이에서 이익이 가장 낮은 지점."""
    # 두 봉우리(약 41.6, 86.2) 사이 구간 [45, 80]의 최솟값
    n = round((hi - lo) / step)
    return min(((lo + i * step, profit(lo + i * step)) for i in range(n + 1)), key=lambda t: t[1])


def grid_scan(lo=20.0, hi=120.0, step=0.01):
    n = round((hi - lo) / step)
    pts = [(lo + i * step, profit(lo + i * step)) for i in range(n + 1)]
    # 국소 최대: 왼쪽보다 엄격히 크고 오른쪽 이상(평평한 꼭대기를 한 번만 세려는 비대칭 비교)
    local = [pts[i] for i in range(1, n)
             if pts[i][1] > pts[i - 1][1] and pts[i][1] >= pts[i + 1][1]]
    return pts, local


def ascend(p0, lo=20.0, hi=120.0, step=1.0, tol=1e-6, max_iter=100000):
    """경사 상승. 한 번에 움직이는 폭을 step으로 막고, 값이 줄면 폭을 절반으로 줄인다."""
    # 기울기가 0에 가까우면 정류점. 정류점은 국소 최대일 수도, 골짜기일 수도 있다
    p, it = p0, 0
    while it < max_iter:
        g = grad(p)
        if abs(g) < tol:
            break
        s = step
        while True:
            # 기울기 부호 방향으로 이동하되 [lo, hi] 밖으로 나가지 않게 자른다
            q = min(hi, max(lo, p + s * (1 if g > 0 else -1)))
            if profit(q) >= profit(p) or s < 1e-12:
                break
            s /= 2
        if q == p:   # 더 움직일 수 없다(경계이거나 폭이 바닥남)
            break
        p = q
        it += 1
    return p, profit(p), it


def multi_start(starts):
    # 여러 시작점에서 각각 올라 가장 높은 이익을 고른다. 전역 최적 보장은 아니고 시작점이 봉우리를 덮을 때만 유효
    runs = [ascend(s) for s in starts]
    return runs, max(runs, key=lambda r: r[1])


if __name__ == "__main__":
    pts, local = grid_scan()
    print("격자(0.01)에서 찾은 국소 최대:", [(round(p, 2), round(v, 2)) for p, v in local])
    best = max(pts, key=lambda t: t[1])
    print("격자 전역 최대:", round(best[0], 2), round(best[1], 2))
    for s in (25, 40, 60, 80, 110):
        p, v, it = ascend(s)
        print(f"시작 {s} -> p={p:.2f} 이익={v:.2f} 반복={it}")
    runs, b = multi_start([25, 40, 60, 80, 110])
    print("다중 시작 최선:", round(b[0], 2), round(b[1], 2))
    vp, vv = valley()
    print("골짜기:", round(vp, 2), round(vv, 2))
    print("곡률(2계 미분) 41.58, 86.16:", round(curvature(41.58), 3), round(curvature(86.16), 3))
    print("p=50 부근 기울기:", round(grad(50), 3), " p=75 부근 기울기:", round(grad(75), 3))
    print("수요(p=60), 수요(p=95):", round(demand(60), 2), round(demand(95), 2))
