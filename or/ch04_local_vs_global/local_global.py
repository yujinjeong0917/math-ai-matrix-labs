"""4장: 국소 최적과 전역 최적. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용.

운동화 한 켤레 가격 p(천원)를 고른다. 수요는 두 고객 묶음의 합이다.
 - 가격에 덜 민감한 30명: 100천원 근처까지 산다
 - 가격에 민감한 70명: 45천원 근처까지 산다
원가는 20천원. 이익 f(p) = (p - 20) * 수요(p). 천원 단위, 한 주 기준 가상값.
"""
import math

COST = 20.0


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def demand(p):
    return 30 * sigmoid((100 - p) / 6) + 70 * sigmoid((45 - p) / 4)


def profit(p):
    return (p - COST) * demand(p)


def grad(p, h=1e-5):
    return (profit(p + h) - profit(p - h)) / (2 * h)


def curvature(p, h=1e-3):
    return (profit(p + h) - 2 * profit(p) + profit(p - h)) / h ** 2


def valley(lo=45.0, hi=80.0, step=0.01):
    """두 봉우리 사이에서 이익이 가장 낮은 지점."""
    n = round((hi - lo) / step)
    return min(((lo + i * step, profit(lo + i * step)) for i in range(n + 1)), key=lambda t: t[1])


def grid_scan(lo=20.0, hi=120.0, step=0.01):
    n = round((hi - lo) / step)
    pts = [(lo + i * step, profit(lo + i * step)) for i in range(n + 1)]
    local = [pts[i] for i in range(1, n)
             if pts[i][1] > pts[i - 1][1] and pts[i][1] >= pts[i + 1][1]]
    return pts, local


def ascend(p0, lo=20.0, hi=120.0, step=1.0, tol=1e-6, max_iter=100000):
    """경사 상승. 한 번에 움직이는 폭을 step으로 막고, 값이 줄면 폭을 절반으로 줄인다."""
    p, it = p0, 0
    while it < max_iter:
        g = grad(p)
        if abs(g) < tol:
            break
        s = step
        while True:
            q = min(hi, max(lo, p + s * (1 if g > 0 else -1)))
            if profit(q) >= profit(p) or s < 1e-12:
                break
            s /= 2
        if q == p:
            break
        p = q
        it += 1
    return p, profit(p), it


def multi_start(starts):
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
