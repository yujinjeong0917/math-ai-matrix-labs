"""2장: 쌍대 변수(그림자가격)와 민감도. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용.

첫 번째 길: 재봉 시간 b1을 바꿔 가며 다시 풀어 최적값 곡선과 기울기를 본다.
두 번째 길: 쌍대 인증서 y를 직접 확인한다(A^T y >= c, y >= 0, 쓰고 남는 제약의 y = 0, b.y = 주문제 최적값).
"""
from fractions import Fraction as F
import lp2d

C = lp2d.C


def constraints(b1=100, b2=80, b3=40):
    return [((2, 1), b1), ((1, 1), b2), ((1, 0), b3)]


def optimum(b1=100, b2=80, b3=40):
    return lp2d.solve(constraints(b1, b2, b3))


def finite_difference(which, h=1):
    """제약 which(0:재봉 1:포장 2:주문한도)의 우변을 h만큼 늘렸을 때 최적값 변화."""
    b = [100, 80, 40]
    base = optimum(*b)[1]
    b[which] += h
    return optimum(*b)[1] - base


def check_dual_certificate(y, constraints_=None):
    cons = constraints_ or constraints()
    y = [F(v) for v in y]
    A = [[F(a) for a in row] for row, _ in cons]
    b = [F(bb) for _, bb in cons]
    feasible = all(v >= 0 for v in y) and all(
        sum(A[i][j] * y[i] for i in range(len(y))) >= C[j] for j in range(2))
    x, z = lp2d.solve(cons)
    slack = [b[i] - (A[i][0] * x[0] + A[i][1] * x[1]) for i in range(len(y))]
    comp = all(y[i] == 0 or slack[i] == 0 for i in range(len(y)))
    dual_value = sum(b[i] * y[i] for i in range(len(y)))
    return feasible, comp, dual_value, z


def value_curve(b1_values, which=0):
    out = []
    for v in b1_values:
        b = [100, 80, 40]
        b[which] = v
        out.append((v, optimum(*b)[1]))
    return out


def slopes(curve):
    return [(curve[i][0], curve[i + 1][0], (curve[i + 1][1] - curve[i][1]) / (curve[i + 1][0] - curve[i][0]))
            for i in range(len(curve) - 1)]


if __name__ == "__main__":
    print("기본 최적:", {k: str(v) for k, v in zip(("x1", "x2"), optimum()[0])}, str(optimum()[1]))
    print("한 시간 증분(재봉/포장/주문한도):", [str(finite_difference(i)) for i in range(3)])
    print("쌍대 인증서 y=(1,1,0):", [str(v) if not isinstance(v, bool) else v for v in check_dual_certificate((1, 1, 0))])
    for h in (20, 30):
        print(f"재봉 +{h}시간 -> 최적값 증가", str(optimum(100 + h)[1] - 180))
    pk = value_curve(range(40, 121), which=1)
    chg2, prev = [], None
    for s in slopes(pk):
        if prev is not None and s[2] != prev[2]:
            chg2.append((s[0], str(prev[2]), str(s[2])))
        prev = s
    print("포장 시간 b2: 기울기가 바뀌는 지점:", chg2)
    curve = value_curve(range(60, 161, 10))
    print("b1 -> 최적값:", [(a, str(b)) for a, b in curve])
    print("구간 기울기:", [(a, b, str(s)) for a, b, s in slopes(curve)])
    fine = value_curve(range(70, 131))
    chg = []
    prev = None
    for s in slopes(fine):
        if prev is not None and s[2] != prev[2]:
            chg.append((s[0], str(prev[2]), str(s[2])))
        prev = s
    print("기울기가 바뀌는 b1:", chg)
    # 잘못된 인증서는 걸러져야 한다
    print("y=(2,0,0):", [str(v) if not isinstance(v, bool) else v for v in check_dual_certificate((2, 0, 0))])
