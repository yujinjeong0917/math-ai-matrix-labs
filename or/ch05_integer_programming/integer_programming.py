"""5장. 상담 인력 배치: LP 완화, 분기한정, 전수 탐색 대조.

표준 라이브러리만 쓴다. 모든 숫자는 학습용 가상값(예시용)이다.
최대화 문제:  maximize c·x  subject to  A x <= b,  x >= 0,  x 정수.
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import floor

# 가상 사례: x1 = 정규 상담원(명), x2 = 시간제 상담원(명)
C = (5, 4)                      # 1명이 더하는 처리량(가상 점수)
A = ((4, 3), (1, 2))            # 인건비(가상 단위), 감독 부담(가상 단위)
B = (13, 6)                     # 쓸 수 있는 한도


def _solve_square(M, v):
    """정사각 연립방정식을 Fraction으로 푼다. 특이하면 None."""
    n = len(M)
    a = [[F(x) for x in row] + [F(y)] for row, y in zip(M, v)]
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        a[col] = [x / p for x in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return tuple(a[i][n] for i in range(n))


def solve_lp(c, A, b, lower=None, upper=None):
    """꼭짓점 열거로 LP를 정확히(Fraction) 푼다. 가능해가 없으면 None.

    lower/upper: 분기에서 붙는 변수 범위(없으면 0 이상, 위쪽 제한 없음).
    반환: (최적값, 해) 또는 None.
    """
    n = len(c)
    lower = list(lower) if lower else [0] * n
    upper = list(upper) if upper else [None] * n
    rows = [(tuple(map(F, r)), F(v)) for r, v in zip(A, b)]
    for i in range(n):
        e = [F(0)] * n
        e[i] = F(-1)
        rows.append((tuple(e), F(-lower[i])))
        if upper[i] is not None:
            e = [F(0)] * n
            e[i] = F(1)
            rows.append((tuple(e), F(upper[i])))
    best = None
    for pick in combinations(range(len(rows)), n):
        x = _solve_square([rows[i][0] for i in pick], [rows[i][1] for i in pick])
        if x is None:
            continue
        if all(sum(a * xi for a, xi in zip(r, x)) <= v for r, v in rows):
            val = sum(F(ci) * xi for ci, xi in zip(c, x))
            if best is None or val > best[0]:
                best = (val, x)
    return best


def is_feasible(x, A, b):
    return all(xi >= 0 for xi in x) and all(
        sum(a * xi for a, xi in zip(r, x)) <= v for r, v in zip(A, b))


def objective(c, x):
    return sum(ci * xi for ci, xi in zip(c, x))


def brute_force(c, A, b, limit=30):
    """0..limit 범위의 모든 정수점을 직접 확인한다(대조용)."""
    best = None
    count = 0
    for x in product(range(limit + 1), repeat=len(c)):
        if is_feasible(x, A, b):
            count += 1
            v = objective(c, x)
            if best is None or v > best[0]:
                best = (v, x)
    return best, count


def rounding_report(c, A, b):
    """LP 해를 반올림·버림·올림했을 때 무엇이 되는지 정리한다."""
    _, x = solve_lp(c, A, b)
    nearest = tuple(round(xi) for xi in x)
    cands = []
    for pick in product(*[(floor(xi), floor(xi) + 1) for xi in x]):
        if is_feasible(pick, A, b):
            cands.append((objective(c, pick), pick))
    best_round = max(cands) if cands else None
    return {
        "lp_x": x,
        "nearest": nearest,
        "nearest_feasible": is_feasible(nearest, A, b),
        "best_feasible_rounding": best_round,
    }


def branch_and_bound(c, A, b):
    """깊이 우선 분기한정. 목적 계수가 정수라 상한은 내림해서 가지치기한다.

    분기 규칙: 값이 분수인 변수 중 번호가 가장 작은 것, 내림 가지를 먼저 본다.
    반환: dict(best, x, nodes, log)
    """
    n = len(c)
    incumbent = [None, None]  # (값, 해)
    log = []
    nodes = [0]

    def visit(lower, upper, depth):
        nodes[0] += 1
        no = nodes[0]
        res = solve_lp(c, A, b, lower, upper)
        if res is None:
            log.append((no, depth, "가능해 없음", None, None))
            return
        val, x = res
        bound = floor(val)
        if incumbent[0] is not None and bound <= incumbent[0]:
            log.append((no, depth, "가지치기", val, x))
            return
        frac = next((i for i in range(n) if x[i].denominator != 1), None)
        if frac is None:
            log.append((no, depth, "정수해 채택", val, x))
            if incumbent[0] is None or val > incumbent[0]:
                incumbent[0], incumbent[1] = int(val), tuple(int(v) for v in x)
            return
        log.append((no, depth, f"x{frac + 1} 분기", val, x))
        fl = floor(x[frac])
        up = list(upper)
        up[frac] = fl
        visit(lower, up, depth + 1)
        lo = list(lower)
        lo[frac] = fl + 1
        visit(lo, upper, depth + 1)

    visit([0] * n, [None] * n, 0)
    return {"best": incumbent[0], "x": incumbent[1], "nodes": nodes[0], "log": log}


def fmt(x):
    return "(" + ", ".join(str(v) for v in x) + ")"


def sweep_budget(c, A, b, budgets):
    """인건비 한도를 바꿔 가며 LP 상한과 정수 최적값의 차이를 본다."""
    rows = []
    for v in budgets:
        bb = (v, b[1])
        lp_val, _ = solve_lp(c, A, bb)
        bnb = branch_and_bound(c, A, bb)
        rows.append((v, lp_val, bnb["best"], bnb["x"], bnb["nodes"]))
    return rows


def main():
    lp_val, lp_x = solve_lp(C, A, B)
    print("LP 완화 최적:", fmt(lp_x), "처리량", lp_val, "=", float(lp_val))
    rep = rounding_report(C, A, B)
    print("반올림:", fmt(rep["nearest"]), "가능?", rep["nearest_feasible"])
    print("바닥/천장 조합 중 가능한 최선:", rep["best_feasible_rounding"])
    (bv, bx), n_feas = brute_force(C, A, B)
    print("전수 탐색:", fmt(bx), "처리량", bv, "가능한 정수점", n_feas, "개")
    r = branch_and_bound(C, A, B)
    print("분기한정:", fmt(r["x"]), "처리량", r["best"], "노드", r["nodes"])
    for row in r["log"]:
        no, d, what, val, x = row
        print(" ", no, "깊이", d, what, "" if val is None else f"{val} {fmt(x)}")
    print("인건비 한도 훑기 (한도, LP 상한, 정수 최적, 해, 노드):")
    for row in sweep_budget(C, A, B, range(10, 17)):
        v, lv, best, x, nd = row
        print(" ", v, lv, best, fmt(x), nd)


if __name__ == "__main__":
    main()
