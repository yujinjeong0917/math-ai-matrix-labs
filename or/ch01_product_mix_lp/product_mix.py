"""1장: 두 상품 생산량 선형계획. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용.

문제: max c.x  s.t.  A x <= b, x >= 0  (변수 2개)
첫 번째 길: 제약선 두 개씩 교점을 모두 구해 꼭짓점 중 가장 좋은 값을 고른다(Fraction, 정확 계산).
두 번째 길: 가능한 정수 생산량을 모두 세어 본다.

입력: 제약 목록 [((a1, a2), b), ...]와 마진 C. 출력: 최적 생산량과 마진(만원).
대응: README 기대 출력 "기본 LP 최적: (20, 60), 마진 180만원", "양말 최소 70: (10, 70), 170",
"양말 최소 90: 가능한 생산량 없음".
"""
from fractions import Fraction as F
from itertools import combinations

C = (3, 2)                      # 마진(만원): 운동화 켤레당 3, 양말 5켤레 묶음당 2
BASE = [((2, 1), 100),          # 재봉 시간(시간)
        ((1, 1), 80),           # 포장 시간(시간)
        ((1, 0), 40)]           # 운동화 주문 한도(켤레)


def _rows(constraints):
    rows = [(F(a1), F(a2), F(b)) for (a1, a2), b in constraints]
    # 비음 조건 x >= 0도 같은 형식(a1*x1 + a2*x2 <= b)의 제약 행으로 넣어 교점 계산에 함께 쓴다
    rows += [(F(-1), F(0), F(0)), (F(0), F(-1), F(0))]   # x1>=0, x2>=0
    return rows


def vertices(constraints):
    rows = _rows(constraints)
    out = set()
    for (a1, a2, b), (d1, d2, e) in combinations(rows, 2):
        # 두 제약선의 교점을 크래머 공식으로 구한다. det == 0이면 평행(교점 없음)
        det = a1 * d2 - a2 * d1
        if det == 0:
            continue
        x1 = (b * d2 - a2 * e) / det
        x2 = (a1 * e - b * d1) / det
        # 모든 제약을 만족하는 교점만 꼭짓점(가능영역의 모서리). Fraction이라 == 비교가 정확하다
        if all(r1 * x1 + r2 * x2 <= rb for r1, r2, rb in rows):
            out.add((x1, x2))
    return sorted(out)


def solve(constraints, c=C):
    vs = vertices(constraints)
    if not vs:
        return None
    # 선형 목적함수의 최댓값은 가능영역의 꼭짓점 중 하나에서 나온다
    best = max(vs, key=lambda v: c[0] * v[0] + c[1] * v[1])
    return best, c[0] * best[0] + c[1] * best[1]


def brute_force(constraints, c=C, limit=200):
    # 상한 200은 이 가상 문제의 범위를 덮기 위한 값(제약의 최대값 100 이하)
    best = None
    for x1 in range(limit + 1):
        for x2 in range(limit + 1):
            if all(a1 * x1 + a2 * x2 <= b for (a1, a2), b in constraints):
                v = c[0] * x1 + c[1] * x2
                if best is None or v > best[1]:
                    best = ((x1, x2), v)
    return best


def with_min_socks(m):
    return BASE + [((0, -1), -m)]    # x2 >= m  <=>  -x2 <= -m


if __name__ == "__main__":
    for name, cons in (("기본", BASE), ("양말 최소 70", with_min_socks(70)),
                       ("양말 최소 90", with_min_socks(90))):
        s = solve(cons)
        bf = brute_force(cons)
        print(name, "꼭짓점:", [(str(a), str(b)) for a, b in vertices(cons)])
        print("  LP 최적:", None if s is None else ((str(s[0][0]), str(s[0][1])), str(s[1])), "| 전수:", bf)
