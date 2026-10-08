"""생산성 도구 3장: 프로토타입을 그래프로 보고 검사한다 + 접근성 숫자 검사.

프로토타입 = (화면, 상태) 노드와 전환 간선의 방향 그래프.
검사 세 가지:
  1. 시작에서 닿지 않는 상태(그려 놓고 갈 길이 없는 화면)
  2. 막다른 상태(나가는 연결이 없는 상태)
  3. 오류 상태인데 오류가 아닌 상태로 돌아갈 길이 없는 것
같은 검사를 두 방식(너비 우선 탐색 / 워셜 전이 폐포)으로 따로 구현해 결과를 대조한다.
"""

from collections import deque


def _nodes(nodes, edges):
    ns = set(nodes)
    for a, b, *_ in edges:
        ns.add(a)
        ns.add(b)
    return sorted(ns)


# ---------------------------------------------------------------- 구현 1: 너비 우선 탐색

def reach_bfs(start, edges):
    adj = {}
    for a, b, *_ in edges:
        adj.setdefault(a, []).append(b)
    seen, dq = {start}, deque([start])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    return seen


def check_bfs(nodes, edges, start, is_error):
    ns = _nodes(nodes, edges)
    reach = reach_bfs(start, edges) if start in ns else set()
    out_deg = {n: 0 for n in ns}
    for a, b, *_ in edges:
        out_deg[a] += 1
    unreachable = [n for n in ns if n not in reach]
    dead_ends = [n for n in ns if out_deg[n] == 0]
    no_recovery = [n for n in ns if is_error(n) and not any(not is_error(m) for m in reach_bfs(n, edges) if m != n)]
    return {"unreachable": unreachable, "dead_ends": dead_ends, "error_without_recovery": no_recovery}


# ---------------------------------------------------------------- 구현 2: 워셜 전이 폐포(행렬)

def closure_warshall(ns, edges):
    idx = {n: i for i, n in enumerate(ns)}
    m = len(ns)
    R = [[False] * m for _ in range(m)]
    for i in range(m):
        R[i][i] = True
    for a, b, *_ in edges:
        R[idx[a]][idx[b]] = True
    for k in range(m):
        for i in range(m):
            if R[i][k]:
                rk = R[k]
                ri = R[i]
                for j in range(m):
                    if rk[j]:
                        ri[j] = True
    return R, idx


def check_warshall(nodes, edges, start, is_error):
    ns = _nodes(nodes, edges)
    R, idx = closure_warshall(ns, edges)
    m = len(ns)
    has_out = [False] * m
    for a, b, *_ in edges:
        has_out[idx[a]] = True
    unreachable = [n for n in ns if start not in idx or not R[idx[start]][idx[n]]]
    dead_ends = [n for n in ns if not has_out[idx[n]]]
    no_recovery = [n for n in ns if is_error(n) and not any(R[idx[n]][idx[x]] and x != n and not is_error(x) for x in ns)]
    return {"unreachable": unreachable, "dead_ends": dead_ends, "error_without_recovery": no_recovery}


# ---------------------------------------------------------------- 접근성 숫자

def _lin(c8):
    c = c8 / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg, bg):
    """WCAG 2.2 정의: (밝은 쪽 L + 0.05) / (어두운 쪽 L + 0.05)."""
    l1, l2 = sorted([relative_luminance(fg), relative_luminance(bg)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


# 화면에서 쓰는 글자색/배경색 짝. inactive=True는 비활성 버튼처럼 WCAG 1.4.3이 예외로 두는 경우.
TEXT_PAIRS = [
    {"name": "본문 글자 / 흰 배경", "fg": "#111827", "bg": "#FFFFFF", "inactive": False},
    {"name": "안내 문구 / 흰 배경", "fg": "#6B7280", "bg": "#FFFFFF", "inactive": False},
    {"name": "입력창 예시 글자(placeholder) / 흰 배경", "fg": "#9CA3AF", "bg": "#FFFFFF", "inactive": False},
    {"name": "버튼 글자 흰색 / 버튼 색", "fg": "#FFFFFF", "bg": "#0B7A55", "inactive": False},
    {"name": "비활성 버튼 글자 / 비활성 버튼", "fg": "#FFFFFF", "bg": "#D1D5DB", "inactive": True},
]

# 누를 수 있는 것과 그 크기(px). 그린 크기와 실제로 누르는 영역이 다를 수 있다.
TARGETS = [
    {"name": "뒤로 가기(아이콘만 그린 파일)", "w": 24, "h": 24},
    {"name": "뒤로 가기(체크리스트대로 터치 영역 지정)", "w": 44, "h": 44},
    {"name": "목록 셀의 '예약' 버튼", "w": 60, "h": 36},
    {"name": "큰 버튼(예약하기·예약 확정·목록으로)", "w": 343, "h": 48},
    {"name": "날짜 바꾸기 칩", "w": 72, "h": 32},
]


def accessibility_report():
    pairs = []
    for p in TEXT_PAIRS:
        r = contrast_ratio(p["fg"], p["bg"])
        pairs.append({**p, "ratio": round(r, 2), "pass_4_5": r >= 4.5, "exempt": p["inactive"]})
    targets = []
    for t in TARGETS:
        m = min(t["w"], t["h"])
        targets.append({**t, "pass_24_AA": m >= 24, "pass_44_AAA": m >= 44})
    return {"text_pairs": pairs, "targets": targets}
