"""생산성 도구 3장: 프로토타입을 그래프로 보고 검사한다 + 접근성 숫자 검사.

프로토타입 = (화면, 상태) 노드와 전환 간선의 방향 그래프.
검사 세 가지:
  1. 시작에서 닿지 않는 상태(그려 놓고 갈 길이 없는 화면)
  2. 막다른 상태(나가는 연결이 없는 상태)
  3. 오류 상태인데 오류가 아닌 상태로 돌아갈 길이 없는 것
같은 검사를 두 방식(너비 우선 탐색 / 워셜 전이 폐포)으로 따로 구현해 결과를 대조한다.

입력: nodes(상태 노드 목록), edges((출발, 도착, 라벨...) 튜플 목록), start(시작 노드), is_error(노드가 오류 상태인지 판정하는 함수).
출력: {"unreachable", "dead_ends", "error_without_recovery"} 세 목록. 접근성 숫자 쪽은 색 대비(WCAG 2.2)와 터치 영역 크기를 돌려준다.
테스트가 확인하는 값(검정/흰색 대비 21.0, 회색 #6B7280 대비 4.83 등)이다(장 본문 숫자와의 대응은 README로 확인 필요).
"""

from collections import deque


def _nodes(nodes, edges):
    """주어진 노드와 간선 끝점을 합쳐 정렬한 노드 목록(간선에만 나오는 노드도 포함). 간선은 라벨이 붙은 3원소여도 앞 2개만 쓴다."""
    ns = set(nodes)
    for a, b, *_ in edges:
        ns.add(a)
        ns.add(b)
    return sorted(ns)


# ---------------------------------------------------------------- 구현 1: 너비 우선 탐색

def reach_bfs(start, edges):
    """start에서 간선을 따라 닿는 모든 노드 집합(start 자신 포함)."""
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
    reach = reach_bfs(start, edges) if start in ns else set()  # 시작 노드가 그래프에 없으면 아무것도 닿지 않는다
    out_deg = {n: 0 for n in ns}
    for a, b, *_ in edges:
        out_deg[a] += 1
    unreachable = [n for n in ns if n not in reach]
    dead_ends = [n for n in ns if out_deg[n] == 0]
    # 오류 노드에서 닿는 노드(자기 자신 제외) 중 오류가 아닌 것이 하나도 없으면 "돌아갈 길 없음"
    no_recovery = [n for n in ns if is_error(n) and not any(not is_error(m) for m in reach_bfs(n, edges) if m != n)]
    return {"unreachable": unreachable, "dead_ends": dead_ends, "error_without_recovery": no_recovery}


# ---------------------------------------------------------------- 구현 2: 워셜 전이 폐포(행렬)

def closure_warshall(ns, edges):
    """도달 가능 행렬 R: R[i][j]가 True면 노드 i에서 j로 갈 수 있다(자기 자신 포함). 반환: (R, 노드 -> 번호)."""
    idx = {n: i for i, n in enumerate(ns)}
    m = len(ns)
    R = [[False] * m for _ in range(m)]
    for i in range(m):
        R[i][i] = True
    for a, b, *_ in edges:
        R[idx[a]][idx[b]] = True
    # 함정: 중간 경유 노드 k가 가장 바깥 반복이어야 한다(i, j보다 안쪽에 두면 경로를 놓친다)
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
    # 시작 노드가 그래프에 없으면 모든 노드를 도달 불가로 센다(BFS 구현의 빈 집합과 같은 결과)
    unreachable = [n for n in ns if start not in idx or not R[idx[start]][idx[n]]]
    dead_ends = [n for n in ns if not has_out[idx[n]]]
    no_recovery = [n for n in ns if is_error(n) and not any(R[idx[n]][idx[x]] and x != n and not is_error(x) for x in ns)]
    return {"unreachable": unreachable, "dead_ends": dead_ends, "error_without_recovery": no_recovery}


# ---------------------------------------------------------------- 접근성 숫자

def _lin(c8):
    """8비트 sRGB 채널 값(0~255)을 선형 밝기로 바꾼다(WCAG 상대 휘도 정의의 감마 보정 풀기)."""
    c = c8 / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color):
    """"#RRGGBB" 색의 상대 휘도 L = 0.2126 R + 0.7152 G + 0.0722 B(선형 값 기준). 초록에 가장 민감하다는 사람 눈의 가중치다."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg, bg):
    """WCAG 2.2 정의: (밝은 쪽 L + 0.05) / (어두운 쪽 L + 0.05)."""
    # 어느 쪽이 글자색인지와 무관하게 밝은 쪽을 분자에 둔다(그래서 대비는 대칭이다)
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
# 합격선 4.5:1은 WCAG 일반 크기 글자 기준이다. 큰 글자(3:1)와 비활성 예외는 따로 판정하지 않고 exempt 표시만 한다

# 누를 수 있는 것과 그 크기(px). 그린 크기와 실제로 누르는 영역이 다를 수 있다.
TARGETS = [
    {"name": "뒤로 가기(아이콘만 그린 파일)", "w": 24, "h": 24},
    {"name": "뒤로 가기(체크리스트대로 터치 영역 지정)", "w": 44, "h": 44},
    {"name": "목록 셀의 '예약' 버튼", "w": 60, "h": 36},
    {"name": "큰 버튼(예약하기·예약 확정·목록으로)", "w": 343, "h": 48},
    {"name": "날짜 바꾸기 칩", "w": 72, "h": 32},
]


def accessibility_report():
    """글자색 짝마다 대비 비율(4.5 이상이면 합격)과, 누르는 영역마다 짧은 변이 24px(AA 기준)·44px(AAA 기준) 이상인지 돌려준다."""
    pairs = []
    for p in TEXT_PAIRS:
        r = contrast_ratio(p["fg"], p["bg"])
        pairs.append({**p, "ratio": round(r, 2), "pass_4_5": r >= 4.5, "exempt": p["inactive"]})
    targets = []
    for t in TARGETS:
        m = min(t["w"], t["h"])  # 가로세로 중 짧은 변 기준(손가락으로 누르기 가장 어려운 방향)
        targets.append({**t, "pass_24_AA": m >= 24, "pass_44_AAA": m >= 44})
    return {"text_pairs": pairs, "targets": targets}
