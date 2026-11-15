"""생산성 도구 2장: auto layout 최소 구현과, 같은 화면을 고정 좌표로 얼린 판.

규칙 출처: Figma Help Center "Guide to auto layout"(확인일 2026-10-08)
https://help.figma.com/hc/en-us/articles/360040451373-Guide-to-auto-layout
- 안쪽 여백(padding)은 부모 가장자리와 자식 사이의 빈 공간, 간격(gap)은 자식과 자식 사이 거리.
- 크기 규칙: Hug contents(자식에 맞춰 줄고 늘어남, auto layout 프레임만),
  Fill container(부모의 남는 공간을 채움, auto layout 프레임의 자식만), Fixed(그대로), min/max(범위 제한).
- 자식 중 하나라도 Fill container면, 그 축에서 부모는 더 이상 hug가 아니고 Fixed가 된다.
- 문서의 예: 폭 40px 글자 + 좌우 여백 10px이면 hug 프레임 폭 60px, 글자가 50px이 되면 70px.

이 파일은 위 규칙 가운데 가로·세로 쌓기(Horizontal/Vertical)만 구현한다. wrap, grid,
정렬(align), 'Auto' 간격, ignore auto layout은 다루지 않는다. 실제 Figma의 계산과 세부가
다를 수 있으며, 교육용 모형이다.

노드는 dict 하나다.
  name, kind("frame" | "text"), layout(None | "H" | "V"), pad=(왼, 오, 위, 아래), gap,
  sx, sy("fixed" | "hug" | "fill"), w, h, min_w, max_w, text, x, y(고정 좌표 판에서만),
  intent("stretch": 부모 폭을 좌우 같은 여백으로 채우려는 요소), children

무엇을 재는가: 같은 화면을 auto layout 판과 고정 좌표 판으로 만들어 화면 폭을 바꿨을 때(375px -> 430px) 글자 넘침과 좌우 여백 불일치가 몇 곳에서 생기는지 센다.
입력: 노드 dict 트리와 화면 폭. 출력: 각 노드에 box = (x, y, w, h)를 채운 사본, check_layout의 문제 목록. 단위는 px.
테스트가 확인하는 숫자(hug 폭 60/70, 430px에서 고정 좌표판의 stretch 불일치 13곳 등)이다(장 본문 숫자와의 대응은 README로 확인 필요).
"""

import copy
import re

# 글자 폭을 문자 종류별 상수로 줄인 모형이다. 한글(other) 14px, 영문·숫자(ascii) 8px, 공백 4px. 실제 폰트 측정과는 다르다
CHAR_W = {"space": 4, "ascii": 8, "other": 14}  # 글자 폭 모형(px). 실제 글꼴 폭은 아니다.
TEXT_H = 20  # 글자 한 줄의 높이(px), 폰트 크기와 무관한 모형 상수


def text_width(s):
    """문자열의 모형 폭(px). 코드 포인트가 128 미만이면 ascii, 그 외(한글 등)는 other로 센다."""
    w = 0
    for ch in s:
        if ch == " ":
            w += CHAR_W["space"]
        elif ord(ch) < 128:
            w += CHAR_W["ascii"]
        else:
            w += CHAR_W["other"]
    return w


def _clamp(node, axis, v):
    """min_w/max_w(또는 min_h/max_h)가 있으면 값을 그 범위로 자른다. 최솟값을 먼저 적용하고 최댓값을 나중에 적용한다."""
    lo = node.get("min_" + axis)
    hi = node.get("max_" + axis)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


def _main(layout):
    """쌓는 방향의 축 이름: 가로("H")면 폭("w"), 세로면 높이("h")."""
    return "w" if layout == "H" else "h"


def effective_sizing(node, axis):
    """부모 hug + 자식 fill이면 그 축에서 부모는 fixed가 된다(도움말의 Note)."""
    # 크기 규칙 키: 폭은 "sx", 높이는 "sy". 기본값은 fixed
    s = node.get("s" + ("x" if axis == "w" else "y"), "fixed")
    if s == "hug" and node.get("layout") and any(
        c.get("s" + ("x" if axis == "w" else "y")) == "fill" for c in node.get("children", [])
    ):
        return "fixed"
    return s


def intrinsic(node, axis):
    """부모가 정해 주기 전, 노드가 스스로 원하는 크기."""
    if node["kind"] == "text":
        v = text_width(node["text"]) if axis == "w" else TEXT_H
        if node.get("s" + ("x" if axis == "w" else "y")) == "fixed":
            v = node[axis]
        return _clamp(node, axis, v)
    s = effective_sizing(node, axis)
    if s == "fixed" or not node.get("layout"):
        return _clamp(node, axis, node.get(axis, 0))
    l, r, t, b = node.get("pad", (0, 0, 0, 0))
    kids = node.get("children", [])
    # fill 자식은 부모가 남는 공간을 나눠 주는 대상이라 hug 계산에는 최솟값(없으면 0)만 기여한다
    sizes = [intrinsic(c, axis) if c.get("s" + ("x" if axis == "w" else "y")) != "fill" else _clamp(c, axis, 0) for c in kids]
    edge = (l + r) if axis == "w" else (t + b)
    if _main(node["layout"]) == axis:  # 쌓는 방향: 자식 크기의 합 + 간격 × (개수 - 1)
        v = edge + sum(sizes) + node.get("gap", 0) * max(len(kids) - 1, 0)
    else:  # 가로지르는 방향: 가장 큰 자식
        v = edge + (max(sizes) if sizes else 0)
    return _clamp(node, axis, v)


def layout(node, x=0, y=0, w=None, h=None):
    """노드 트리에 절대 좌표 box=(x, y, w, h)를 채운다. 원본은 건드리지 않고 사본을 돌려준다."""
    node = copy.deepcopy(node)
    # w, h를 안 주면 루트도 hug 계산으로 크기를 정한다. 주면(예: 375x812) 그 크기에 맞춘다
    _place(node, x, y, intrinsic(node, "w") if w is None else w, intrinsic(node, "h") if h is None else h)
    return node


def _place(node, x, y, w, h):
    """부모가 정해 준 (x, y, w, h)를 box에 기록하고, 자식들의 상자를 재귀로 정한다. 좌표는 모두 화면 기준 절대 좌표다."""
    node["box"] = (x, y, w, h)
    kids = node.get("children", [])
    if not kids:
        return
    if not node.get("layout"):  # 고정 좌표: 자식은 저장된 상대 좌표·크기를 그대로 쓴다
        for c in kids:
            _place(c, x + c["x"], y + c["y"], c["w"], c["h"])
        return
    l, r, t, b = node.get("pad", (0, 0, 0, 0))
    gap = node.get("gap", 0)
    main = _main(node["layout"])
    cross = "h" if main == "w" else "w"
    inner_main = (w - l - r) if main == "w" else (h - t - b)
    inner_cross = (h - t - b) if main == "w" else (w - l - r)
    key = lambda a: "s" + ("x" if a == "w" else "y")
    fixed_sum = sum(intrinsic(c, main) for c in kids if c.get(key(main)) != "fill")
    fills = [c for c in kids if c.get(key(main)) == "fill"]
    # 남는 공간 = 안쪽 크기 - 고정 자식 합 - 간격 합. fill 자식들이 똑같이 나눈다.
    # 주의: free가 음수가 돼도 막지 않는다(min이 없는 fill 자식은 음수 크기가 된다)
    free = inner_main - fixed_sum - gap * (len(kids) - 1)
    share = free / len(fills) if fills else 0
    cur = l if main == "w" else t
    for c in kids:
        m = _clamp(c, main, share) if c.get(key(main)) == "fill" else intrinsic(c, main)
        # cr: 가로지르는 방향 크기. fill이면 안쪽 크기 전체, 아니면 자신이 원하는 크기
        cr = _clamp(c, cross, inner_cross) if c.get(key(cross)) == "fill" else intrinsic(c, cross)
        if main == "w":
            _place(c, x + cur, y + t + (inner_cross - cr) / 2, m, cr)  # 가로지르는 축은 가운데 정렬
        else:
            _place(c, x + l + (inner_cross - cr) / 2 if c.get(key(cross)) != "fill" else x + l, y + cur, cr, m)
        cur += m + gap


def freeze(laid, root=True):
    """auto layout으로 계산한 결과를 '고정 좌표로 그린 파일'로 얼린다(375px에서는 똑같이 보인다)."""
    # root 인자는 함수 안에서 쓰이지 않는다(호출부가 넘기는지는 확인 필요)
    out = copy.deepcopy(laid)
    _freeze(out, out["box"][0], out["box"][1], top=True)
    return out


def _freeze(node, px, py, top=False):
    x, y, w, h = node.pop("box")
    if not top:
        node["x"], node["y"] = x - px, y - py  # 부모 기준 상대 좌표로 바꿔 저장(고정 좌표 판의 형식)
    node["w"], node["h"] = w, h
    if node.get("children"):
        node["layout"] = None  # auto layout을 끄면 이후 폭이 바뀌어도 자식 좌표가 따라 움직이지 않는다
    if node["kind"] == "text":
        node["sx"] = node["sy"] = "fixed"
    for c in node.get("children", []):
        _freeze(c, x, y)


# ---------------------------------------------------------------------------
# 검사: 폭을 바꾼 뒤 무엇이 어긋났나
# ---------------------------------------------------------------------------

def walk(node, parent=None):
    """트리를 깊이 우선으로 돌며 (노드, 부모)를 차례로 내놓는다. 루트의 부모는 None."""
    yield node, parent
    for c in node.get("children", []):
        yield from walk(c, node)


def check_layout(laid, eps=0.5):
    """넘침(자식이 부모 상자 밖으로 나감, 글자가 상자보다 넓음)과
    stretch 의도 요소의 좌우 여백 불일치를 센다."""
    # eps는 반 픽셀 허용 오차: 가운데 정렬로 생기는 .5px 어긋남을 문제로 세지 않기 위함
    overflow, stretch_fail = {}, {}  # 같은 레이어를 두 번 세지 않도록 노드 단위로 모은다
    for n, p in walk(laid):
        x, y, w, h = n["box"]
        if n["kind"] == "text" and text_width(n["text"]) > w + eps:
            overflow[id(n)] = n["name"]
        if p is not None:
            px, py, pw, ph = p["box"]
            if x < px - eps or x + w > px + pw + eps:
                overflow[id(n)] = n["name"]
            if n.get("intent") == "stretch":
                left, right = x - px, (px + pw) - (x + w)  # 부모 상자와 자식 상자 사이의 좌우 여백
                if abs(left - right) > eps:
                    stretch_fail[id(n)] = n["name"]
    return {"overflow": sorted(overflow.values()), "stretch_fail": sorted(stretch_fail.values())}


def resize_screen(frame, new_w):
    """화면 프레임의 폭만 바꾼 사본. 고정 좌표 판은 자식이 그대로라 여백이 어긋나고, auto layout 판은 자식이 따라간다."""
    f = copy.deepcopy(frame)
    f["w"] = new_w
    return f


def set_text(node, name, new_text):
    """이름이 name인 글자 레이어의 문구를 바꾼다. 바꾼 레이어 수를 돌려준다."""
    hits = 0
    for n, _ in walk(node):
        if n["kind"] == "text" and n["name"] == name:
            n["text"] = new_text
            hits += 1
    return hits


# 프레임 이름 규칙 "S01_목록_기본"꼴: S + 숫자 두 자리 + _ + 한글/영문 + _ + 한글/영문
FRAME_NAME = re.compile(r"^S\d{2}_[가-힣A-Za-z]+_[가-힣A-Za-z]+$")
