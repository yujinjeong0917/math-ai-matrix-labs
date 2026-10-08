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
"""

import copy
import re

CHAR_W = {"space": 4, "ascii": 8, "other": 14}  # 글자 폭 모형(px). 실제 글꼴 폭은 아니다.
TEXT_H = 20


def text_width(s):
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
    lo = node.get("min_" + axis)
    hi = node.get("max_" + axis)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


def _main(layout):
    return "w" if layout == "H" else "h"


def effective_sizing(node, axis):
    """부모 hug + 자식 fill이면 그 축에서 부모는 fixed가 된다(도움말의 Note)."""
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
    _place(node, x, y, intrinsic(node, "w") if w is None else w, intrinsic(node, "h") if h is None else h)
    return node


def _place(node, x, y, w, h):
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
    free = inner_main - fixed_sum - gap * (len(kids) - 1)
    share = free / len(fills) if fills else 0
    cur = l if main == "w" else t
    for c in kids:
        m = _clamp(c, main, share) if c.get(key(main)) == "fill" else intrinsic(c, main)
        cr = _clamp(c, cross, inner_cross) if c.get(key(cross)) == "fill" else intrinsic(c, cross)
        if main == "w":
            _place(c, x + cur, y + t + (inner_cross - cr) / 2, m, cr)  # 가로지르는 축은 가운데 정렬
        else:
            _place(c, x + l + (inner_cross - cr) / 2 if c.get(key(cross)) != "fill" else x + l, y + cur, cr, m)
        cur += m + gap


def freeze(laid, root=True):
    """auto layout으로 계산한 결과를 '고정 좌표로 그린 파일'로 얼린다(375px에서는 똑같이 보인다)."""
    out = copy.deepcopy(laid)
    _freeze(out, out["box"][0], out["box"][1], top=True)
    return out


def _freeze(node, px, py, top=False):
    x, y, w, h = node.pop("box")
    if not top:
        node["x"], node["y"] = x - px, y - py
    node["w"], node["h"] = w, h
    if node.get("children"):
        node["layout"] = None
    if node["kind"] == "text":
        node["sx"] = node["sy"] = "fixed"
    for c in node.get("children", []):
        _freeze(c, x, y)


# ---------------------------------------------------------------------------
# 검사: 폭을 바꾼 뒤 무엇이 어긋났나
# ---------------------------------------------------------------------------

def walk(node, parent=None):
    yield node, parent
    for c in node.get("children", []):
        yield from walk(c, node)


def check_layout(laid, eps=0.5):
    """넘침(자식이 부모 상자 밖으로 나감, 글자가 상자보다 넓음)과
    stretch 의도 요소의 좌우 여백 불일치를 센다."""
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
                left, right = x - px, (px + pw) - (x + w)
                if abs(left - right) > eps:
                    stretch_fail[id(n)] = n["name"]
    return {"overflow": sorted(overflow.values()), "stretch_fail": sorted(stretch_fail.values())}


def resize_screen(frame, new_w):
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


FRAME_NAME = re.compile(r"^S\d{2}_[가-힣A-Za-z]+_[가-힣A-Za-z]+$")
