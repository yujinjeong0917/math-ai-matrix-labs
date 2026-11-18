"""1장: 화면을 상자로 쪼개고 정렬선을 찾는다.

x 좌표와 너비는 홈(index.html)의 CSS 값(2026-11-18 열람)에서 계산한 값이다(브라우저에서 잰 값이 아니다).
y 좌표와 높이, 글자 상자의 너비는 가상, 예시용이다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def top(self):
        return self.y


def wrap_padding(viewport):
    """.wrap 의 좌우 안쪽 여백: 22px, 폭 419px 이하에서는 16px (site.css 11절 반응형)."""
    return 16 if viewport <= 419 else 22


def content_width(viewport, wrap_max):
    """.wrap 의 안쪽 너비: min(화면 폭, max-width) - 좌우 padding 씩."""
    return min(viewport, wrap_max) - 2 * wrap_padding(viewport)


def columns(width, min_col, gap):
    """grid-template-columns: repeat(auto-fill, minmax(min_col, 1fr)) 의 열 수와 열 너비."""
    n = max(1, int((width + gap) // (min_col + gap)))
    return n, (width - gap * (n - 1)) / n


def group_layout(viewport, head_fr=0.9, main_fr=2.0, gap=40):
    """.group {grid-template-columns: 0.9fr 2fr; gap: 40px} 의 머리 열과 본문 열 (x, 너비)."""
    wrap_left = (viewport - min(viewport, 1200)) / 2 + wrap_padding(viewport)
    inner = content_width(viewport, 1200) - gap
    head_w = inner * head_fr / (head_fr + main_fr)
    main_w = inner * main_fr / (head_fr + main_fr)
    return (wrap_left, head_w), (wrap_left + head_w + gap, main_w)


def start_section_boxes(viewport=1440):
    """'시작하기' 구역: 머리글 세 줄과 링크 상자 두 개. y, h는 가상."""
    (hx, hw), (mx, mw) = group_layout(viewport)
    n, cw = columns(mw, 340, 16)
    boxes = [
        Box("kicker", hx, 200, 24, 20),
        Box("h2 제목", hx, 224, 140, 48),
        Box("설명", hx, 284, hw, 90),
    ]
    for i in range(2):
        boxes.append(Box(f"링크 상자{i + 1}", mx + i * (cw + 16), 200, cw, 120))
    return boxes, n, cw


def hero_boxes(viewport=1440):
    """첫 화면: 중앙 정렬 상자들. lead 640px, stats 760px는 CSS max-width. 나머지 너비와 y, h는 가상."""
    def centered(name, w, y, h):
        return Box(name, (viewport - w) / 2, y, w, h)

    boxes = [
        centered("kicker", 220, 144, 20),
        centered("h1", 520, 180, 130),
        centered("lead", 640, 330, 70),
        centered("버튼줄", 360, 432, 44),
        centered("stats", 760, 532, 60),
    ]
    return boxes


def find_lines(boxes, kind, tol=0.5):
    """kind: left, right, cx, top. 허용 오차 tol 안에서 같은 값을 가진 상자 2개 이상의 묶음."""
    vals = sorted((getattr(b, kind), b.name) for b in boxes)
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if abs(v[0] - cur[-1][0]) <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [(round(sum(v for v, _ in g) / len(g), 2), [n for _, n in g]) for g in groups if len(g) >= 2]


def density(boxes, canvas_w, canvas_h):
    """상자가 겹치지 않는다고 보고 면적 합 / 화면 면적."""
    return sum(b.w * b.h for b in boxes) / (canvas_w * canvas_h)


if __name__ == "__main__":
    for vp in (390, 1024, 1440):
        w = content_width(vp, 1200)
        print(f"[폭 {vp}px] .wrap--wide 안쪽 너비 {w}px")
    (hx, hw), (mx, mw) = group_layout(1440)
    print(f"[폭 1440px] 머리 열 x={hx:.2f} 너비 {hw:.2f} / 본문 열 x={mx:.2f} 너비 {mw:.2f}")
    boxes, n, cw = start_section_boxes()
    q = (mw + 16) / (340 + 16)
    print(f"열 수 식 값 (C+g)/(m+g) = {q:.3f} -> 바닥 {n}열, 상자 너비 {cw:.2f}px")
    print(f"[폭 1440px] 가운데 정렬 여백 {(1440 - 1200) / 2:.0f}px + 안쪽 여백(padding) {wrap_padding(1440)}px = {hx:.0f}px")
    for kind in ("left", "right", "top"):
        print(kind, find_lines(boxes, kind))
    print("중앙선(첫 화면)", find_lines(hero_boxes(), "cx"))
    doubled = [Box(b.name, b.x, b.y, b.w, b.h * 2) for b in boxes]
    same = all(find_lines(doubled, k) == find_lines(boxes, k) for k in ("left", "right", "top"))
    print("상자 높이를 모두 2배로 해도 왼쪽, 오른쪽, 위쪽 선이 같은가:", same)
    print(f"밀도(가상 y, h) {density(boxes, 1440, 400):.3f}")
