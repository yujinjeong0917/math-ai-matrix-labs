"""2장: 이 사이트 CSS 색의 WCAG 대비 비율과 글자 크기 비율.

색: math-ai-matrix/assets/site.css 1절 토큰(2026-11-18 열람).
식과 기준값: W3C WCAG 2.2 (https://www.w3.org/TR/WCAG22/), 2026-11-18 확인.
  상대 휘도 L = 0.2126 R + 0.7152 G + 0.0722 B,
  채널 c = 8비트/255, c <= 0.04045 면 c/12.92, 아니면 ((c+0.055)/1.055)^2.4
  대비 비율 = (L1 + 0.05) / (L2 + 0.05), L1은 밝은 쪽
  AA 4.5:1 (큰 글자 3:1), AAA 7:1 (큰 글자 4.5:1), 비텍스트 3:1.
  큰 글자: 18pt 이상, 또는 14pt 굵은 글자 이상 (1pt = 4/3 CSS px 로 환산).
"""

COLORS = {
    "ink": "#1d1d1f", "ink-2": "#6e6e73", "bg": "#ffffff", "bg-alt": "#f5f5f7",
    "link": "#0066cc", "btn-bg": "#0066cc", "btn-ink": "#ffffff",
    "tile": "#272729", "on-dark": "#f5f5f7", "on-dark-2": "#a1a1a6",
    "link-on-dark": "#2997ff", "card-on-tile": "#1d1d1f", "black": "#000000",
    "old-ink-2": "#7a7a7a",
}

# (설명, 글자색, 바탕색)
PAIRS = [
    ("라이트 본문 ink / bg", "ink", "bg"),
    ("라이트 보조 글자 ink-2 / bg", "ink-2", "bg"),
    ("라이트 보조 글자 ink-2 / bg-alt", "ink-2", "bg-alt"),
    ("링크 link / bg", "link", "bg"),
    ("링크 link / bg-alt", "link", "bg-alt"),
    ("기본 버튼 btn-ink / btn-bg", "btn-ink", "btn-bg"),
    ("어두운 타일 on-dark / tile", "on-dark", "tile"),
    ("어두운 타일 보조 on-dark-2 / tile", "on-dark-2", "tile"),
    ("어두운 타일 링크 link-on-dark / tile", "link-on-dark", "tile"),
    ("어두운 타일 카드 링크 link-on-dark / card-on-tile", "link-on-dark", "card-on-tile"),
    ("다크 모드 보조 on-dark-2 / black", "on-dark-2", "black"),
    ("옛 보조색 #7a7a7a / bg (현재 조합 아님, CSS 주석이 말한 옛 값 4.3:1)", "old-ink-2", "bg"),
]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lin(c8):
    c = c8 / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color):
    r, g, b = (_lin(c) for c in hex_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(fg, bg):
    l1, l2 = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def is_large(px, bold=False):
    """18pt(=24px) 이상, 또는 굵은 14pt(=18.667px) 이상."""
    pt = px * 3 / 4
    return pt >= 18 or (bold and pt >= 14)


def verdict(ratio, large=False):
    aa = 3.0 if large else 4.5
    aaa = 4.5 if large else 7.0
    return {"AA": ratio >= aa, "AAA": ratio >= aaa}


# 타입 스케일 (site.css 토큰, 1440px 폭 기준 clamp 결과)
SCALE_1440 = [("hero", 56), ("display", 40), ("lead", 24), ("tagline", 21), ("body", 17), ("caption", 14), ("fine", 12)]


def step_ratios(scale):
    return [(a[0], b[0], a[1] / b[1]) for a, b in zip(scale, scale[1:])]


if __name__ == "__main__":
    for label, f, b in PAIRS:
        r = contrast(COLORS[f], COLORS[b])
        v = verdict(r)
        print(f"{label}: {r:.2f}:1  일반글자 AA {'통과' if v['AA'] else '미달'} / AAA {'통과' if v['AAA'] else '미달'}")
    print("# 아래 한 줄: 큰 글자(대비 기준 3:1이 적용되는 글자) 판정. 24px 이상 또는 굵은 글자 18.67px 이상이면 True")
    print("큰 글자 판정: 24px", is_large(24), "/ 19px", is_large(19), "/ 굵은 19px", is_large(19, True),
          "/ 굵은 18px", is_large(18, True))
    r = contrast(COLORS["ink-2"], COLORS["bg"])
    print(f"리드(ink-2/bg) 폭 1440px(24px) 필요 기준 3:1 -> {r:.2f} {verdict(r, True)['AA']},"
          f" 폭 390px(19px) 필요 기준 4.5:1 -> {verdict(r)['AA']}")
    for a, b, q in step_ratios(SCALE_1440):
        print(f"{a}/{b}: {q:.3f}")
    print(f"가장 큰/가장 작은: {56 / 12:.2f}")
