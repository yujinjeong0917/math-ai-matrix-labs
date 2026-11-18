"""0장: 처음 3초에 눈에 들어오는 것은 글자 크기 순서와 얼마나 겹치나.

사이트 홈(index.html)의 첫 화면 글자 크기를 assets/site.css의 실제 값(2026-11-18 열람)으로 계산한다.
화면 폭에 따라 clamp()가 크기를 바꾸므로 폭 세 가지(가상 화면 폭, 예시용)로 비교한다.
"""

# site.css 1절 토큰. (최소 px, vw 비율 %, 최대 px). 고정값은 최소=최대로 둔다.
TOKENS = {
    "hero": (34, 6.2, 56),      # --fs-hero: clamp(34px, 6.2vw, 56px)
    "lead": (19, 2.4, 24),      # --fs-lead: clamp(19px, 2.4vw, 24px)
    "body": (17, 0, 17),        # --fs-body: 17px (.btn 글자도 이 값)
    "caption": (14, 0, 14),     # --fs-caption: 14px (.kicker 글자)
}

# 홈 첫 화면 요소와 쓰는 토큰 (index.html의 .hero 영역)
HERO_ELEMENTS = [
    ("kicker", "caption"),
    ("h1 제목", "hero"),
    ("리드 문장", "lead"),
    ("본문·버튼 글자", "body"),
]

VIEWPORTS = [390, 600, 1440]  # 가상 화면 폭(px), 예시용
HERO_PADDING_TOP = 96  # index.html 인라인 CSS: .hero { padding-top: 96px; } (첫 화면 위쪽 여백)


def clamp_px(lo, vw_pct, hi, viewport):
    """CSS clamp(lo, vw_pct vw, hi)의 결과(px)."""
    if vw_pct == 0:
        return float(lo)
    return min(max(lo, viewport * vw_pct / 100), hi)


def sizes_at(viewport):
    return {name: clamp_px(*TOKENS[name], viewport) for name in TOKENS}


def ranking(viewport):
    """요소를 글자 크기 큰 순서로. 같은 크기는 목록 순서를 유지한다."""
    s = sizes_at(viewport)
    items = [(name, s[tok]) for name, tok in HERO_ELEMENTS]
    return sorted(items, key=lambda x: -x[1])


def hero_ratio(viewport):
    s = sizes_at(viewport)
    return s["hero"] / s["body"]


def viewport_where(token, which):
    """clamp가 최대(max) 또는 최소(min)에 닿는 화면 폭(px)."""
    lo, pct, hi = TOKENS[token]
    return (hi if which == "max" else lo) / (pct / 100)


if __name__ == "__main__":
    for vw in VIEWPORTS:
        s = sizes_at(vw)
        print(f"[폭 {vw}px] h1 {s['hero']:.1f}px / 리드 {s['lead']:.1f}px / 본문 {s['body']:.0f}px"
              f" / h1÷본문 {hero_ratio(vw):.3f}배")
        print("  크기 순서:", " > ".join(f"{n}({v:.1f})" for n, v in ranking(vw)))
    print(f"h1이 최대 56px에 닿는 폭: {viewport_where('hero', 'max'):.1f}px")
    print(f"h1이 최소 34px에서 벗어나는 폭: {viewport_where('hero', 'min'):.1f}px")
    print(f"리드가 최소 19px에서 벗어나는 폭: {viewport_where('lead', 'min'):.1f}px")
    print(f"리드가 최대 24px에 닿는 폭: {viewport_where('lead', 'max'):.1f}px")
    print(f"첫 화면 위쪽 여백(.hero padding-top): {HERO_PADDING_TOP}px")
