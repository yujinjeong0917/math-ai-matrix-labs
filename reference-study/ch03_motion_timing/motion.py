"""3장: 지속 시간을 프레임 수로 바꾸고, ease 곡선에서 진행률을 읽는다.

CSS 값: assets/site.css 의 transition 선언(transitions_2026-11-18.json).
ease = cubic-bezier(0.25, 0.1, 0.25, 1), linear = 항등 함수: W3C CSS Easing Functions Level 1
(https://www.w3.org/TR/css-easing-1/, 2026-11-18 확인).
화면 주사율 60Hz는 가정(예시용)이다. 30Hz(낮은 쪽 극단)와 120Hz(높은 쪽)는 비교용 가정이다. 실제 기기는 다를 수 있다.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EASE = (0.25, 0.1, 0.25, 1.0)


def load():
    with open(os.path.join(HERE, "transitions_2026-11-18.json"), encoding="utf-8") as f:
        return json.load(f)["transitions"]


def bezier_point(s, p=EASE):
    """곡선 위 위치 변수 s에서 (시간 비율 x, 진행률 y). 손계산 확인용."""
    x1, y1, x2, y2 = p
    x = 3 * (1 - s) ** 2 * s * x1 + 3 * (1 - s) * s ** 2 * x2 + s ** 3
    y = 3 * (1 - s) ** 2 * s * y1 + 3 * (1 - s) * s ** 2 * y2 + s ** 3
    return x, y


def load_elsewhere():
    """site.css 밖(홈, 트랙 목록, 장 페이지의 인라인 style, 공용 css/js)에서 직접 찾은 결과(2026-11-18)."""
    with open(os.path.join(HERE, "transitions_2026-11-18.json"), encoding="utf-8") as f:
        return json.load(f)["checked_elsewhere"]


def frames(seconds, fps=60):
    return seconds * fps


def bezier_progress(t, p=EASE):
    """시간 비율 t(0~1)에서 cubic-bezier 진행률. x(s)=t 를 이분법으로 푼 뒤 y(s)."""
    x1, y1, x2, y2 = p

    def coord(s, a, b):
        return 3 * (1 - s) ** 2 * s * a + 3 * (1 - s) * s ** 2 * b + s ** 3

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if coord(mid, x1, x2) < t:
            lo = mid
        else:
            hi = mid
    return coord((lo + hi) / 2, y1, y2)


def scale_at(t, end_scale=0.95, easing="ease"):
    prog = t if easing == "linear" else bezier_progress(t)
    return 1 + (end_scale - 1) * prog


def button_height(min_height_px=44, scale=0.95):
    return min_height_px * scale


def summary(items):
    secs = sorted({i["seconds"] for i in items})
    return {s: sum(1 for i in items if i["seconds"] == s) for s in secs}


def without_reduced_motion(items):
    return [i["selector"] for i in items if i["reduced_motion"] is None]


if __name__ == "__main__":
    items = load()
    print("선언 수", len(items), "지속 시간별", summary(items))
    for s in (0.12, 0.2):
        print(f"{s}초 = 60Hz에서 {frames(s):.1f}프레임, 30Hz에서 {frames(s, 30):.1f}프레임,"
              f" 120Hz에서 {frames(s, 120):.1f}프레임")
    print("ease 진행률(시간 25/50/75%):", [round(bezier_progress(t), 3) for t in (0.25, 0.5, 0.75)])
    x7, y7 = bezier_point(0.7)
    print(f"곡선 위 s=0.7: 시간 {x7:.4f}, 진행률 {y7:.4f} / 시간 50%의 진행률(4자리): {bezier_progress(0.5):.4f}")
    print("linear 진행률(시간 25/50/75%):", [0.25, 0.5, 0.75])
    print("버튼 누름 크기(scale 0.95):", [round(scale_at(t), 4) for t in (0, 0.25, 0.5, 0.75, 1)])
    print(f"44px 버튼이 줄어드는 높이: {44 - button_height():.1f}px (누른 뒤 {button_height():.1f}px)")
    print("움직임 줄이기 설정 선언이 없는 선택자:", without_reduced_motion(items))
    ew = load_elsewhere()
    print("site.css 밖 확인: 인라인 <style>의 transition", ew["inline_transition_rules"], "개, @keyframes/animation", ew["keyframes_or_animation"], "개")
