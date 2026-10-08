"""관통 프로젝트 "동네 스터디룸 예약"의 화면 4개를 두 가지 파일로 만든다.

1. 핑퐁 파일: 버튼을 화면마다 따로 그린 사본, 고정 좌표, 이름 규칙 없음.
2. 명세 파일: 버튼은 main component 1개 + instance, 모든 프레임에 auto layout, 이름 규칙 준수.

component 규칙 출처(확인일 2026-10-08):
- main component를 고치면 연결된 instance에 반영된다(같은 파일은 즉시, 라이브러리는 publish 후).
  https://help.figma.com/hc/en-us/articles/360038665934
- variant를 바꾸거나 instance를 바꿔 끼울 때 Figma는 instance에서 바꾼 값(override)을 보존하려고 한다.
  instance의 밑바탕 구조는 바꿀 수 없다. https://help.figma.com/hc/en-us/articles/360039150733
- detach하면 지금의 레이어·속성을 지닌 보통 프레임이 되고, main component와의 연결이 끊겨 변경을 더 이상 받지 않는다.
  https://help.figma.com/hc/en-us/articles/360038665754

"override한 속성은 main component에서 같은 속성을 바꿔도 instance 값이 유지된다"는 동작은
도움말에 직접 적혀 있지 않다. 이 모형은 그렇게 가정한다. [확인 필요]
"""

import copy

import productivity_ch02_layout as L

SCREEN_W, SCREEN_H = 375, 812
PAD = 16  # 화면 좌우 여백(간격 토큰은 4의 배수)


def text(name, s, sx="hug"):
    return {"name": name, "kind": "text", "text": s, "sx": sx, "sy": "hug"}


def button_main():
    """버튼 main component. 라벨 폭에 맞춰 늘어나는 hug 프레임."""
    return {
        "name": "Button/Primary", "kind": "frame", "layout": "H", "pad": (16, 16, 12, 12), "gap": 0,
        "sx": "hug", "sy": "hug", "fill": "#1F6FEB", "radius": 8,
        "children": [text("label", "예약")],
    }


def instance(main, overrides=None, sx="hug", intent=None):
    return {"kind": "instance", "main": main, "overrides": dict(overrides or {}), "sx": sx, "intent": intent, "detached": False}


def topbar(title):
    return {"name": "TopBar", "kind": "frame", "layout": "H", "pad": (16, 16, 16, 16), "gap": 8,
            "sx": "fill", "sy": "hug", "intent": "stretch", "children": [text("title", title)]}


def row(name, label, btn):
    return {"name": name, "kind": "frame", "layout": "H", "pad": (16, 16, 12, 12), "gap": 8, "sx": "fill", "sy": "hug",
            "intent": "stretch", "children": [text("room", label, sx="fill"), btn]}


def field(name, label):
    return {"name": name, "kind": "frame", "layout": "H", "pad": (12, 12, 12, 12), "gap": 0, "sx": "fill", "sy": "hug",
            "intent": "stretch", "children": [text("hint", label, sx="fill")]}


def screen(name, children):
    return {"name": name, "kind": "frame", "layout": "V", "pad": (PAD, PAD, 0, 24), "gap": 12,
            "sx": "fixed", "sy": "fixed", "w": SCREEN_W, "h": SCREEN_H, "children": children}


def spec_file(overrides_drift=False, detach_drift=False):
    """명세를 따른 파일. drift 인자를 켜면 현업에서 흔한 어긋남을 하나씩 넣는다.
    - overrides_drift: 완료 화면 버튼의 채우기를 instance에서 회색으로 덮어쓴다.
    - detach_drift: 예약 화면 버튼을 아이콘을 넣으려고 detach했다고 친다.
    """
    mains = {"Button/Primary": button_main()}
    cta = lambda label, **kw: instance("Button/Primary", {"label.text": label, **kw}, sx="fill", intent="stretch")
    s1 = screen("S01_목록_기본", [topbar("스터디룸")] + [
        row(f"Row{i}", f"{i}번 방 · 4인", instance("Button/Primary")) for i in range(1, 5)])
    s2 = screen("S02_상세_기본", [topbar("3번 방"), text("info", "창가 자리 · 화이트보드 있음", sx="fill"), cta("예약하기")])
    s3 = screen("S03_예약_기본", [topbar("예약"), field("Date", "날짜를 골라 주세요"), field("Time", "시간을 골라 주세요"), cta("예약 확정")])
    s4 = screen("S04_완료_기본", [topbar("완료"), text("msg", "예약이 끝났어요", sx="fill"),
                                 cta("목록으로", **({"fill": "#8B949E"} if overrides_drift else {}))])
    doc = {"mains": mains, "screens": [s1, s2, s3, s4]}
    if detach_drift:
        inst = doc["screens"][2]["children"][-1]
        detach(doc, inst)
    return doc


def resolve(doc, node):
    """instance를 실제 레이어 트리로 펼친다. main 값 위에 override를 얹는다."""
    if node.get("kind") == "instance":
        base = copy.deepcopy(doc["mains"][node["main"]])
        for path, v in node["overrides"].items():
            _set_path(base, path, v)
        base["sx"] = node.get("sx", base.get("sx"))
        if node.get("intent"):
            base["intent"] = node["intent"]
        base["origin"] = "instance"
        for n, _ in L.walk(base):
            n["from_library"] = True
        return base
    out = {k: v for k, v in node.items() if k != "children"}
    if "children" in node:
        out["children"] = [resolve(doc, c) for c in node["children"]]
    return out


def _set_path(tree, path, v):
    if "." not in path:
        tree[path] = v
        return
    layer, prop = path.split(".", 1)
    for n, _ in L.walk(tree):
        if n.get("name") == layer:
            n[prop] = v


def detach(doc, inst):
    """instance를 그 순간의 모양 그대로 보통 프레임으로 바꾼다. main과의 연결은 끊긴다."""
    snap = resolve(doc, inst)
    snap.pop("origin", None)
    for n, _ in L.walk(snap):
        n.pop("from_library", None)
    snap["detached_from"] = inst["main"]
    inst.clear()
    inst.update(snap)


def instances(doc):
    for s in doc["screens"]:
        for n, _ in L.walk(s):
            if n.get("kind") == "instance":
                yield s["name"], n


def detached(doc):
    return [(s["name"], n) for s in doc["screens"] for n, _ in L.walk(s) if n.get("detached_from")]


def change_main(doc, changes):
    """main component의 속성을 바꾼다. changes = {"fill": ..., "radius": ...}. 고친 위치 수 1을 돌려준다."""
    for path, v in changes.items():
        _set_path(doc["mains"]["Button/Primary"], path, v)
    return 1


def stale_report(doc, changes):
    """main 변경 뒤, 화면에 보이는 버튼 가운데 새 값을 받지 못한 곳을 이유별로 센다."""
    overridden, detached_stale, ok = [], [], 0
    for sname, s in [(s["name"], s) for s in doc["screens"]]:
        for n, _ in L.walk(s):
            if n.get("kind") == "instance":
                hit = [p for p in changes if p in n["overrides"]]
                (overridden.append((sname, hit)) if hit else None)
                ok += 0 if hit else 1
            elif n.get("detached_from"):
                miss = [p for p, v in changes.items() if _get_path(n, p) != v]
                (detached_stale.append((sname, miss)) if miss else None)
    return {"updated": ok, "kept_by_override": overridden, "missed_by_detach": detached_stale}


def _get_path(tree, path):
    if "." not in path:
        return tree.get(path)
    layer, prop = path.split(".", 1)
    for n, _ in L.walk(tree):
        if n.get("name") == layer:
            return n.get(prop)


def resolved_screens(doc):
    return [resolve(doc, s) for s in doc["screens"]]


# ---------------------------------------------------------------------------
# 핑퐁 파일: 375px에서 명세 파일과 똑같이 보이지만, 버튼은 사본이고 좌표는 고정이다.
# ---------------------------------------------------------------------------

PINGPONG_NAMES = ["목록화면_최종", "상세_최종_수정", "예약화면_최종_수정2", "완료(진짜최종)"]


def pingpong_file():
    doc = spec_file()
    screens = []
    for s, nm in zip(resolved_screens(doc), PINGPONG_NAMES):
        laid = L.layout(s, 0, 0, SCREEN_W, SCREEN_H)
        frozen = L.freeze(laid)
        frozen["name"] = nm
        for n, _ in L.walk(frozen):  # 사본이니 라이브러리 출신 표시가 없다
            n.pop("from_library", None)
            if n.get("origin") == "instance":
                n["origin"] = "copy"
        screens.append(frozen)
    return {"mains": {}, "screens": screens}


def buttons_in(doc):
    """화면에 그려진 버튼 수(사본이든 instance든)."""
    c = 0
    for s in doc["screens"]:
        for n, _ in L.walk(s):
            if n.get("kind") == "instance" or n.get("origin") in ("copy",) or n.get("detached_from"):
                c += 1
    return c


def edit_locations_copy(doc, props):
    """사본 방식에서 버튼 스타일 하나를 바꾸려면 버튼마다 그 속성을 고쳐야 한다."""
    return buttons_in(doc) * len(props)


# ---------------------------------------------------------------------------
# 명세 검사 (specs/figma_file_spec.md 의 검증 절차를 코드로)
# ---------------------------------------------------------------------------

def lint(doc, widths=(375, 430)):
    names_bad = [s["name"] for s in doc["screens"] if not L.FRAME_NAME.match(s["name"])]
    no_auto = []
    off_token = []
    screens = [resolve(doc, s) for s in doc["screens"]]
    for s in screens:
        for n, _ in L.walk(s):
            if n["kind"] == "frame" and n.get("children") and not n.get("layout"):
                no_auto.append(f'{s["name"]}/{n["name"]}')
            if not n.get("layout"):
                continue  # 간격 토큰은 auto layout 프레임의 padding·gap에만 매긴다
            for v in list(n.get("pad", ())) + ([n["gap"]] if "gap" in n else []):
                if v % 4:
                    off_token.append(f'{s["name"]}/{n["name"]}')
    per_width = {}
    for w in widths:
        bad = {"overflow": 0, "stretch_fail": 0}
        for s in screens:
            r = L.check_layout(L.layout(L.resize_screen(s, w), 0, 0, w, SCREEN_H))
            bad["overflow"] += len(r["overflow"])
            bad["stretch_fail"] += len(r["stretch_fail"])
        per_width[str(w)] = bad
    return {
        "frame_name_violations": len(names_bad),
        "frames_without_auto_layout": len(no_auto),
        "spacing_off_token": len(set(off_token)),
        "detached_instances": len(detached(doc)),
        "width_checks": per_width,
    }


def adoption_score(doc):
    """Pinterest Gestalt 팀의 방식(라이브러리 출신 레이어 수 ÷ 전체 레이어 수)을 이 파일에 적용한다.
    출처: Ravi Lingineni, Figma 블로그, 2023-02-03 (확인일 2026-10-08)."""
    total = lib = 0
    for s in resolved_screens(doc):
        for n, _ in L.walk(s):
            if n is s:
                continue
            total += 1
            lib += 1 if n.get("from_library") else 0
    return {"library_layers": lib, "total_layers": total, "score": lib / total}
