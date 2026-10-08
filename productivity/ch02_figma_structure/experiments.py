"""생산성 도구 2장 비교 실험. `uv run python productivity/ch02_figma_structure/experiments.py` 로 실행한다.

E1 실패 최소 예제: 두 사람이 사본 두 개를 고친 뒤 파일째 저장 vs 한 파일·속성 단위 동시 편집
E2 변경 비용: 버튼 사본 7개 vs main component 1개, 그리고 override·detach가 남기는 어긋남
E3 화면 폭 375 → 430(과 320): 고정 좌표 vs auto layout, 엔진이 센 넘침·여백 불일치
E4 명세 검사(lint): 핑퐁 파일 vs 명세 파일 vs 명세 파일 + 어긋남
E5 라이브러리 사용률: Pinterest 방식(라이브러리 출신 레이어 ÷ 전체 레이어)
E6 합치는 단위별 손실: 무작위 편집 기록 500개 × 편집량 5단계, 이론값과 비교
E7 대응 구현 대조: 속성 단위 LWW의 두 구현, auto layout 엔진과 닫힌 식

외부 패키지 없이 표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import productivity_ch02_layout as L  # noqa: E402
import productivity_ch02_project as P  # noqa: E402
import productivity_ch02_sync as S  # noqa: E402

OUT = HERE / "results" / "ch02.json"


def _edit(t, who, cell, v):
    return {"t": t, "who": who, "cell": cell, "v": v}


def e1_min_failure():
    pp = P.pingpong_file()
    buttons = []  # (화면, 버튼 번호, 라벨)
    for s in pp["screens"]:
        i = 0
        for n, _ in L.walk(s):
            if n.get("origin") == "copy":
                label = [c for c in n["children"] if c["name"] == "label"][0]["text"]
                buttons.append((s["name"], i, label))
                i += 1
    # 사본 방식: A는 버튼 색(7곳), B는 목록 버튼 문구 "예약"→"예약하기"(4곳)와 버튼 여백(7곳)
    copy_edits, t = [], 0.0
    for sname, i, _ in buttons:
        t += 0.01
        copy_edits.append(_edit(t, "A", (f"{sname}#{i}", "fill"), "#0B7A55"))
    for sname, i, label in buttons:
        if label == "예약":
            t += 0.01
            copy_edits.append(_edit(t, "B", (f"{sname}#{i}", "label.text"), "예약하기"))
    for sname, i, _ in buttons:
        t += 0.01
        copy_edits.append(_edit(t, "B", (f"{sname}#{i}", "pad"), (20, 20, 12, 12)))
    res = {
        "buttons_drawn": len(buttons),
        "copy_edits": {"A": sum(e["who"] == "A" for e in copy_edits), "B": sum(e["who"] == "B" for e in copy_edits)},
        "copy_file_save_B_last": S.score(copy_edits, S.merge(copy_edits, "file", ("A", "B"))),
        "copy_file_save_A_last": S.score(copy_edits, S.merge(copy_edits, "file", ("B", "A"))),
    }
    # 한 파일 + component: A는 main의 색, B는 main의 문구와 여백. 속성 단위 동시 편집.
    comp_edits = [
        _edit(0.10, "A", ("Button/Primary", "fill"), "#0B7A55"),
        _edit(0.11, "B", ("Button/Primary", "label.text"), "예약하기"),
        _edit(0.12, "B", ("Button/Primary", "pad"), (20, 20, 12, 12)),
    ]
    res["component_edits"] = {"A": 1, "B": 2}
    res["component_prop_merge"] = S.score(comp_edits, S.merge(comp_edits, "prop"))
    # 같은 칸을 둘이 고치면 속성 단위에서도 한쪽이 말없이 덮인다(Wallace 2019가 밝힌 충돌)
    clash = comp_edits + [_edit(0.13, "A", ("Button/Primary", "label.text"), "예약할게요")]
    res["component_prop_merge_same_property"] = S.score(clash, S.merge(clash, "prop"))
    # main에 실제로 적용해 화면에서 확인
    doc = P.spec_file()
    P.change_main(doc, {"fill": "#0B7A55", "label.text": "예약하기", "pad": (20, 20, 12, 12)})
    shown = []
    for s in P.resolved_screens(doc):
        for n, _ in L.walk(s):
            if n.get("origin") == "instance":
                lab = [c for c in n["children"] if c["name"] == "label"][0]["text"]
                shown.append((s["name"], n["fill"], lab))
    res["after_main_change_buttons"] = {
        "with_new_fill": sum(f == "#0B7A55" for _, f, _ in shown),
        "list_buttons_new_label": sum(lab == "예약하기" for sn, _, lab in shown if sn.startswith("S01")),
        "cta_labels_kept": sorted(lab for sn, _, lab in shown if not sn.startswith("S01")),
    }
    return res


def e2_change_cost():
    pp, spec = P.pingpong_file(), P.spec_file()
    style = ["fill", "radius"]
    res = {
        "buttons_on_screens": P.buttons_in(spec),
        "style_change_edit_locations": {"copies": P.edit_locations_copy(pp, style), "component": P.change_main(P.spec_file(), {"fill": "#000", "radius": 12}) * len(style)},
        "style_change_places_to_open": {"copies": P.buttons_in(pp), "component": 1},
    }
    # 목록 버튼 문구 "예약" → "예약하기": 사본은 글자 레이어 4개, component는 main 1곳
    pp2 = P.pingpong_file()
    hits = 0
    for s in pp2["screens"]:
        for n, _ in L.walk(s):
            if n["kind"] == "text" and n["name"] == "label" and n["text"] == "예약":
                n["text"] = "예약하기"
                hits += 1
    over_copy = sum(len(L.check_layout(L.layout(s, 0, 0, P.SCREEN_W, P.SCREEN_H))["overflow"]) for s in pp2["screens"])
    spec2 = P.spec_file()
    P.change_main(spec2, {"label.text": "예약하기"})
    over_spec = 0
    widths = []
    for s in P.resolved_screens(spec2):
        laid = L.layout(s, 0, 0, P.SCREEN_W, P.SCREEN_H)
        over_spec += len(L.check_layout(laid)["overflow"])
        widths += [n["box"][2] for n, _ in L.walk(laid) if n.get("origin") == "instance" and s["name"].startswith("S01")]
    res["label_change"] = {
        "text_layers_to_edit": {"copies": hits, "component": 1},
        "overflow_after": {"copies_fixed": over_copy, "component_auto": over_spec},
        "list_button_width": {"before": 16 + L.text_width("예약") + 16, "after_auto": widths[0]},
        "label_width": {"before": L.text_width("예약"), "after": L.text_width("예약하기")},
    }
    # 어긋남: 완료 화면 버튼은 채우기를 override, 예약 화면 버튼은 detach
    drift = P.spec_file(overrides_drift=True, detach_drift=True)
    change = {"fill": "#0B7A55"}
    P.change_main(drift, change)
    rep = P.stale_report(drift, change)
    res["drift_after_fill_change"] = {
        "updated": rep["updated"],
        "kept_by_override": len(rep["kept_by_override"]),
        "missed_by_detach": len(rep["missed_by_detach"]),
        "where": {"override": [s for s, _ in rep["kept_by_override"]], "detach": [s for s, _ in rep["missed_by_detach"]]},
    }
    # Figma 도움말의 체크박스 예: n개를 모든 조합으로 프로토타입하면 프레임 2^n, 연결 n·2^n
    res["variant_combinatorics"] = {str(n): {"frames": 2**n, "connections": n * 2**n} for n in range(1, 7)}
    return res


def _width_breakdown(doc, w):
    out = {"overflow": 0, "stretch_fail": 0, "stretch_fail_names": {}}
    for s in P.resolved_screens(doc):
        r = L.check_layout(L.layout(L.resize_screen(s, w), 0, 0, w, P.SCREEN_H))
        out["overflow"] += len(r["overflow"])
        out["stretch_fail"] += len(r["stretch_fail"])
        for nm in r["stretch_fail"]:
            key = "Row" if nm.startswith("Row") else nm
            out["stretch_fail_names"][key] = out["stretch_fail_names"].get(key, 0) + 1
    return out


def e3_width():
    pp, spec = P.pingpong_file(), P.spec_file()
    stretch_targets = sum(1 for s in P.resolved_screens(spec) for n, _ in L.walk(s) if n.get("intent") == "stretch")
    res = {"stretch_intent_elements": stretch_targets}
    for w in (320, 375, 430):
        res[str(w)] = {"fixed": _width_breakdown(pp, w), "auto": _width_breakdown(spec, w)}
    # 430에서 고정 좌표 요소의 좌우 여백
    s = P.resolved_screens(pp)[0]
    laid = L.layout(L.resize_screen(s, 430), 0, 0, 430, P.SCREEN_H)
    tb = [n for n, _ in L.walk(laid) if n["name"] == "TopBar"][0]
    res["fixed_topbar_margins_at_430"] = {"left": tb["box"][0], "right": 430 - tb["box"][0] - tb["box"][2], "width": tb["box"][2]}
    s = P.resolved_screens(spec)[0]
    laid = L.layout(L.resize_screen(s, 430), 0, 0, 430, P.SCREEN_H)
    tb = [n for n, _ in L.walk(laid) if n["name"] == "TopBar"][0]
    res["auto_topbar_margins_at_430"] = {"left": tb["box"][0], "right": 430 - tb["box"][0] - tb["box"][2], "width": tb["box"][2]}
    return res


def e4_lint():
    return {
        "pingpong": P.lint(P.pingpong_file()),
        "spec": P.lint(P.spec_file()),
        "spec_with_drift": P.lint(P.spec_file(overrides_drift=True, detach_drift=True)),
    }


def e5_adoption():
    return {
        "pingpong": P.adoption_score(P.pingpong_file()),
        "spec": P.adoption_score(P.spec_file()),
        "spec_with_drift": P.adoption_score(P.spec_file(overrides_drift=True, detach_drift=True)),
    }


def e6_sync(n_layers=40, n_props=6, ks=(5, 10, 20, 40, 80), trials=500):
    res = {"n_layers": n_layers, "n_props": n_props, "trials": trials, "by_k": {}}
    for k in ks:
        acc = {m: {"lost": [], "silent": []} for m in ("file", "object", "prop", "baton")}
        waits = []
        for seed in range(trials):
            e = S.make_edits(n_layers, n_props, k, seed=1000 * k + seed)
            for m in acc:
                sc = S.score(e, S.merge(e, m), informed=(m == "baton"))
                acc[m]["lost"].append(sc["lost"])
                acc[m]["silent"].append(sc["silent"])
            waits.append(S.baton_wait(e))
        th = S.expected(n_layers, n_props, k)
        res["by_k"][str(k)] = {
            "measured": {m: {"lost": statistics.mean(v["lost"]), "silent": statistics.mean(v["silent"])} for m, v in acc.items()},
            "theory": {m: th[m] for m in acc},
            "touched_per_person_theory": th["touched_per_person"],
            "prop_trials_with_any_silent": sum(1 for x in acc["prop"]["silent"] if x > 0) / trials,
            "baton_wait_mean": statistics.mean(waits),
        }
    return res


def e7_cross_check(trials=2000):
    rng = random.Random(7)
    mism = 0
    for i in range(trials):
        e = S.make_edits(rng.randint(1, 20), rng.randint(1, 6), rng.randint(1, 30), seed=i)
        mism += S.merge_prop_log(e) != S.merge_prop_state(e)
    lay_mism = 0
    for i in range(trials):
        n = rng.randint(1, 6)
        kids = [P.text(f"t{j}", "가" * rng.randint(0, 6) + " " * rng.randint(0, 2)) for j in range(n)]
        pl, pr, g = (rng.randrange(0, 33, 4) for _ in range(3))
        f = {"name": "F", "kind": "frame", "layout": "H", "pad": (pl, pr, 8, 8), "gap": g, "sx": "hug", "sy": "hug", "children": kids}
        closed = pl + sum(L.text_width(k["text"]) for k in kids) + g * (n - 1) + pr
        lay_mism += L.layout(f)["box"][2] != closed
    hug = {"name": "F", "kind": "frame", "layout": "H", "pad": (10, 10, 0, 0), "gap": 0, "sx": "hug", "sy": "hug",
           "children": [{"name": "t", "kind": "text", "text": "x", "sx": "fixed", "sy": "hug", "w": 40}]}
    a = L.layout(hug)["box"][2]
    hug["children"][0]["w"] = 50
    b = L.layout(hug)["box"][2]
    return {"prop_lww_log_vs_state_mismatch": mism, "prop_trials": trials,
            "layout_vs_closed_form_mismatch": lay_mism, "layout_trials": trials,
            "help_doc_hug_example": {"text40": a, "text50": b}}


def main():
    res = {
        "e1_min_failure": e1_min_failure(),
        "e2_change_cost": e2_change_cost(),
        "e3_width": e3_width(),
        "e4_lint": e4_lint(),
        "e5_adoption": e5_adoption(),
        "e6_sync": e6_sync(),
        "e7_cross_check": e7_cross_check(),
        "env": {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system()},
    }
    OUT.parent.mkdir(exist_ok=True)
    data = HERE / "data"  # 두 파일의 구조를 JSON으로 남긴다(합성, 제3자 권리 없음)
    data.mkdir(exist_ok=True)
    (data / "file_spec.json").write_text(json.dumps(P.spec_file(), ensure_ascii=False, indent=1, default=list))
    (data / "file_pingpong.json").write_text(json.dumps(P.pingpong_file(), ensure_ascii=False, indent=1, default=list))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=list))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=list))


if __name__ == "__main__":
    main()
