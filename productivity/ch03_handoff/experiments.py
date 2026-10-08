"""생산성 도구 3장 비교 실험. `uv run python productivity/ch03_handoff/experiments.py` 로 실행한다.

E1 실패 최소 예제: 이미지 + 메신저로만 넘겼을 때 질문 20개(배율 설명 없음 / 있음)
E2 상태 완결성: 정의된 칸 / 필요한 칸, 4x4 칸 중 정한 칸
E3 프로토타입 그래프 검사: 닿지 않는 상태, 막다른 상태, 돌아갈 길 없는 오류(두 구현 대조 포함)
E4 비교 측정: 넘겨주는 방식 5가지의 답 분류, 질문 수·왕복 수, 디자이너가 만든 것의 개수
E5 토큰 이름 vs 값 복사: 버튼 색 토큰을 바꿨을 때 구현 7곳 중 따라오는 곳
E6 접근성 숫자: 글자 대비(WCAG 1.4.3), 터치 영역(WCAG 2.5.8, 2.5.5)
E7 대응 구현 대조: 무작위 그래프 2,000개에서 BFS와 워셜 결과, 완결성 두 계산

외부 패키지 없이 표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import productivity_ch03_checks as C  # noqa: E402
import productivity_ch03_model as M  # noqa: E402

OUT = HERE / "results" / "ch03.json"
SEED = 3


def _is_error(n):
    return n[1] == "오류"


def _graph_of(pkg):
    start = M.START if M.START in pkg["drawn"] else ("S01", "기본")
    return pkg["drawn"], pkg["transitions"], start


def _k(n):
    return M.node_key(n)


def _keys(d):
    return {k: [_k(n) for n in v] for k, v in d.items()}


def e1_min_failure():
    out = {}
    for scale_note in (False, True):
        pkg = M.package_image(scale_note)
        s = M.score(pkg)
        out["scale_note" if scale_note else "no_scale_note"] = {
            "name": pkg["name"],
            "score": {k: v for k, v in s.items() if k != "per_question"},
            "per_question": s["per_question"],
            "by_category": M.by_category(pkg),
            "misread_values": {q["id"]: M.answer(q, pkg)[0] for q in M.QUESTIONS if s["per_question"][q["id"]] == "misread"},
        }
    return out


def e2_completeness():
    out = {}
    for pkg in M.all_packages():
        have, need = M.completeness(pkg["drawn"])
        out[pkg["name"]] = {"defined": have, "required": need, "ratio": round(have / need, 4),
                            "cells_decided_of_16": M.decided_cells(pkg["drawn"], pkg["not_here"]),
                            "transitions": len(pkg["transitions"]), "transitions_required": len(M.FULL_TRANSITIONS)}
    out["required_by_screen"] = {s: M.REQUIRED_STATES[s] for s in M.SCREENS}
    out["not_here"] = {f"{s}:{st}": r for (s, st), r in M.NOT_HERE_REASON.items()}
    return out


def _forward_only():
    """상태는 11칸 다 그렸지만 연결은 '앞으로 가는 길'만 이은 파일(뒤로·다시 시도·고치기 연결 없음)."""
    back_words = ("다시 시도", "뒤로", "고르기", "날짜 바꾸기", "목록으로")
    edges = [e for e in M.FULL_TRANSITIONS if not any(w in e[2] for w in back_words)]
    return M.required_cells(), edges, M.START


def e3_graph():
    cases = {}
    for pkg in M.all_packages():
        nodes, edges, start = _graph_of(pkg)
        a = C.check_bfs(nodes, edges, start, _is_error)
        b = C.check_warshall(nodes, edges, start, _is_error)
        cases[pkg["name"]] = {"nodes": len(set(nodes)), "edges": len(edges), "start": _k(start),
                              "bfs": _keys(a), "agree": a == b}
    nodes, edges, start = _forward_only()
    a = C.check_bfs(nodes, edges, start, _is_error)
    b = C.check_warshall(nodes, edges, start, _is_error)
    cases["상태는 다 그렸고 연결은 앞으로만"] = {"nodes": len(nodes), "edges": len(edges), "start": _k(start),
                                       "bfs": _keys(a), "agree": a == b}
    return cases


def e4_compare():
    out = {}
    for pkg in M.all_packages():
        s = M.score(pkg)
        out[pkg["name"]] = {
            "score": {k: v for k, v in s.items() if k != "per_question"},
            "per_question": s["per_question"],
            "by_category": M.by_category(pkg),
            "asking": M.question_rounds(pkg),
            "designer_cost": M.designer_cost(pkg),
            "token_names_visible": pkg["token_names"],
            "live_values": pkg["live"],
        }
    return out


def e5_token_drift():
    return {"copy_hex": M.token_change_follow(False), "token_name": M.token_change_follow(True),
            "sites": M.BUTTON_SITES, "old": M.TOKENS["color.primary"], "new": "#0A6B4B"}


def e6_accessibility():
    rep = C.accessibility_report()
    rep["sanity"] = {"black_on_white": round(C.contrast_ratio("#000000", "#FFFFFF"), 4),
                     "same_color": round(C.contrast_ratio("#0B7A55", "#0B7A55"), 4)}
    return rep


def e7_cross_check(n=2000):
    rng = random.Random(SEED)
    disagree = 0
    for _ in range(n):
        m = rng.randint(2, 12)
        ns = [(f"S{i:02d}", rng.choice(["기본", "오류"])) for i in range(m)]
        ns = list(dict.fromkeys(ns))
        edges = [(rng.choice(ns), rng.choice(ns), "e") for _ in range(rng.randint(0, 2 * len(ns)))]
        start = rng.choice(ns)
        if C.check_bfs(ns, edges, start, _is_error) != C.check_warshall(ns, edges, start, _is_error):
            disagree += 1
    comp_disagree = 0
    cells = [(s, st) for s in M.SCREENS for st in M.STATES]
    for _ in range(n):
        drawn = [c for c in cells if rng.random() < 0.5]
        if M.completeness(drawn) != M.completeness_by_formula(drawn):
            comp_disagree += 1
    return {"graphs": n, "graph_disagree": disagree, "completeness_samples": n, "completeness_disagree": comp_disagree}


def main():
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": SEED, "checked": "2026-10-08", "questions": len(M.QUESTIONS)},
        "e1_min_failure": e1_min_failure(),
        "e2_completeness": e2_completeness(),
        "e3_graph": e3_graph(),
        "e4_compare": e4_compare(),
        "e5_token_drift": e5_token_drift(),
        "e6_accessibility": e6_accessibility(),
        "e7_cross_check": e7_cross_check(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "questions.json").write_text(json.dumps(M.QUESTIONS, ensure_ascii=False, indent=1))
    (data / "transitions.json").write_text(json.dumps(
        [{"from": _k(a), "to": _k(b), "when": w} for a, b, w in M.FULL_TRANSITIONS], ensure_ascii=False, indent=1))
    for name, v in res["e4_compare"].items():
        s = v["score"]
        print(f"{name:32s} 맞음 {s['correct']:2d} 잘못읽음 {s['misread']:2d} 추측맞음 {s['guessed_right']:2d} "
              f"추측틀림 {s['guessed_wrong']:2d} | 질문 {v['asking']['questions']:2d} 왕복 {v['asking']['rounds']} "
              f"| 그린 칸 {v['designer_cost']['frames_drawn']:2d} 정한 칸 {v['designer_cost']['cells_decided']:2d} 연결 {v['designer_cost']['transitions']:2d}")
    print(json.dumps(res["e3_graph"], ensure_ascii=False)[:1500])
    print(res["e6_accessibility"], res["e7_cross_check"], res["e5_token_drift"])


if __name__ == "__main__":
    main()
