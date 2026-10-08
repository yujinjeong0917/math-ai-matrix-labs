"""생산성 도구 3장 검증. `uv run pytest -q productivity/ch03_handoff` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import json
import random
from pathlib import Path

import productivity_ch03_checks as C
import productivity_ch03_model as M

HERE = Path(__file__).parent


def _err(n):
    return n[1] == "오류"


def test_first_screen_hand_numbers():
    # 필요한 상태 칸: 목록 4 + 상세 3 + 예약 3 + 완료 1 = 11, 이미지는 기본 4칸만
    assert sum(len(v) for v in M.REQUIRED_STATES.values()) == 4 + 3 + 3 + 1 == 11
    assert M.completeness([(s, "기본") for s in M.SCREENS]) == (4, 11)
    # 4x4 = 16칸 = 그린 11 + '이 화면엔 없음' 5
    assert len(M.NOT_HERE_REASON) == 5 and 11 + 5 == 16
    # 2배로 내보낸 이미지에서 16px 여백은 32픽셀로 잰다
    assert M.package_image(False)["info"]["measure:cell.pad"] == 32


def test_questions_are_twenty_with_unique_ids():
    ids = [q["id"] for q in M.QUESTIONS]
    assert len(ids) == 20 and len(set(ids)) == 20
    for q in M.QUESTIONS:
        assert q["screen"] in M.SCREENS and q["needs"]


def test_image_handoff_scores():
    s0 = M.score(M.package_image(False))
    s1 = M.score(M.package_image(True))
    assert (s0["correct"], s0["misread"], s0["guessed_right"], s0["guessed_wrong"]) == (3, 3, 2, 12)
    assert (s1["correct"], s1["misread"], s1["guessed_right"], s1["guessed_wrong"]) == (6, 0, 2, 12)
    # 배율 설명이 고치는 건 치수 3개뿐이고, 상태·동작 질문은 그대로 비어 있다
    assert s0["unanswerable"] == s1["unanswerable"] == 14


def test_devmode_on_default_only_file_still_misses_states():
    s = M.score(M.package_devmode(False))
    assert s["misread"] == 1 and s["per_question"]["Q20"] == "misread"  # 그린 24px을 그대로 보여 준다
    for qid in ["Q09", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17"]:
        assert s["per_question"][qid] == "guessed_wrong"
    assert s["unanswerable"] == 11


def test_checklist_packages_answer_all():
    for pkg in (M.package_checklist(), M.package_devmode(True)):
        s = M.score(pkg)
        assert s["correct"] == 20 and M.question_rounds(pkg)["rounds"] == 0
        assert M.decided_cells(pkg["drawn"], pkg["not_here"]) == 16


def test_rounds_rule_one_per_blocked_screen():
    r = M.question_rounds(M.package_image(True))
    assert r["rounds"] == sum(1 for v in r["by_screen"].values() if v)
    assert r["questions"] == 14


def test_graph_checks_known_cases():
    full = C.check_bfs(M.required_cells(), M.FULL_TRANSITIONS, M.START, _err)
    assert full == {"unreachable": [], "dead_ends": [], "error_without_recovery": []}
    # 다시 시도 연결만 빼면 목록 오류가 돌아갈 길 없는 상태가 된다
    edges = [e for e in M.FULL_TRANSITIONS if not (e[0] == ("S01", "오류"))]
    r = C.check_bfs(M.required_cells(), edges, M.START, _err)
    assert ("S01", "오류") in r["dead_ends"] and ("S01", "오류") in r["error_without_recovery"]


def test_bfs_and_warshall_agree_random():
    rng = random.Random(7)
    for _ in range(300):
        ns = list(dict.fromkeys((f"S{i}", rng.choice(["기본", "오류"])) for i in range(rng.randint(2, 9))))
        edges = [(rng.choice(ns), rng.choice(ns)) for _ in range(rng.randint(0, 15))]
        st = rng.choice(ns)
        assert C.check_bfs(ns, edges, st, _err) == C.check_warshall(ns, edges, st, _err)


def test_completeness_two_ways():
    rng = random.Random(11)
    cells = [(s, st) for s in M.SCREENS for st in M.STATES]
    for _ in range(300):
        d = [c for c in cells if rng.random() < 0.5]
        assert M.completeness(d) == M.completeness_by_formula(d)


def test_contrast_reference_values():
    assert abs(C.contrast_ratio("#000000", "#FFFFFF") - 21.0) < 1e-9
    assert abs(C.contrast_ratio("#777777", "#777777") - 1.0) < 1e-12
    # 대칭: 글자와 배경을 바꿔도 같다
    assert C.contrast_ratio("#6B7280", "#FFFFFF") == C.contrast_ratio("#FFFFFF", "#6B7280")
    assert round(C.contrast_ratio("#6B7280", "#FFFFFF"), 2) == 4.83


def test_token_vs_hex():
    assert M.token_change_follow(False) == {"follow": 0, "stale": 7, "edit_locations": 7}
    assert M.token_change_follow(True) == {"follow": 7, "stale": 0, "edit_locations": 1}


def test_results_file_matches_model():
    p = HERE / "results" / "ch03.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())
    for pkg in M.all_packages():
        s = M.score(pkg)
        assert res["e4_compare"][pkg["name"]]["score"]["correct"] == s["correct"]
    assert res["e7_cross_check"]["graph_disagree"] == 0
