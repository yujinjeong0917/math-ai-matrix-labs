"""생산성 도구 1장 검증. `uv run pytest -q productivity/ch01_source_of_truth` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import copy
import json
import random
import statistics
from pathlib import Path

import productivity_ch01_np as N
import productivity_ch01_project as P
import productivity_ch01_sim as S

HERE = Path(__file__).parent


def test_first_screen_hand_numbers():
    # 자리 3곳: 원본 말고 2곳을 손으로. 한 번 바꾼 뒤 다 맞을 확률 0.9 x 0.9 = 0.81
    assert round(0.9 ** 2, 4) == 0.81
    # 8번 바꾼 뒤에도 한 번도 안 어긋났을 확률 0.81^8
    assert round(0.81 ** 8, 4) == 0.1853
    # 손으로 옮기는 곳이 하나라면 0.9^8
    assert round(0.9 ** 8, 4) == 0.4305
    assert S.generic_exact(3, 0.9, 8)[0] == (0.9 ** 2) ** 8


def test_min_failure_copy_mode():
    r = S.run_copy(P.CARDS)
    assert r["per_step"] == [2, 4, 6, 6, 7, 7, 7, 7]
    assert r["per_step"][-1] == 7                       # 9칸 중 7칸이 진짜 값과 다르다
    assert S.distinct_values(r["state"]) == {"price": 3, "hours": 3, "cancel": 3}
    # 운영 시간의 진짜 값 08-24는 세 자리 어디에도 없다
    assert S.truth_anywhere(r["state"], r["truth"]) == {"price": True, "hours": False, "cancel": True}
    assert P.render("hours", r["truth"]["hours"]) == "08:00-24:00"
    # 08-22(운영 시트), 09-24(기획 문서)는 한 번도 진짜였던 적이 없다
    assert S.never_true_cells(r["state"], P.CARDS) == 2
    assert P.render("hours", r["state"]["sheets"]["hours"]) == "08:00-22:00"
    assert P.render("hours", r["state"]["notion"]["hours"]) == "09:00-24:00"


def test_judging_after_the_fact_fails():
    j = S.judge(S.run_copy(P.CARDS))
    assert j["majority"] == {"right": 0, "wrong": 0, "undecided": 3}
    assert j["latest_file"] == {"right": 0, "wrong": 3, "undecided": 0}
    assert j["rule_place_after_the_fact"]["right"] == 1


def test_register_mode_keeps_truth_at_source():
    r = S.run_register(P.CARDS)
    assert r["per_step"][-1] == 4                       # 손으로 옮기는 4칸이 옛 값
    assert all(r["state"][P.SOURCE[v]][v] == r["truth"][v] for v in P.VALUES)
    assert S.never_true_cells(r["state"], P.CARDS) == 0
    done = S.run_register(P.CARDS, lambda c, p: True)
    assert done["per_step"] == [0] * 8
    # 손으로 옮기는 자리 수: 요금 1, 운영 시간 2, 취소 규정 1
    assert sum(h == "hand" for v in P.LINKS for h in P.LINKS[v].values()) == 4


def test_edit_counts():
    assert sum(S.edits_per_card(c, "copy") for c in P.CARDS) == 24        # 8장 x 3곳
    assert sum(S.edits_per_card(c, "register") for c in P.CARDS) == 3 * 2 + 3 * 3 + 2 * 2  # 19


def test_three_generic_implementations_agree():
    for corr in (False, True):
        for k in (1, 3, 5):
            ex = S.generic_exact(k, 0.8, 8, corr)
            py = S.generic_python(k, 0.8, 8, 6000, seed=3, correlated=corr)
            npv = N.generic_numpy(k, 0.8, 8, 60000, seed=3, correlated=corr)
            for a, b, c in zip(ex, py, npv):
                assert abs(a - b) < 0.03 and abs(a - c) < 0.01


def test_independence_only_changes_all_clean_probability():
    # 변경당 어긋난 자리 기댓값은 독립이든 함께 놓치든 (k-1)(1-q)로 같고, '모두 맞을 확률'만 달라진다
    a, b = S.generic_exact(3, 0.9, 8, False), S.generic_exact(3, 0.9, 8, True)
    assert abs(a[2] - b[2]) < 1e-12 and a[0] < b[0]


def test_montecarlo_copy_matches_formula():
    # 프로젝트 모형의 복사 방식: 변경마다 다른 두 자리 -> 한 번도 안 어긋날 확률 0.81^8
    xs = [S.simulate("copy", 0.9, random.Random(s))["ever_clean"] for s in range(3000)]
    p = statistics.mean(xs)
    se = (0.1853 * (1 - 0.1853) / 3000) ** 0.5
    assert abs(p - 0.81 ** 8) < 4 * se


def test_register_lint():
    assert S.lint_register(P.REGISTER) == []
    assert len(P.REGISTER) == 15
    bad = copy.deepcopy(P.REGISTER)
    bad.append(dict(bad[5], id="I06b"))                 # 시간당 요금을 한 번 더 등록 = 원본 둘
    assert any("2번 등록" in msg for _, msg in S.lint_register(bad))
    bad2 = copy.deepcopy(P.REGISTER)
    bad2[4]["refs"].append(("sheets", "운영표", "typed"))
    assert any("다시 적음" in msg for _, msg in S.lint_register(bad2))


def test_results_file_matches_code():
    path = HERE / "results" / "ch01.json"
    if not path.exists():
        return
    res = json.loads(path.read_text())
    assert res["e1_copy"]["mismatch_cells"] == S.run_copy(P.CARDS)["per_step"][-1]
    assert res["e0_hand"]["after_m_never_wrong"] == 0.1853
    assert res["e4_edits"]["total_8_cards"] == {"copy": 24, "register": 19, "register_ideal": 8}


def test_exercise_c8_in_sheets():
    cards = copy.deepcopy(P.CARDS)
    cards[7]["where"] = "sheets"
    r = S.run_copy(cards)
    assert r["per_step"][-1] == 6
    assert S.never_true_cells(r["state"], cards) == 0
    assert all(S.truth_anywhere(r["state"], r["truth"]).values())


def test_each_card_leaves_one_right_cell_for_changed_value():
    state = {p: copy.deepcopy(P.INITIAL) for p in P.PLACES}
    truth = copy.deepcopy(P.INITIAL)
    for c in P.CARDS:
        truth[c["value"]][c["field"]] = c["new"]
        state[c["where"]][c["value"]][c["field"]] = c["new"]
        right = sum(state[p][c["value"]] == truth[c["value"]] for p in P.PLACES)
        assert right <= 1
    # 운영 시간은 C5 이후 맞는 칸이 없다
    assert sum(state[p]["hours"] == truth["hours"] for p in P.PLACES) == 0


def test_register_hand_formula_and_exercises():
    assert round(((0.9 + 0.81 + 0.9) / 3) ** 8, 3) == 0.328
    assert round(0.729 ** 8, 4) == 0.0798
    assert round(0.64 ** 8, 3) == 0.028 and round(0.8 ** 8, 3) == 0.168
    assert abs((3 - 1) * (1 - 0.8) - 0.4) < 1e-12
