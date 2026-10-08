"""생산성 도구 4장 검증. `uv run pytest -q productivity/ch04_notion_relations` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import json
import random
from pathlib import Path

import productivity_ch04_data as DATA
import productivity_ch04_pages as P
import productivity_ch04_relational as R

HERE = Path(__file__).parent


def test_first_screen_hand_numbers():
    # 결정 가(취소 규정)에 할 일 3개, 그중 완료 1개 -> 미완료 3 - 1 = 2
    db = {"decisions": {"가": {}, "나": {}, "다": {}},
          "tasks": {"1": {"status": "완료"}, "2": {"status": "진행"}, "3": {"status": "할 일"},
                    "4": {"status": "완료"}, "5": {"status": "진행"}},
          "task_dec": {("1", "가"), ("2", "가"), ("3", "가"), ("4", "나"), ("5", "다")}}
    assert R.rollup_open_tasks(db) == {"가": 2, "나": 0, "다": 1}
    assert R.rollup_open_tasks_formula(db) == {"가": 2, "나": 0, "다": 1}


def test_data_sizes_match_design():
    assert (len(DATA.DECISIONS), len(DATA.TASKS), len(DATA.SPECS)) == (10, 20, 4)
    assert len(DATA.links()) == 10 + 12 + 22 == 44


def test_truth_answers():
    a = R.answer_all(R.build())
    assert a["Q1"] == ["T08", "T09", "T10", "T14"]
    assert a["Q2"] == ["D01", "D04", "D05", "D07", "D10"] and len(a["Q2"]) == 5
    assert a["Q3"] == ["D02", "D04", "D07", "D08", "D09", "D10"]
    assert a["Q5"] == ["S02", "S03", "S04"]


def test_check1_questions_answered_by_filters_and_rollups():
    # 설계도 검증 ①: 질문 5개를 필터·rollup만으로 답한다(r = 1.0이면 모두 정답)
    db = R.build()
    sq = R.sql_answer_all(R.to_sqlite(db))
    assert {q: sq[q] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")} == R.answer_all(db)


def test_check2_rollup_matches_manual_filter_count():
    # 설계도 검증 ②: rollup 값 = 손으로 필터해서 센 개수
    db = R.build()
    roll = R.rollup_open_tasks(db)
    for d in DATA.DECISIONS:
        manual = sum(1 for t, v in DATA.TASKS.items() if d in v["decisions"] and v["status"] != DATA.DONE)
        assert roll[d] == manual


def test_check3_reversal_is_linked_both_ways():
    # 설계도 검증 ③: 번복된 결정이 원래 결정과 양방향으로 연결
    assert DATA.DECISIONS["D10"]["supersedes"] == "D07"
    superseded_by = {v["supersedes"]: d for d, v in DATA.DECISIONS.items() if v["supersedes"]}
    assert superseded_by == {"D07": "D10"}
    pages = P.build_pages()
    assert "번복: D07" in pages["decisions"]["D10"]
    assert not any("번복" in ln for ln in pages["decisions"]["D07"])  # 페이지 방식은 옛 결정 쪽에 안 보인다


def test_check4_no_task_without_relation():
    # 설계도 검증 ④: relation 없는 Task 0개
    assert R.empty_checks(R.build()) == {"task_no_decision": [], "decision_no_owner": [], "decision_no_spec": []}


def test_pages_cannot_answer_q5_but_answer_others_at_full_recording():
    ans, lines = P.answer_all(P.build_pages())
    tru = R.answer_all(R.build())
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert ans[q] == tru[q]
    assert ans["Q5"] == P.UNANSWERABLE
    assert sum(lines.values()) == 21 + 76 + 97 + 76 + 29


def test_same_lost_links_same_answers_q1_to_q4():
    rng = random.Random(0)
    for _ in range(100):
        lost = frozenset(l for l in DATA.links() if rng.random() < 0.3)
        pa, _ = P.answer_all(P.build_pages(lost))
        ra = R.answer_all(R.build(lost))
        assert pa["Q2"] == ra["Q2"] and pa["Q3"] == ra["Q3"]
        assert pa["Q1"] == (ra["Q1"] or P.UNANSWERABLE)


def test_empty_check_misses_second_link():
    lost = frozenset([("task", "D10", "T13")])
    assert R.detected_links(R.build(lost), lost) == set()
    lost = frozenset([("task", "D04", "T10")])
    assert R.detected_links(R.build(lost), lost) == lost


def test_connection_trap():
    db = R.build()
    assert set(R.q2_via_tasks(db)) - set(R.q2(db)) == {"D09"}
    assert set(R.q2(db)) - set(R.q2_via_tasks(db)) == {"D04"}


def test_sqlite_matches_python_on_random_dbs():
    rng = random.Random(1)
    for _ in range(100):
        nd, nt = rng.randint(2, 8), rng.randint(3, 15)
        db = {"decisions": {f"D{i}": {"title": "", "date": "", "owner": None, "supersedes": None} for i in range(nd)},
              "tasks": {f"T{i}": {"title": "", "status": rng.choice(["완료", "진행"]), "owner": "",
                                  "updated": f"2026-09-{rng.randint(1, 30):02d}"} for i in range(nt)},
              "specs": {"S0": {"title": "", "edited": "2026-09-15"}},
              "task_dec": set(), "dec_spec": set(), "task_spec": set()}
        for t in db["tasks"]:
            if rng.random() < 0.8:
                db["task_dec"].add((t, f"D{rng.randrange(nd)}"))
            if rng.random() < 0.5:
                db["task_spec"].add((t, "S0"))
        sq = R.sql_answer_all(R.to_sqlite(db), "D0", "S0")
        assert sq["rollup_open"] == R.rollup_open_tasks(db) == R.rollup_open_tasks_formula(db)
        assert sq["Q5"] == R.q5(db) and sq["Q3"] == R.q3(db)


def test_results_file_numbers():
    res = json.loads((HERE / "results" / "ch04.json").read_text())
    assert res["e1_min_failure"]["pages_lines_read_total"] == 299
    assert res["e7_cross"]["mismatches"] == {k: 0 for k in res["e7_cross"]["mismatches"]}
    for r in ("0.9", "0.8", "0.6"):
        m = res["e2_rates"][r]["mean_correct_of_5"]
        assert m["pages"] <= m["relational"] <= m["relational_fixed"]
    assert all(v["rollup_mismatch_steps"] == 0 for v in res["e6_drift"].values())
