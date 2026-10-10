"""AI 도구 실전 8장 검증. `uv run pytest -q tools/ch08_choosing` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch08_checklist as K
import tools_ch08_cost as C
import tools_ch08_data as D
import tools_ch08_injection as I

HERE = Path(__file__).parent


# ---------- 사실 파일 ----------

def test_policy_file_has_url_and_date_for_every_row():
    pol = D.load_policies()
    assert pol["checked"] == "2026-10-10" and D.POLICY_FILE.name == "policies_2026-10-10.json"
    for r in pol["rows"] + pol["injection"]:
        assert r["url"].startswith("https://") and r["checked"] == "2026-10-10", r
    assert len(pol["rows"]) >= 8


def test_unconfirmed_items_are_marked():
    pol = D.load_policies()
    for r in pol["rows"]:
        if r["trains_default"] in ("setting", "plan") or r["retention"] is None:
            assert "[확인 필요]" in r["note"] or r["retention"] is not None, r["product"]


def test_no_em_dash_in_data_and_code():
    for p in list(HERE.glob("*.py")) + list((HERE / "data").glob("*.json")) + [HERE / "README.md"]:
        assert chr(0x2014) not in p.read_text(encoding="utf-8"), p.name


# ---------- 앞 장 스냅숏 ----------

@pytest.mark.parametrize("spec", D.UPSTREAM_SPEC, ids=[s[0] for s in D.UPSTREAM_SPEC])
def test_upstream_snapshot_matches_sibling_results(spec):
    live = D.read_upstream_live(spec[0])
    if live is None:
        pytest.skip(f"{spec[1]} 결과 파일이 없음")
    assert D.load_upstream()[spec[0]] == live


# ---------- 비용 표 ----------

def test_month_rule_matches_chapter1():
    up = D.load_upstream()
    assert C.r4(C.month(up["ch01_week_paste"])) == up["ch01_month_paste"] == 3.7076


def test_cost_table_rows_and_units():
    rows = C.table()
    assert [r["chapter"] for r in rows] == [1, 2, 3, 3, 4, 4, 5, 6, 7, 7]
    assert not any(r.get("pending") for r in rows)
    ch6 = next(r for r in rows if r["chapter"] == 6)
    assert (ch6["correct_none"], ch6["correct_checked"], ch6["month"]) == (0.5117, 0.9339, 4.3333)
    ch7 = [r for r in rows if r["chapter"] == 7]
    assert ch7[0]["month"] == [13.8667, 104.0] and ch7[1]["month"] == [0.2283, 2.3227]
    usd = {(r["chapter"], r["setup"][:6]): r["month"] for r in rows if r["unit"] == "USD"}
    assert 0.8788 in usd.values() and 0.8265 in usd.values() and 2.524 in usd.values()
    ch4 = [r for r in rows if r["chapter"] == 4]
    assert ch4[0]["week"] == C.r4(F(56760) * 2 / 1_000_000) == 0.1135
    ch5 = next(r for r in rows if r["chapter"] == 5)
    assert ch5["unit"] == "calls" and ch5["month"] < ch5["limit"]


def test_hand_example():
    h = C.hand()
    assert h["times52"] == 44.4912 and h["month"] == 3.7076
    assert h["model_range"]["ratio"] == 100.0


# ---------- 주입 장난감 ----------

def _sc(key):
    return next(s for s in I.all_scenarios() if s["key"] == key)


def test_inbox_regenerates_identically():
    assert D.make_inbox() == D.load_inbox()


def test_naive_loop_follows_all_four_hidden_instructions():
    a = _sc("A")
    assert [x[0] for x in a["followed"]] == ["q04", "q09", "q13", "q17"]
    assert a["notion_rows_changed"] == 14 and a["figma_texts_changed"] == 1
    assert a["emails_leaked"] == 20 and a["tamper_reaches_reader"]


def test_keyword_filter_misses_polite_one_and_flags_plain_question():
    b = _sc("B")
    assert b["flagged_for_review"] == ["q04", "q09", "q11", "q13"]
    assert b["false_positives"] == ["q11"]
    assert b["notion_rows_changed"] == 0 and b["report_tampered"]


def test_gate_in_code_blocks_writes_even_for_naive_loop():
    d = _sc("D")
    assert d["notion_rows_changed"] == 0 and d["figma_texts_changed"] == 0 and d["emails_leaked"] == 0
    assert d["approval_prompts"] == d["attempted_calls"] == 22
    assert d["tamper_reaches_reader"]          # 글 속 변조는 확인 문을 지나간다


def test_separate_alone_leaks_and_gate_plus_review_stop_all():
    c, e, f = _sc("C"), _sc("E"), _sc("F")
    assert c["figma_texts_changed"] == 1
    assert e["figma_texts_changed"] == 0 and e["approval_prompts"] == 1 and e["tamper_reaches_reader"]
    assert f["tamper_reaches_reader"] is False
    assert (f["notion_rows_changed"], f["figma_texts_changed"], f["emails_leaked"]) == (0, 0, 0)


def test_yearly_illustration():
    y = I.yearly()
    assert y["n"] == 208 and y["after_separate"] == pytest.approx(10.4) and y["after_gate"] == pytest.approx(0.52)


# ---------- 점검표 ----------

def test_checklist_has_ten_questions_and_no_score():
    cl = K.load_checklist()
    assert len(cl["questions"]) == 10
    for s in K.evaluate():
        assert set(a["verdict"] for a in s["answers"]) <= {K.PASS, K.BLOCK, K.CHECK, K.NA}
        assert "score" not in s and "rank" not in s


def test_checklist_verdicts():
    ev = {s["id"]: s for s in K.evaluate()}
    assert ev["S3"]["blocked_by"] == ["Q5", "Q7", "Q8", "Q9"]
    assert ev["S1"]["blocked_by"] == [] and ev["S1"]["tally"][K.CHECK] == 5
    assert ev["S2"]["tally"][K.PASS] == 9
    assert next(a for a in ev["S4"]["answers"] if a["id"] == "Q4")["verdict"] == K.CHECK


# ---------- 결과 파일 ----------

def test_results_file_matches_modules():
    res = json.loads((HERE / "results" / "ch08.json").read_text(encoding="utf-8"))
    assert res["e1_cost"] == C.table()
    assert res["e3_injection"]["scenarios"] == json.loads(json.dumps(I.all_scenarios(), ensure_ascii=False))
    assert res["e5_checklist"] == K.evaluate()
    assert res["e6_determinism"]["toy_agrees_with_direct_count"] is True
