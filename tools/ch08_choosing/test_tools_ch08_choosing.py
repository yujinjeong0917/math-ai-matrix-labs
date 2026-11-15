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


# 학습 기본값이 설정·요금제에 따라 다르거나 보관 기간이 비어 있으면, 확인이 안 된 항목이라는 표시([확인 필요])가 있어야 한다.
def test_unconfirmed_items_are_marked():
    pol = D.load_policies()
    for r in pol["rows"]:
        if r["trains_default"] in ("setting", "plan") or r["retention"] is None:
            assert "[확인 필요]" in r["note"] or r["retention"] is not None, r["product"]


# 저장소 문체 규칙 검사: 엠대시 금지. 코드 포인트로 비교하는 이유는 이 파일 자체에 그 글자를 적지 않으려는 것.
def test_no_em_dash_in_data_and_code():
    for p in list(HERE.glob("*.py")) + list((HERE / "data").glob("*.json")) + [HERE / "README.md"]:
        assert chr(0x2014) not in p.read_text(encoding="utf-8"), p.name


# ---------- 앞 장 스냅숏 ----------

@pytest.mark.parametrize("spec", D.UPSTREAM_SPEC, ids=[s[0] for s in D.UPSTREAM_SPEC])
# 앞 장 결과를 다시 만들면 스냅숏도 다시 만들어야 한다. 어긋나면 이 테스트가 알려 준다.
def test_upstream_snapshot_matches_sibling_results(spec):
    live = D.read_upstream_live(spec[0])
    if live is None:
        pytest.skip(f"{spec[1]} 결과 파일이 없음")
    assert D.load_upstream()[spec[0]] == live


# ---------- 비용 표 ----------

def test_month_rule_matches_chapter1():
    up = D.load_upstream()
    # 0.8556 x 52/12 = 3.7076달러(1장 값).
    assert C.r4(C.month(up["ch01_week_paste"])) == up["ch01_month_paste"] == 3.7076


def test_cost_table_rows_and_units():
    rows = C.table()
    # 행 순서: 1, 2, 3, 3, 4, 4, 5, 6, 7, 7(장마다 설정 수가 다름).
    assert [r["chapter"] for r in rows] == [1, 2, 3, 3, 4, 4, 5, 6, 7, 7]
    assert not any(r.get("pending") for r in rows)
    ch6 = next(r for r in rows if r["chapter"] == 6)
    # 0.97^22 = 0.5117 -> 확인과 승인을 넣으면 0.9339. 6장의 'month' 열은 승인 횟수 1 x 52/12 = 4.3333이고 달러가 아니다.
    assert (ch6["correct_none"], ch6["correct_checked"], ch6["month"]) == (0.5117, 0.9339, 4.3333)
    ch7 = [r for r in rows if r["chapter"] == 7]
    # 영상 한 달 13.8667~104.0달러(주 3.2~24달러 x 52/12), 이미지 0.2283~2.3227달러. 모두 마음에 들 확률 1/4 가정의 기대값.
    assert ch7[0]["month"] == [13.8667, 104.0] and ch7[1]["month"] == [0.2283, 2.3227]
    usd = {(r["chapter"], r["setup"][:6]): r["month"] for r in rows if r["unit"] == "USD"}
    assert 0.8788 in usd.values() and 0.8265 in usd.values() and 2.524 in usd.values()
    ch4 = [r for r in rows if r["chapter"] == 4]
    # 56,760토큰 x 2달러 / 100만 = 0.11352 -> 0.1135.
    assert ch4[0]["week"] == C.r4(F(56760) * 2 / 1_000_000) == 0.1135
    ch5 = next(r for r in rows if r["chapter"] == 5)
    assert ch5["unit"] == "calls" and ch5["month"] < ch5["limit"]


def test_hand_example():
    h = C.hand()
    assert h["times52"] == 44.4912 and h["month"] == 3.7076
    # 가장 싼 모델과 가장 비싼 모델의 한 달 비용 비율이 100배.
    assert h["model_range"]["ratio"] == 100.0


# ---------- 주입 장난감 ----------

def _sc(key):
    return next(s for s in I.all_scenarios() if s["key"] == key)


def test_inbox_regenerates_identically():
    assert D.make_inbox() == D.load_inbox()


def test_naive_loop_follows_all_four_hidden_instructions():
    a = _sc("A")
    assert [x[0] for x in a["followed"]] == ["q04", "q09", "q13", "q17"]
    # 상태가 답변 완료가 아닌 14행, 문구 1곳, 20개 주소 전체 유출. 숨은 지시 4개(q04, q09, q13, q17)를 모두 따른다.
    assert a["notion_rows_changed"] == 14 and a["figma_texts_changed"] == 1
    assert a["emails_leaked"] == 20 and a["tamper_reaches_reader"]


def test_keyword_filter_misses_polite_one_and_flags_plain_question():
    b = _sc("B")
    # 낱말 거르개가 걸러낸 4건 중 q11은 평범한 문의(거짓 양성). 반대로 공손한 q17은 못 걸러 보고서 변조가 남는다.
    assert b["flagged_for_review"] == ["q04", "q09", "q11", "q13"]
    assert b["false_positives"] == ["q11"]
    assert b["notion_rows_changed"] == 0 and b["report_tampered"]


def test_gate_in_code_blocks_writes_even_for_naive_loop():
    d = _sc("D")
    assert d["notion_rows_changed"] == 0 and d["figma_texts_changed"] == 0 and d["emails_leaked"] == 0
    # 호출마다 묻기: 20(상태) + 1(메일) + 1(문구) = 22번 질문. 보고서 변조는 도구 호출이 아니라 확인 문을 지나간다.
    assert d["approval_prompts"] == d["attempted_calls"] == 22
    assert d["tamper_reaches_reader"]          # 글 속 변조는 확인 문을 지나간다


def test_separate_alone_leaks_and_gate_plus_review_stop_all():
    c, e, f = _sc("C"), _sc("E"), _sc("F")
    assert c["figma_texts_changed"] == 1
    # 자료로만 읽기에서 새는 종류(문구 변경, 보고서 변조)만 남고, 종류마다 한 번 묻는 확인 문은 질문 1번으로 문구 변경을 거절한다. 보고서 변조는 사람이 읽어야(F) 잡힌다.
    assert e["figma_texts_changed"] == 0 and e["approval_prompts"] == 1 and e["tamper_reaches_reader"]
    assert f["tamper_reaches_reader"] is False
    assert (f["notion_rows_changed"], f["figma_texts_changed"], f["emails_leaked"]) == (0, 0, 0)


def test_yearly_illustration():
    y = I.yearly()
    # 208 x 0.05 = 10.4, 10.4 x 0.05 = 0.52. 부동소수점이라 approx로 비교한다.
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
    # S3: 쓰기 도구 과다(Q5), 치명적 3요소(Q7), 확인 문 없음(Q8), 사람이 안 읽음(Q9). S1은 막힘 없음 + 확인 필요 5개, S2는 통과 9개(Q8은 쓰기 도구가 없어 해당 없음).
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
