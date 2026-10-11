"""내 데이터로 LLM 파인튜닝하기 1장 검증. `uv run pytest -q finetune/ch01_when_finetune` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산하고,
results/ch01.json(본문 숫자의 출처)이 그 값과 같은지 대조한다. GPU·네트워크·API 키 없이 돈다.
"""

import json
from pathlib import Path

import pytest

import finetune_ch01_compare as C
import finetune_ch01_retrieve as RT

HERE = Path(__file__).parent
RES = json.loads((HERE / "results" / "ch01.json").read_text(encoding="utf-8"))
V1, V2 = C.load("policies_v1.json"), C.load("policies_v2.json")
ITEMS = C.load("inquiries.json")["items"]


# ---------- 입력 파일 ----------

def test_rule_file_dated_and_every_cell_has_basis():
    rules = C.load_rules()
    assert C.RULE_FILE.name == "fix_rules_2026-10-11.json" and rules["checked"] == "2026-10-11"
    for m in "PRF":
        for t in ["K", "F", "T", "I", "R"]:
            cell = rules["rules"][m][t]
            assert cell["basis"]
            assert cell["basis"].startswith("[가정]") or cell.get("url", "").startswith("https://")


def test_inquiry_set_shape():
    assert len(ITEMS) == 40 and len({it["id"] for it in ITEMS}) == 40
    assert C.counts(ITEMS) == {"K": 12, "F": 8, "T": 8, "I": 6, "R": 4, "OK": 2}
    assert all(it["gold"] in {d["id"] for d in V1["docs"]} for it in ITEMS if it["failure"] == "K")


def test_policy_v2_changes_only_refund_and_free_shipping():
    diff = [a["id"] for a, b in zip(V1["docs"], V2["docs"]) if a["text"] != b["text"]]
    assert diff == ["D1", "D2"]
    assert "14일" in V2["docs"][0]["text"] and "5만 원" in V2["docs"][1]["text"]


# ---------- 첫 화면 손계산 ----------

def test_hand_example():
    c = {"K": 3, "F": 2, "T": 2, "I": 2, "R": 1, "OK": 0}
    assert C.hand_remaining(c, {"F": 2, "I": 2}) == 6                      # 프롬프트
    assert C.hand_remaining(c, {"K": 3}) == 7                              # 검색
    assert C.hand_remaining(c, {"F": 2, "T": 2, "I": 2, "K": 1}) == 3      # 파인튜닝
    assert C.hand_remaining(c, {"F": 2, "I": 2, "K": 3}) == 3              # 프롬프트 + 검색
    assert C.hand_remaining(c, {"F": 2, "T": 2, "I": 2, "K": 3}) == 1      # 검색 + 파인튜닝
    assert {r["combo"]: r["remaining"] for r in RES["e0_hand"]["rows"]} == {"P": 6, "R": 7, "F": 3, "PR": 3, "RF": 1}


# ---------- 토큰 예산 ----------

def test_token_budget():
    assert C.tokens_per_call(400, 4, 250, 150) == 1550
    assert C.k_max(8192, 400, 250, 150, 500) == 28          # 7,142 / 250 = 28.57
    assert C.k_max(131072, 400, 250, 150, 500) == 520       # 130,022 / 250 = 520.09
    assert C.tokens_per_call(400, 100, 250, 150) == 25550
    b = RES["e1_budget"]
    assert b["hand_tokens"] == 1550 and b["per_call_finetuned"] == 250
    assert b["monthly_tokens_prompt_T"] == 76_650_000 and b["monthly_tokens_finetuned"] == 750_000
    assert b["ratio_prompt_over_ft"] == 102.2


def test_k_max_is_largest_that_fits():
    for w in (4096, 8192, 32768, 131072):
        k = C.k_max(w, 400, 250, 150, 500)
        assert C.tokens_per_call(400, k, 250, 150) + 500 <= w
        assert C.tokens_per_call(400, k + 1, 250, 150) + 500 > w


# ---------- 검색기 ----------

def test_bigrams():
    assert RT.bigrams("환불을 신청") == {"환불", "불을", "신청"}
    assert RT.bigrams("더 걸려요?") == {"더", "걸려", "려요"}


def test_retrieval_hits_and_misses():
    hits = {it["id"]: RT.top1(it["text"], V1["docs"]) == it["gold"] for it in ITEMS if it["failure"] == "K"}
    assert sum(hits.values()) == 8
    assert [k for k, v in hits.items() if not v] == ["Q02", "Q04", "Q08", "Q11"]
    # "공짜"는 "무료"와 글자가 하나도 겹치지 않는다
    assert RT.score("얼마 넘게 사면 공짜로 보내 주세요?", V1["docs"][1]) == 0
    assert RES["e2_retrieval"]["hits"] == 8 and RES["e2_retrieval"]["hits_v2"] == 8


def test_tie_is_a_miss():
    # Q02는 D1, D3, D7이 1점씩 같아 문서 순서로 고르지 않는다
    q = next(it for it in ITEMS if it["id"] == "Q02")
    assert [s for s, _ in RT.rank(q["text"], V1["docs"])[:3]] == [1, 1, 1]
    assert RT.top1(q["text"], V1["docs"]) is None


# ---------- 비교 ----------

def remaining(model, combo):
    return model.tally(ITEMS, combo)["remaining"]


def test_compare_window_8k():
    m = C.Model(V1, window=8192)
    got = {c: remaining(m, c) for c in C.COMBOS}
    assert got == {"": 38, "P": 24, "R": 30, "F": 7, "PR": 16, "PF": 7, "RF": 5, "PRF": 5}
    assert {r["combo"]: r["remaining"] for r in RES["e3_compare"]["rows"]} == got


def test_reasoning_failures_never_fixed():
    for w in (8192, 131072):
        m = C.Model(V1, window=w)
        for c in C.COMBOS:
            assert m.tally(ITEMS, c)["by_type"]["R"]["ok"] == 0


def test_policy_change_makes_finetuned_answers_stale():
    m = C.Model(V2, trained_on=V1)
    assert [it["id"] for it in ITEMS if m.outcome(it, "F") == "stale"] == ["Q01", "Q02", "Q03", "Q04"]
    assert [it["id"] for it in ITEMS if m.outcome(it, "RF") == "stale"] == ["Q02", "Q04"]
    assert remaining(m, "F") == 11 and remaining(m, "RF") == 7 and remaining(m, "PR") == 16
    # 다시 학습하면(v2로 학습) 옛 값 문제는 사라진다
    m2 = C.Model(V2, trained_on=V2)
    assert m2.tally(ITEMS, "F")["stale"] == 0 and remaining(m2, "F") == 7


def test_big_window_prompt_fixes_tone():
    m = C.Model(V1, window=131072)
    assert m.kmax() == 520
    assert remaining(m, "P") == 16 and remaining(m, "PR") == 8
    assert {r["combo"]: r["remaining"] for r in RES["e5_big_window"]["rows"]}["PR"] == 8


@pytest.mark.parametrize("n,f_v1", [(1, 4), (2, 5), (3, 7), (6, 8)])
def test_sensitivity_stale_count_does_not_depend_on_threshold(n, f_v1):
    rules = C.load_rules()
    rules["params"]["min_fact_count"] = n
    assert remaining(C.Model(V1, rules=rules), "F") == f_v1
    assert C.Model(V2, rules=rules, trained_on=V1).tally(ITEMS, "F")["stale"] == 4


def test_basis_counts():
    b = RES["e7_basis"]
    assert (b["cells"], b["with_url"], b["assumed_only"], b["computed"]) == (15, 6, 9, 3)


# ---------- 본문 연습 문제 ----------

def test_exercises():
    assert C.k_max(32768, 400, 250, 150, 500) == 126            # 31,718 / 250 = 126.87
    docs = json.loads(json.dumps(V1["docs"]))
    docs[1]["text"] += " 공짜"
    q04 = next(it for it in ITEMS if it["id"] == "Q04")
    assert RT.rank(q04["text"], docs)[:2] == [(1, "D2"), (0, "D1")]
    assert RT.top1(q04["text"], docs) == "D2"


def test_full_tables_policy_change_and_big_window():
    m4 = C.Model(V2, trained_on=V1, window=8192)
    got4 = {c: remaining(m4, c) for c in C.COMBOS}
    assert got4 == {"": 38, "P": 24, "R": 30, "F": 11, "PR": 16, "PF": 11, "RF": 7, "PRF": 7}
    assert {r["combo"]: r["remaining"] for r in RES["e4_policy_change"]["rows"]} == got4
    m5 = C.Model(V1, window=131072)
    got5 = {c: remaining(m5, c) for c in C.COMBOS}
    assert got5 == {"": 38, "P": 16, "R": 30, "F": 7, "PR": 8, "PF": 7, "RF": 5, "PRF": 5}
    assert {r["combo"]: r["remaining"] for r in RES["e5_big_window"]["rows"]} == got5


def test_checks_csv_has_eight_columns():
    import csv
    with open(HERE / "checks" / "ch01.csv", encoding="utf-8") as f:
        assert {len(r) for r in csv.reader(f)} == {8}
