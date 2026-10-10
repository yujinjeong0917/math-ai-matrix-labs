"""1인 개발 앱 출시 실전 9장 검증. `uv run pytest -q webapp/ch09_ai_coding_era` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 점검표는 webapp_ch09_checklist 로 다시 만든다.
서버를 띄우지 않고, 바깥 네트워크와 API 키 없이 돈다.
"""

import json
from pathlib import Path

import pytest

import webapp_ch09_checklist as L

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def items():
    return L.build_items()


@pytest.fixture(scope="module")
def saved():
    return json.loads((HERE / "results" / "ch09.json").read_text(encoding="utf-8"))


# ---------- 점검표 손계산 ----------

def test_tally(items):
    t = L.tally(items)
    assert t["items"] == 31
    assert (t["recheck"], t["quoted"], t["manual"]) == (9, 11, 11)
    assert t["auto"] == t["recheck"] + t["quoted"] == 20
    assert t["release_pass"] == 20 and t["release_fail"] == []
    assert (t["first_judged"], t["first_pass"]) == (15, 2)
    assert t["first_fail_blocking"] == 11
    assert [t["by_ch"][c]["items"] for c in range(1, 10)] == [3, 2, 5, 3, 5, 3, 4, 4, 2]


def test_saved_results_match_live(items, saved):
    """results/ch09.json 이 지금 다시 만든 점검표와 같다(앞 장 결과가 바뀌면 실패해서 알려 줘요)."""
    assert saved["e1_items"] == json.loads(json.dumps(items, ensure_ascii=False))
    assert saved["e0_tally"] == json.loads(json.dumps(L.tally(items), ensure_ascii=False))
    assert saved["e5_crosscheck"]["all"] is True


def test_every_item_has_a_method_and_gate(items):
    for i in items:
        assert i["how"] in ("다시 검사", "결과 인용", "사람 확인")
        assert i["gate"] in ("막음", "확인")
        if i["how"] == "사람 확인":
            assert i["first"] is None and i["release"] is None
        else:
            assert i["release"] in (True, False)


# ---------- 인용한 값은 지금의 앞 장 JSON과 같다 ----------

def test_quoted_ch03_equals_live_json(items):
    r3 = L.load_result(3)
    by = {i["id"]: i for i in items}
    for k, iid in enumerate(["c3_rls", "c3_ai_key", "c3_admin", "c3_upload", "c3_rate_limit"]):
        row = r3["e0_checklist"]["items"][k]
        assert by[iid]["question"] == row["check"]
        assert (by[iid]["first"], by[iid]["release"]) == (row["before_pass"], row["after_pass"])


def test_quoted_ch04_ch05_ch06_equal_live_json(items):
    by = {i["id"]: i for i in items}
    r4, r5, r6 = L.load_result(4), L.load_result(5), L.load_result(6)
    assert by["c4_month_cost"]["value"]["decided_bill_usd"] == r4["e5_vercel_month"]["decided"]["bill_usd"]
    assert by["c4_month_cost"]["value"]["server_preview_bill_usd"] == r4["e5_vercel_month"]["server_preview"]["bill_usd"]
    assert by["c5_storage_policies"]["value"]["blocked_with_policies"] == r5["e1_blocked_with_policies"]
    assert by["c5_table_writes"]["value"] == r5["e10_tables"]["writes"]
    assert by["c5_free_tier_month"]["value"]["commercial_month1_usd"] == r5["e6_costs"]["commercial_month1"]["total_min"]
    assert by["c6_precache"]["value"]["missing"] == r6["e6_crosscheck"]["precache_missing"]


# ---------- 다시 검사한 값은 앞 장 결과와 같다 ----------

def test_rechecks_agree_with_chapter_results(items):
    by = {i["id"]: i for i in items}
    r2, r6, r7 = L.load_result(2), L.load_result(6), L.load_result(7)
    assert by["c7_release_config"]["value"] == {"first": r7["e1_checker"]["cardnews_mistake"]["blocking"],
                                                "release": r7["e1_checker"]["cardnews_good"]["blocking"]}
    assert by["c7_release_config"]["value"]["first"] == 12
    assert by["c7_signing_machine"]["value"] == {"first": "0/6", "release": "6/6"}
    assert by["c6_manifest"]["value"]["first"]["failed"] == r6["e1_manifests"]["broken_first_draft"]["failed"]
    first_scan = {x["folder"]: x["secret"] for x in by["c2_secret_in_build"]["value"]["first"]}
    assert first_scan["ch02_api_keys/project/dist_mistake"] == r2["e2_builds"]["mistake"]["scan"]["demo"]["summary"]["secret"] == 2


def test_source_map_only_in_first_editor(items):
    by = {i["id"]: i for i in items}
    v = by["c1_source_map"]["value"]
    assert v["first"]["map_files"] == ["ch01_everything_visible/editor/public/app.min.js.map"]
    assert v["release"] == {"map_files": [], "sourcemap_lines": []}


def test_server_only_markers_absent_in_both(items):
    by = {i["id"]: i for i in items}
    v = by["c1_server_only"]["value"]
    assert sum(v["first"].values()) == 0 and sum(v["release"].values()) == 0
    assert v["first"] == L.load_result(1)["e3_server_only_markers_in_downloads"]


def test_shape_scanner_misses_ch03_key_but_ch03_item_catches_it(saved):
    o = saved["e6_overlap"]
    assert (o["ch02_scanner_secret_found"], o["ch03_counted_ai_secret_key"], o["scanner_missed"]) == (0, 1, 1)
    assert o["caught_by_other_item"] is True


# ---------- 8장 수동 항목과 시장 숫자 ----------

def test_ch08_manual_items_have_official_urls():
    d = json.loads(L.CH08_MANUAL.read_text(encoding="utf-8"))
    assert [x["id"] for x in d["items"]] == ["c8_target_platform", "c8_store_fee", "c8_account_cost", "c8_play_closed_test"]
    fees = d["items"][2]["facts"]
    assert (fees["apple_usd_per_year"], fees["google_usd_once"]) == (99, 25)
    assert d["items"][3]["facts"]["testers"] == 12 and d["items"][3]["facts"]["days"] == 14
    for x in d["items"][1:]:
        assert all(v.startswith("https://") for k, v in x["facts"].items() if k.endswith("url"))


def test_ch08_refs_equal_live_json_when_present(items):
    by = {i["id"]: i for i in items}
    if not L.CH08_RESULT.is_file():
        assert all(by[k]["value"]["ch08"] is None for k in ("c8_target_platform", "c8_store_fee", "c8_account_cost"))
        return
    r8 = json.loads(L.CH08_RESULT.read_text(encoding="utf-8"))
    nets = {row["id"]: row["net"] for row in r8["e1_one_payment"]}
    ref = by["c8_store_fee"]["value"]["ch08"]
    assert (ref["net_web_pg"], ref["net_gp_15_tier"]) == (nets["web_pg"], nets["gp_15_tier"])
    assert by["c8_account_cost"]["value"]["ch08"]["ios_no_mac"] == r8["e3_fixed"]["ios_no_mac"]["first_year_cash"]
    for k in ("c8_target_platform", "c8_store_fee", "c8_account_cost", "c8_play_closed_test"):
        assert by[k]["how"] == "사람 확인"


def test_market_ratios_and_bookend(saved):
    m = saved["e4_market_ratios"]
    assert m["launch_growth"] == round(14700 / 2000, 2) == 7.35
    assert m["top10_over_median"] == 35.8
    assert m["share_1k_to_10k_kept"] == round(0.046 / 0.173, 3)
    b = saved["e3_bookend_ch01"]
    assert (b["visible_bytes"], b["server_only_bytes"]) == (8405, 45885)


def test_no_real_secrets_in_this_folder():
    """이 장 폴더의 데이터·결과에는 진짜 키 모양(접두어 + 긴 꼬리)이 없다."""
    import re
    pat = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|sb_secret_[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{30,})")
    for sub in ("data", "results", "checks"):
        for p in (HERE / sub).rglob("*"):
            if p.is_file():
                assert not pat.search(p.read_text(encoding="utf-8")), p
