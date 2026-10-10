"""1인 개발 앱 출시 실전 8장 검증. `uv run pytest -q webapp/ch08_platform_choice` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산하고,
results/ch08.json 이 지금 코드·데이터로 다시 낸 값과 같은지 대조한다. 네트워크 없이 돈다.
"""

import csv
import itertools
import json
from pathlib import Path

import pytest

import webapp_ch08_fees as F
import webapp_ch08_select as S

HERE = Path(__file__).parent
RESULTS = json.loads((HERE / "results" / "ch08.json").read_text(encoding="utf-8"))
SOURCES = json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))
FEES = F.load_fees()
SOURCE_IDS = {f["id"] for f in SOURCES["facts"]}


# ---------- 출처와 데이터 파일 ----------

def test_every_fact_has_url_and_date():
    for f in SOURCES["facts"]:
        assert f["url"].startswith("https://"), f["id"]
        assert f["checked"] == "2026-10-10", f["id"]
        assert f["claim"], f["id"]


def test_every_source_id_used_in_code_and_data_exists():
    used = set()
    for ch in FEES["channels"]:
        used |= set(ch["sources"])
    for v in FEES["pg"].values():
        used |= set(v["sources"])
    used |= set(FEES["accounts"]["apple"]["sources"]) | set(FEES["accounts"]["google"]["sources"])
    used |= set(FEES["mac"]["sources"])
    for t in [S.Target(payment=p, pricing=pr, features=("push", "bluetooth", "camera"), has_mac=m,
                       devices_chosen_by_us=d)
              for p, pr, m, d in itertools.product(("none", "ads", "digital", "physical"),
                                                   ("free", "paid_app", "subscription"), (False, True), (False, True))]:
        for p in S.evaluate(t)["platforms"]:
            for i in p["items"]:
                used |= set(i["sources"])
    assert used - SOURCE_IDS == set()


def test_share_files_are_twelve_full_months_that_add_up():
    for name in S.COUNTRY_FILE.values():
        rows = list(csv.DictReader((S.SHARE_DIR / name).open(encoding="utf-8")))
        assert [r["month"] for r in rows][0] == "2025-10" and rows[-1]["month"] == "2026-09"
        assert len(rows) == 12
        for r in rows:
            assert 99.0 <= float(r["android"]) + float(r["ios"]) <= 100.01


# ---------- 수수료 계산 ----------

def test_one_payment_hand_calculation():
    """월 4,900원 한 건: 부가세 445원(4,900 ÷ 11), 남은 4,455원에 스토어 수수료, PG는 4,900원 전체에."""
    rows = {r["id"]: r for r in F.channel_table(4900, FEES)}
    assert rows["web_pg"]["vat"] == 445 and rows["web_pg"]["base"] == 4455
    assert rows["web_pg"]["pg_fee"] == 167 and rows["web_pg"]["net"] == 4288      # 4,900 × 0.034 = 166.6
    assert rows["gp_15_tier"]["store_fee"] == 668 and rows["gp_15_tier"]["net"] == 3786   # 4,455 × 0.15 = 668.2
    assert rows["apple_standard_sub"]["store_fee"] == 1336                            # 4,455 × 0.30 = 1,336.4
    assert rows["apple_sbp"]["net"] == rows["gp_15_tier"]["net"]


def test_alternative_billing_pays_both_fees():
    """대체 결제는 줄어든 스토어 수수료와 PG 수수료를 둘 다 낸다. 4%p 감면이 PG 3.4%와 거의 상쇄된다."""
    rows = {r["id"]: r for r in F.channel_table(4900, FEES)}
    alt = rows["gp_alt_billing_kr"]
    assert alt["store_fee"] > 0 and alt["pg_fee"] > 0
    assert abs(alt["net"] - rows["gp_15_tier"]["net"]) < 50


def test_unverified_rate_stays_unknown_not_zero():
    for ch in FEES["channels"]:
        r = F.split(10000, ch["store_rate"], 0.0)
        if ch["store_rate"] is None:
            assert r["net"] is None and r["unverified"]


def test_subscription_blend_is_per_subscriber_share():
    assert F.blended_store_rate(0.30, 0.15, 0.0) == pytest.approx(0.30)
    assert F.blended_store_rate(0.30, 0.15, 1.0) == pytest.approx(0.15)
    assert F.blended_store_rate(0.30, 0.15, 0.4) == pytest.approx(0.24)
    with pytest.raises(ValueError):
        F.blended_store_rate(0.30, 0.15, 1.2)


def test_fixed_costs_and_threshold():
    fc = F.fixed_costs(FEES, ["ios"], need_new_mac=True)
    assert fc["first_year_cash"] == 99 * FEES["assumptions"]["krw_per_usd"] + FEES["mac"]["krw"]
    assert F.fixed_costs(FEES, ["android"], need_new_mac=True)["first_year_cash"] == 25 * FEES["assumptions"]["krw_per_usd"]
    assert F.threshold_in_krw(FEES)["krw"] == 1_000_000 * FEES["assumptions"]["krw_per_usd"]
    assert F.subscribers_to_cover(100, 10, 1.0) == 10 and F.subscribers_to_cover(101, 10, 1.0) == 11


# ---------- 선택 도구 ----------

def test_order_never_changes():
    for mac, pay, feats in itertools.product((False, True), ("none", "digital", "physical"),
                                             ((), ("bluetooth",), ("push", "camera"))):
        r = S.evaluate(S.Target(has_mac=mac, payment=pay, features=feats))
        assert [p["platform"] for p in r["platforms"]] == ["web", "android", "ios"]
        for p in r["platforms"]:
            assert "score" not in p and "rank" not in p
            assert {i["status"] for i in p["items"]} <= {S.OK, S.NO, S.CHECK}


def test_no_mac_blocks_ios_only():
    r = S.evaluate(S.Target(has_mac=False))
    cand = {p["platform"]: p["candidate_now"] for p in r["platforms"]}
    assert cand == {"web": True, "android": True, "ios": False}
    r = S.evaluate(S.Target(has_mac=True))
    assert all(p["candidate_now"] for p in r["platforms"])


def test_bluetooth_on_guest_phones_blocks_web():
    r = S.evaluate(S.Target(features=("bluetooth",), devices_chosen_by_us=False, has_mac=True))
    web = r["platforms"][0]
    assert not web["candidate_now"]
    assert any(i["item"] == "블루투스 기기 연결" and i["status"] == S.NO for i in web["items"])


def test_paid_download_not_on_web():
    r = S.evaluate(S.Target(pricing="paid_app", has_mac=True))
    assert not r["platforms"][0]["candidate_now"]


# ---------- 결과 파일이 지금 코드와 같은가 ----------

def test_results_match_recomputation():
    assert RESULTS["e1_one_payment"] == F.channel_table(4900, FEES, 0.0)
    assert RESULTS["e2_monthly"]["gross"] == 4900 * 200
    assert RESULTS["e2_monthly"]["share_later_40"] == F.channel_table(4900 * 200, FEES, 0.4)
    assert RESULTS["e4_threshold"]["krw"] == F.threshold_in_krw(FEES)["krw"]
    for c in ("KR", "US", "JP", "WW"):
        assert RESULTS["e0_shares"][c] == S.load_share(c)
    assert RESULTS["e6_order"]["distinct_orders"] == [["web", "android", "ios"]]
