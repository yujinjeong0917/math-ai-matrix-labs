"""1인 개발 앱 출시 실전 4장 검증. `uv run pytest -q webapp/ch04_browser_vs_server` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산하고,
results/ch04.json이 지금 코드의 결과와 같은지도 확인한다. 네트워크와 API 키 없이 돈다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

import pytest

import webapp_ch04_cost as C
import webapp_ch04_decide as D
import webapp_ch04_storage as S

HERE = Path(__file__).parent
RES = json.loads((HERE / "results" / "ch04.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pa():
    return C.load()


# ---------- 판단표 ----------

def test_decisions():
    got = {r["id"]: (r["where"], r["rule"]) for r in D.table()}
    assert got == {
        "edit": ("브라우저", 3), "preview": ("브라우저", 3), "template": ("브라우저", 3),
        "ai_background": ("서버", 1), "export_png": ("서버", 1),
        "draft_save": ("브라우저", 3), "archive_save": ("서버", 2)}
    assert RES["e0_decisions"] == D.table()


def test_secret_overrides_speed():
    """보이면 안 되는 것을 쓰면, 아무리 빨라야 하는 기능이어도 서버로 간다."""
    f = {"id": "x", "name": "x", "touches": ["비밀"], "must_survive_device_change": False,
         "uses_per_card": 500, "budget_ms": 10}
    assert D.decide(f)[0] == "서버"


# ---------- 요청 수와 디바운스 ----------

def test_typing_and_debounce(pa):
    _, a = pa
    t = C.keystroke_times(a["typing"])
    assert len(t) == 120 and t[-1] == 23680
    sent = C.debounced_requests(t, 300)
    assert len(sent) == 6  # 묶음 6개, 묶음마다 마지막 키 뒤 300ms에 한 번
    assert sent[0] == 19 * 120 + 300
    assert len(C.debounced_requests(t, 100)) == 120  # 키 간격 120ms보다 짧게 기다리면 줄지 않는다


def test_requests_per_card(pa):
    _, a = pa
    got = {d: C.usage_per_card(a, d)["requests"] for d in C.DESIGNS}
    assert got == {"png_in_browser": 14, "decided": 15, "server_preview": 135, "server_preview_debounced": 21}


# ---------- 대기 시간 ----------

def test_latency(pa):
    _, a = pa
    lt = C.latency_table(a)
    assert lt["preview_server"] == F(40) + F(48, 100) + 30 + 32 == F(10248, 100)
    assert lt["preview_server"] > D.INSTANT_MS > lt["preview_browser"]
    assert lt["ai_background_server"] == F(833024, 100)
    assert C.latency_table(a, "iad1")["preview_server"] == F(26248, 100)


# ---------- 서버비 ----------

def test_vercel_month(pa):
    p, a = pa
    m = C.monthly_usage(a, "decided")
    c = C.vercel_cost(p, a, m)
    assert m["cards"] == 20000 and m["requests"] == 300000 and m["function_requests"] == 80000
    assert c["cpu_hours"] == F(18100000, 3600000)
    assert round(float(c["usage"]), 4) == 26.5801
    assert c["bill"] == 20 + (c["usage"] - 20)  # 크레딧 20달러를 넘은 만큼만 더 낸다
    sp = C.vercel_cost(p, a, C.monthly_usage(a, "server_preview"))
    assert sp["cdn_tier_usd"] == 20  # 270만 요청 > 100만 → 다음 등급
    assert round(float(sp["bill"]), 4) == 167.5057


def test_ai_wait_is_memory_not_cpu(pa):
    """AI 배경은 8초 기다리는 동안 CPU 요금은 멈추고 메모리 요금은 계속 나간다."""
    p, a = pa
    one = RES["e4_one_request_icn1"]["ai_background"]
    assert one["memory_usd"] > 20 * one["active_cpu_usd"]


def test_hobby_and_netlify(pa):
    p, a = pa
    m = C.monthly_usage(a, "decided")
    over = [k for k, v in C.hobby_check(p, a, m).items() if v["over"]]
    assert over == ["active_cpu_hours", "fast_origin_transfer_gb"]
    n = C.netlify_credits(p, a, m)
    assert n["plan"] == "Pro" and 2900 < n["total"] < 3000
    assert C.netlify_credits(p, a, C.monthly_usage(a, "server_preview"))["usd"] == 90


# ---------- 저장소 ----------

def test_storage():
    assert S.LOCAL_STORAGE_BYTES == 5_242_880
    assert S.base64_chars(1_500_000) == 2_000_000
    assert S.utf16_units("가a") == 2 and S.utf16_units("😀") == 2
    e = RES["e9_storage"]
    assert e["drafts_that_fit"] == 5_242_880 // ((e["draft_chars_utf16"] + 15) * 2)
    assert e["ai_pngs_that_fit"] == 1
    ev = {r["scenario"]: r["evicted"] for r in S.eviction_table()}
    assert ev == {"Chrome, 매일 쓰는 사용자": False,
                  "Chrome, 폰 저장 공간이 꽉 차고 우리 앱을 가장 오래 안 씀": True,
                  "Safari, 매일 쓰지만 우리 앱만 열흘 안 연 사용자": True,
                  "Safari, 우리 앱만 열흘 안 열었지만 영구 저장을 받음": False,
                  "아무 브라우저, 사생활 보호 창을 닫음": True}
    assert S.idb_cap_bytes(128 * 10 ** 9, "Firefox") == 10 * 1024 ** 3  # 10%(12.8GB)보다 그룹 한도 10GiB가 작다


# ---------- 결과 파일이 지금 코드와 같은가 ----------

def test_results_match_code(pa):
    p, a = pa
    for d in C.DESIGNS:
        m = C.monthly_usage(a, d)
        assert RES["e5_vercel_month"][d]["bill_usd"] == round(float(C.vercel_cost(p, a, m)["bill"]), 4)
        assert RES["e7_netlify_month"][d]["total"] == round(float(C.netlify_credits(p, a, m)["total"]), 4)
    assert RES["e10_crosscheck"]["per_card_times_cards_equals_month"] is True
    assert RES["e10_crosscheck"]["hand_equals_model"] is True
