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
# experiments.py가 만든 결과 파일. 아래 테스트가 지금 코드 결과와 같은지 대조한다.
RES = json.loads((HERE / "results" / "ch04.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pa():
    return C.load()


# ---------- 판단표 ----------

def test_decisions():
    got = {r["id"]: (r["where"], r["rule"]) for r in D.table()}
    assert got == {
        # 기대 판단표: 비밀을 쓰는 기능(AI 배경, 최종 PNG)은 규칙 1, 다른 기기에서도 남아야 하는 보관 저장은 규칙 2, 나머지는 규칙 3.
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
    # 키 20번 x 6묶음 = 120번. 한 묶음은 19x120 = 2280ms + 묶음 간격 2000ms = 4280ms. 마지막 키 시각 = 5 x 4280 + 2280 = 23680.
    assert len(t) == 120 and t[-1] == 23680
    sent = C.debounced_requests(t, 300)
    assert len(sent) == 6  # 묶음 6개, 묶음마다 마지막 키 뒤 300ms에 한 번
    # 첫 요청 = 첫 묶음 마지막 키(19 x 120ms) + 디바운스 300ms.
    assert sent[0] == 19 * 120 + 300
    # 키 간격(120ms)보다 짧게 기다리면 모든 키가 요청이 되어 줄지 않는다.
    assert len(C.debounced_requests(t, 100)) == 120  # 키 간격 120ms보다 짧게 기다리면 줄지 않는다


def test_requests_per_card(pa):
    _, a = pa
    got = {d: C.usage_per_card(a, d)["requests"] for d in C.DESIGNS}
    # 14 = 페이지 8 + 템플릿 3 + AI 2 + 보관 1. decided는 PNG 1 추가(15). 서버 미리보기는 +120(135), 디바운스는 +6(21).
    assert got == {"png_in_browser": 14, "decided": 15, "server_preview": 135, "server_preview_debounced": 21}
    # 페이지 열기 7,659바이트는 요청 8개의 합이다. 8을 한 번 더 곱하면 안 된다.
    # 7659(페이지) + 3 x 255(템플릿) + 2 x 1.5MB(AI) + 200(보관) + 1.8MB(PNG) = 사용자가 받는 바이트.
    assert C.usage_per_card(a, "decided")["bytes_to_visitor"] == 7659 + 3 * 255 + 2 * 1_500_000 + 200 + 1_800_000


# ---------- 대기 시간 ----------

def test_latency(pa):
    _, a = pa
    lt = C.latency_table(a)
    # 손계산: RTT 40 + 올리기 0.48ms(600B x 8 / 10Mbps) + 서버 30 + 내려받기 32ms(200KB x 8 / 50Mbps) = 102.48ms.
    assert lt["preview_server"] == F(40) + F(48, 100) + 30 + 32 == F(10248, 100)
    assert lt["preview_server"] > D.INSTANT_MS > lt["preview_browser"]
    assert lt["ai_background_server"] == F(833024, 100)
    # 미국 동부는 RTT 200ms라 160ms가 더해져 262.48ms.
    assert C.latency_table(a, "iad1")["preview_server"] == F(26248, 100)


# ---------- 서버비 ----------

def test_vercel_month(pa):
    p, a = pa
    m = C.monthly_usage(a, "decided")
    c = C.vercel_cost(p, a, m)
    assert m["cards"] == 20000 and m["requests"] == 300000 and m["function_requests"] == 80000
    # 한 달 서버 CPU 합 18,100,000ms를 시간으로 환산. 함수 요청 8만 건의 cpu_ms 합이다.
    assert c["cpu_hours"] == F(18100000, 3600000)
    # 사용량 요금 합 26.5801달러(가격표·가정값 기준). float 비교라 소수 넷째 자리까지 반올림해서 본다.
    assert round(float(c["usage"]), 4) == 26.5801
    # 청구 = 기본료 20달러 + 크레딧(20달러) 초과분. 분수 계산이라 정확히 같다.
    assert c["bill"] == 20 + (c["usage"] - 20)  # 크레딧 20달러를 넘은 만큼만 더 낸다
    sp = C.vercel_cost(p, a, C.monthly_usage(a, "server_preview"))
    # 서버 미리보기 설계는 요청이 270만으로 첫 CDN 등급(100만 요청)을 넘어 다음 등급 요금이 붙는다.
    assert sp["cdn_tier_usd"] == 20  # 270만 요청 > 100만 → 다음 등급
    assert round(float(sp["bill"]), 4) == 167.5057


def test_ai_wait_is_memory_not_cpu(pa):
    """AI 배경은 8초 기다리는 동안 CPU 요금은 멈추고 메모리 요금은 계속 나간다."""
    p, a = pa
    one = RES["e4_one_request_icn1"]["ai_background"]
    # AI 배경은 8초 동안 CPU를 거의 쓰지 않지만 함수는 살아 있어 메모리 요금만 계속 쌓인다. 비가 20배 이상.
    assert one["memory_usd"] > 20 * one["active_cpu_usd"]


def test_hobby_and_netlify(pa):
    p, a = pa
    m = C.monthly_usage(a, "decided")
    over = [k for k, v in C.hobby_check(p, a, m).items() if v["over"]]
    # Hobby 포함량을 넘는 항목 두 가지(CPU 시간, Fast Origin Transfer).
    assert over == ["active_cpu_hours", "fast_origin_transfer_gb"]
    n = C.netlify_credits(p, a, m)
    # Netlify 크레딧 합계 2900~3000 범위. 정확한 값은 results 파일 대조 테스트가 본다.
    assert n["plan"] == "Pro" and 2900 < n["total"] < 3000
    assert C.netlify_credits(p, a, C.monthly_usage(a, "server_preview"))["usd"] == 90


# ---------- 저장소 ----------

def test_storage():
    assert S.LOCAL_STORAGE_BYTES == 5_242_880
    # 1.5MB x 4/3 = 2,000,000글자(3으로 나누어떨어지는 크기).
    assert S.base64_chars(1_500_000) == 2_000_000
    assert S.utf16_units("가a") == 2 and S.utf16_units("😀") == 2
    e = RES["e9_storage"]
    # 15는 키 "draft:card-0001"의 글자 수, 2는 글자당 바이트.
    assert e["drafts_that_fit"] == 5_242_880 // ((e["draft_chars_utf16"] + 15) * 2)
    # 2,000,000글자 x 2바이트 = 4MB라 5 MiB(5.24MB)에 한 장만 들어간다.
    assert e["ai_pngs_that_fit"] == 1
    ev = {r["scenario"]: r["evicted"] for r in S.eviction_table()}
    assert ev == {"Chrome, 매일 쓰는 사용자": False,
                  "Chrome, 폰 저장 공간이 꽉 차고 우리 앱을 가장 오래 안 씀": True,
                  "Safari, 매일 쓰지만 우리 앱만 열흘 안 연 사용자": True,
                  "Safari, 우리 앱만 열흘 안 열었지만 영구 저장을 받음": False,
                  "아무 브라우저, 사생활 보호 창을 닫음": True}
    # 10%(12.8GB)보다 그룹 한도 10GiB(약 10.74GB)가 작아서 후자.
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
