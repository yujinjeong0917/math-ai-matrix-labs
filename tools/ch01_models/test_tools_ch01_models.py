"""AI 도구 실전 1장 검증. `uv run pytest -q tools/ch01_models` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import csv
import json
import math
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch01_answers as A
import tools_ch01_cost as C

HERE = Path(__file__).parent
RES = json.loads((HERE / "results" / "ch01.json").read_text(encoding="utf-8"))


# ---------- 가격표 ----------

def test_price_file_has_source_and_date_for_every_entry():
    p = C.load_prices()
    assert p["checked"] == "2026-10-10" and C.PRICE_FILE.name == "prices_2026-10-10.json"
    for m in p["models"]:
        assert m["checked"] == "2026-10-10"
        for k in ("url", "url_price", "url_thinking"):
            assert m[k].startswith("https://")
        assert m["weights"]["url"].startswith("https://")
        for k in ("url", "url_training", "url_region"):
            assert m["data"][k].startswith("https://")


def test_unknown_cells_are_marked():
    for m in C.load_prices()["models"]:
        unknown = m["knowledge_cutoff"] is None or m["max_output"] is None or m["data"]["trains_on_api"] == "unknown"
        if unknown:
            assert "[확인 필요]" in m["note"] or m["note"] == "위와 같다."


def test_prices_match_chapter2_price_file():
    ch2 = json.loads((HERE.parent / "ch02_api_caching" / "data" / "prices_2026-10-10.json").read_text(encoding="utf-8"))
    for m2 in ch2["models"]:
        try:
            m1 = C.price_of(m2["model"])
        except KeyError:
            continue
        assert (m1["input"], m1["output"]) == (m2["input"], m2["output"])


def test_fixed_vendor_order_not_a_ranking():
    provs = [m["provider"] for m in C.sorted_models()]
    assert provs == sorted(provs, key=C.ORDER.index)
    assert C.ORDER == ["Anthropic", "OpenAI", "Google", "DeepSeek", "xAI"]


def test_weights_and_licenses():
    assert C.price_of("deepseek-v4-pro")["weights"] == {**C.price_of("deepseek-v4-pro")["weights"], "public": True, "license": "MIT"}
    assert C.price_of("deepseek-flash")["weights"]["license"] == "MIT"
    for name in ("claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.8-flash", "grok-4.7"):
        assert C.price_of(name)["weights"]["public"] is False


# ---------- 첫 화면 손계산 ----------

def test_first_screen_hand_numbers():
    w = C.load_workload()
    assert C.input_tokens_per_call(w) == 32_150
    assert F(32_150 * 2, 10**6) == F("0.0643") and F(700 * 10, 10**6) == F("0.007")
    m = C.price_of("claude-sonnet-5-5")
    assert C.call_cost(m, 32_150, 700) == F("0.0713")
    assert C.week_cost(m, w) == F("0.8556")
    assert C.weeks_per_month(w) == F(13, 3)
    assert round(float(C.month_cost(m, w)), 4) == 3.7076
    assert C.week_cost(m, w) * 52 == F("44.4912")


def test_no_cache_week_equals_chapter2_table():
    # 2장 7절 표의 "출력 포함 합계"에서 화살표 왼쪽(캐싱 없음) 값
    expect = {"claude-sonnet-5-5": "0.8556", "claude-opus-5-5": "1.7112", "claude-haiku-5-5": "0.04278",
              "gpt-6.1-sol": "0.8556", "gpt-6-astra": "4.278", "gemini-3.8-flash": "0.32085"}
    w = C.load_workload()
    for name, v in expect.items():
        assert C.week_cost(C.price_of(name), w) == F(v)


# ---------- 단가 규칙 ----------

def test_long_prompt_rules():
    hk, sol, grok, son = (C.price_of(n) for n in ("claude-haiku-5-5", "gpt-6.1-sol", "grok-4.7", "claude-sonnet-5-5"))
    assert C.unit_prices(hk, 100_000)[:2] == (F("0.1"), F("0.5"))
    assert C.unit_prices(hk, 100_001)[:2] == (F("0.5"), F("2.5"))
    assert C.unit_prices(sol, 272_000)[:2] == (F(2), F(10))
    assert C.unit_prices(sol, 272_001)[:2] == (F(4), F(15))
    assert C.unit_prices(grok, 199_999)[:2] == (F(2), F(6))
    assert C.unit_prices(grok, 200_000)[:2] == (F(4), F(12))
    assert C.unit_prices(son, 900_000)[:2] == (F(2), F(10))


@pytest.mark.parametrize("day,hour,peak", [(0, 9, False), (0, 10, True), (0, 12, True), (0, 13, False),
                                           (0, 14, False), (0, 15, True), (0, 18, True), (0, 19, False),
                                           (4, 18, True), (5, 10, False), (6, 15, False), (0, 20, False)])
def test_deepseek_peak_in_kst(day, hour, peak):
    assert C.is_peak_kst(day, hour) is peak


def test_offpeak_is_half():
    for n in ("deepseek-v4-pro", "deepseek-flash"):
        m = C.price_of(n)
        assert C.call_cost(m, 32_150, 700, peak=False) * 2 == C.call_cost(m, 32_150, 700, peak=True)


def test_thinking_tokens_raise_cost():
    w = C.load_workload()
    m = C.price_of("claude-sonnet-5-5")
    assert round(float(C.month_cost(m, w, thinking=2000)), 4) == 4.7476
    # 출력이 700 -> 2,700토큰: 한 주 출력값 12 x 2,700 x 10 / 100만 = 0.324달러
    assert C.week_cost(m, w, thinking=2000) - C.week_cost(m, w) == F("0.24")


def test_quarter_prompt_fits_every_window():
    w = C.load_workload()
    it = C.input_tokens_per_call(w, 13)
    assert it == 320_150
    for m in C.load_prices()["models"]:
        assert C.fits(m, it, 700) in (True, None)


# ---------- 조건으로 거르기 ----------

def test_team_conditions():
    rows = {r["model"]: r["status"] for r in RES["e6_choose"]}
    assert rows["deepseek-v4-pro"] == "보류" and rows["deepseek-flash"] == "보류"
    assert rows["gpt-6-astra"] == "탈락"
    assert rows["claude-sonnet-5-5"] == "통과" and rows["grok-4.7"] == "통과"


# ---------- 답이 갈리는 장난감 ----------

def test_softmax_hand_numbers():
    p = A.softmax(A.MODEL_A, 1.0)
    s = math.e**2 + math.e + 1
    assert abs(p["배송"] - math.e**2 / s) < 1e-12
    assert round(s, 3) == 11.107 and round(math.e**2 / s, 3) == 0.665
    p2 = A.softmax(A.MODEL_A, 0.5)
    assert round(p2["배송"], 3) == 0.867 and round(math.e**4 + math.e**2 + 1, 3) == 62.987


def test_sampling_is_reproducible_and_greedy_differs_by_model():
    c1, _ = A.sample(A.MODEL_A, 1.0, 1000, 1)
    c2, _ = A.sample(A.MODEL_A, 1.0, 1000, 1)
    assert c1 == c2 and sum(c1.values()) == 1000
    assert A.greedy(A.MODEL_A) == "배송" and A.greedy(A.MODEL_B) == "환불"
    with pytest.raises(ValueError):
        A.softmax(A.MODEL_A, 0)


# ---------- 결과 파일과 독자 기록 ----------

def test_results_file_matches_recomputation():
    w = C.load_workload()
    for row in RES["e2_month"]:
        m = C.price_of(row["model"])
        assert row["week"] == round(float(C.week_cost(m, w)), 6)
        assert row["month"] == round(float(C.month_cost(m, w)), 4)
    assert RES["e8_crosscheck"]["mismatches"] == 0
    assert RES["e7_answers"]["T1.0"]["counts"] == A.sample(A.MODEL_A, 1.0, 1000, 1)[0]


def test_reader_log_example_rows():
    with open(HERE / "checks" / "ch01.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = C.price_of(row["모델"])
            day = "월화수목금토일".index(row["한국 시각"][0])
            hour = int(row["한국 시각"].split()[1].split(":")[0])
            cost = C.call_cost(m, int(row["input_tokens"]), int(row["output_tokens(생각 포함)"]), C.is_peak_kst(day, hour))
            assert cost == F(row["계산한 금액(달러)"])
