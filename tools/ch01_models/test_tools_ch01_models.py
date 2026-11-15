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
# experiments.py가 만든 결과 파일. 먼저 실험을 한 번 돌려야 존재한다(import 시점에 읽음).
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
    """1장과 2장이 같은 날짜 가격표를 따로 들고 있으므로 단가가 어긋나지 않는지 본다."""
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
    """장 첫 화면의 손계산 숫자와 코드가 일치하는지 본다."""
    w = C.load_workload()
    # 32,150 = 규칙 + 예시 + 문의 + 질문 토큰 블록의 합(workload.json 기준).
    assert C.input_tokens_per_call(w) == 32_150
    # 입력 32,150 x $2/100만 = 0.0643, 출력 700 x $10/100만 = 0.007. 합 0.0713.
    assert F(32_150 * 2, 10**6) == F("0.0643") and F(700 * 10, 10**6) == F("0.007")
    m = C.price_of("claude-sonnet-5-5")
    assert C.call_cost(m, 32_150, 700) == F("0.0713")
    # 0.0713 x 12번 = 0.8556. Fraction 비교라 허용 오차가 필요 없다.
    assert C.week_cost(m, w) == F("0.8556")
    assert C.weeks_per_month(w) == F(13, 3)
    # 0.8556 x 13/3 = 3.7076. float은 반올림해서만 비교한다.
    assert round(float(C.month_cost(m, w)), 4) == 3.7076
    assert C.week_cost(m, w) * 52 == F("44.4912")


def test_no_cache_week_equals_chapter2_table():
    """2장 표의 캐싱 없음 값과 한 주 비용이 같은지 본다."""
    # 2장 7절 표의 "출력 포함 합계"에서 화살표 왼쪽(캐싱 없음) 값
    expect = {"claude-sonnet-5-5": "0.8556", "claude-opus-5-5": "1.7112", "claude-haiku-5-5": "0.04278",
              "gpt-6.1-sol": "0.8556", "gpt-6-astra": "4.278", "gemini-3.8-flash": "0.32085"}
    w = C.load_workload()
    for name, v in expect.items():
        assert C.week_cost(C.price_of(name), w) == F(v)


# ---------- 단가 규칙 ----------

def test_long_prompt_rules():
    """긴 프롬프트 경계값 확인. 임계 바로 위아래에서 단가가 바뀌는지 본다(off-by-one 함정)."""
    hk, sol, grok, son = (C.price_of(n) for n in ("claude-haiku-5-5", "gpt-6.1-sol", "grok-4.7", "claude-sonnet-5-5"))
    assert C.unit_prices(hk, 100_000)[:2] == (F("0.1"), F("0.5"))
    assert C.unit_prices(hk, 100_001)[:2] == (F("0.5"), F("2.5"))
    assert C.unit_prices(sol, 272_000)[:2] == (F(2), F(10))
    assert C.unit_prices(sol, 272_001)[:2] == (F(4), F(15))
    assert C.unit_prices(grok, 199_999)[:2] == (F(2), F(6))
    # xAI는 200,000 이상부터라서 경계 값 자체가 비싼 쪽(다른 회사는 초과부터).
    assert C.unit_prices(grok, 200_000)[:2] == (F(4), F(12))
    # 긴 프롬프트 규칙이 없는 모델은 90만 토큰이어도 기본 단가.
    assert C.unit_prices(son, 900_000)[:2] == (F(2), F(10))


@pytest.mark.parametrize("day,hour,peak", [(0, 9, False), (0, 10, True), (0, 12, True), (0, 13, False),
                                           (0, 14, False), (0, 15, True), (0, 18, True), (0, 19, False),
                                           (4, 18, True), (5, 10, False), (6, 15, False), (0, 20, False)])
def test_deepseek_peak_in_kst(day, hour, peak):
    """피크 UTC 01-04, 06-10은 KST 10-13, 15-19. 경계 시각(9, 13, 14, 19)과 주말을 모두 넣었다."""
    assert C.is_peak_kst(day, hour) is peak


def test_offpeak_is_half():
    """비피크 비용은 피크의 정확히 절반."""
    for n in ("deepseek-v4-pro", "deepseek-flash"):
        m = C.price_of(n)
        assert C.call_cost(m, 32_150, 700, peak=False) * 2 == C.call_cost(m, 32_150, 700, peak=True)


def test_thinking_tokens_raise_cost():
    w = C.load_workload()
    m = C.price_of("claude-sonnet-5-5")
    assert round(float(C.month_cost(m, w, thinking=2000)), 4) == 4.7476
    # 출력이 700 -> 2,700토큰: 한 주 출력값 12 x 2,700 x 10 / 100만 = 0.324달러
    # 증가분만 확인: 2,000토큰 x 12번 x $10/100만 = 0.24달러.
    assert C.week_cost(m, w, thinking=2000) - C.week_cost(m, w) == F("0.24")


def test_quarter_prompt_fits_every_window():
    """문의 13주치 입력 320,150토큰이 모든 모델의 문맥 창에 들어간다(None은 최대 출력을 모르는 경우)."""
    w = C.load_workload()
    it = C.input_tokens_per_call(w, 13)
    assert it == 320_150
    for m in C.load_prices()["models"]:
        assert C.fits(m, it, 700) in (True, None)


# ---------- 조건으로 거르기 ----------

def test_team_conditions():
    """팀 조건 필터 결과. 보류는 데이터 학습 정책을 문서로 확인하지 못한 모델."""
    rows = {r["model"]: r["status"] for r in RES["e6_choose"]}
    assert rows["deepseek-v4-pro"] == "보류" and rows["deepseek-flash"] == "보류"
    assert rows["gpt-6-astra"] == "탈락"
    assert rows["claude-sonnet-5-5"] == "통과" and rows["grok-4.7"] == "통과"


# ---------- 답이 갈리는 장난감 ----------

def test_softmax_hand_numbers():
    p = A.softmax(A.MODEL_A, 1.0)
    s = math.e**2 + math.e + 1
    assert abs(p["배송"] - math.e**2 / s) < 1e-12
    # e^2 + e + 1 = 7.389 + 2.718 + 1 = 11.107, 배송 확률 7.389 / 11.107 = 0.665.
    assert round(s, 3) == 11.107 and round(math.e**2 / s, 3) == 0.665
    p2 = A.softmax(A.MODEL_A, 0.5)
    # T=0.5면 점수를 2배로 키운 것과 같아 분모가 e^4 + e^2 + 1 = 62.987, 확률 0.867(1등 쪽으로 쏠림).
    assert round(p2["배송"], 3) == 0.867 and round(math.e**4 + math.e**2 + 1, 3) == 62.987


def test_sampling_is_reproducible_and_greedy_differs_by_model():
    """같은 시드는 같은 횟수. 온도 0은 오류. 모델마다 1등이 다르다."""
    c1, _ = A.sample(A.MODEL_A, 1.0, 1000, 1)
    c2, _ = A.sample(A.MODEL_A, 1.0, 1000, 1)
    assert c1 == c2 and sum(c1.values()) == 1000
    assert A.greedy(A.MODEL_A) == "배송" and A.greedy(A.MODEL_B) == "환불"
    with pytest.raises(ValueError):
        A.softmax(A.MODEL_A, 0)


# ---------- 결과 파일과 독자 기록 ----------

def test_results_file_matches_recomputation():
    """저장된 결과 파일이 현재 코드의 재계산과 같은지 본다(float은 저장 때와 같은 자릿수로 반올림해 비교)."""
    w = C.load_workload()
    for row in RES["e2_month"]:
        m = C.price_of(row["model"])
        assert row["week"] == round(float(C.week_cost(m, w)), 6)
        assert row["month"] == round(float(C.month_cost(m, w)), 4)
    assert RES["e8_crosscheck"]["mismatches"] == 0
    assert RES["e7_answers"]["T1.0"]["counts"] == A.sample(A.MODEL_A, 1.0, 1000, 1)[0]


def test_reader_log_example_rows():
    """독자 기록용 CSV의 예시 행이 계산기 결과와 맞는지 본다."""
    with open(HERE / "checks" / "ch01.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = C.price_of(row["모델"])
            day = "월화수목금토일".index(row["한국 시각"][0])
            hour = int(row["한국 시각"].split()[1].split(":")[0])
            cost = C.call_cost(m, int(row["input_tokens"]), int(row["output_tokens(생각 포함)"]), C.is_peak_kst(day, hour))
            assert cost == F(row["계산한 금액(달러)"])
