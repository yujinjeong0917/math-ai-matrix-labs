"""AI 도구 실전 1장: 모델 비교표 읽기와 "우리 일에 드는 한 달 비용" 계산기.

- 가격표(data/prices_*.json)를 읽는다. 단가는 100만 토큰당 달러.
- 한 번 호출 값 = 입력 토큰 x 입력 단가 + 출력 토큰 x 출력 단가 (각각 100만으로 나눔)
- 단가가 바뀌는 세 가지 규칙을 따른다.
  1) 긴 프롬프트 구간(Haiku 5.5 100,000 초과, OpenAI 272K 초과, Grok 200K 이상): 요청 전체에 다른 단가
  2) 시간대(DeepSeek 피크/비피크): 월~금 UTC 01-04시, 06-10시가 피크, 나머지는 반값
  3) 생각(추론) 토큰: 화면에 안 보여도 출력으로 센다(Anthropic, OpenAI, Google 문서가 밝힘)
- 한 달 = 52주 / 12달 = 13/3주(2장의 "한 해 52주"와 맞추려는 정의).

표준 라이브러리만 쓴다. 금액은 Fraction으로 계산해 반올림 오차가 없다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
PRICE_FILE = HERE / "data" / "prices_2026-10-10.json"
WORKLOAD_FILE = HERE / "data" / "workload.json"
MTOK = 1_000_000
ORDER = ["Anthropic", "OpenAI", "Google", "DeepSeek", "xAI"]   # 고정 순서(순위 아님)


def frac(x):
    return None if x is None else F(str(x))


def load_prices(path=PRICE_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_workload(path=WORKLOAD_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def price_of(model, prices=None):
    prices = prices or load_prices()
    for m in prices["models"]:
        if m["model"] == model:
            return m
    raise KeyError(model)


def weeks_per_month(w=None):
    w = w or load_workload()
    return F(w["weeks_per_year"], w["months_per_year"])


def input_tokens_per_call(w=None, inquiry_weeks=1):
    w = w or load_workload()
    b = w["blocks"]
    return b["rules"] + b["examples"] + b["inquiries"] * inquiry_weeks + b["question"]


def is_peak_kst(weekday_kst, hour_kst):
    """한국 시각(요일 0=월 ... 6=일, 시)이 DeepSeek 피크인지. KST = UTC + 9.
    피크 UTC 01-04, 06-10은 KST 10-13, 15-19이고 날짜가 바뀌지 않는다. 중국 공휴일은 무시(가정)."""
    hour_utc = hour_kst - 9
    weekday_utc = weekday_kst
    if hour_utc < 0:
        hour_utc += 24
        weekday_utc = (weekday_kst - 1) % 7
    if weekday_utc >= 5:
        return False
    return any(a <= hour_utc < b for a, b in [(1, 4), (6, 10)])


def unit_prices(m, prompt_tokens, peak=True):
    """이 요청에 적용될 (입력 단가, 출력 단가, 어느 규칙이 걸렸는지)."""
    i, o = frac(m["input"]), frac(m["output"])
    rule = "기본"
    if m.get("peak") and not peak:
        i, o = frac(m["peak"]["offpeak_input"]), frac(m["peak"]["offpeak_output"])
        rule = "비피크"
    elif m.get("peak"):
        rule = "피크"
    L = m.get("long")
    if L and prompt_tokens > L["over_tokens"]:
        if "input_mult" in L:
            i, o = i * frac(L["input_mult"]), o * frac(L["output_mult"])
        else:
            i, o = frac(L["input"]), frac(L["output"])
        rule = "긴 프롬프트"
    return i, o, rule


def call_cost(m, in_tok, out_tok, peak=True):
    i, o, _ = unit_prices(m, in_tok, peak)
    return (in_tok * i + out_tok * o) / MTOK


def fits(m, in_tok, out_tok):
    """문맥 창 안에 들어가는지. 최대 출력을 모르면 None."""
    if in_tok + out_tok > m["context"]:
        return False
    if m["max_output"] is None:
        return None
    return out_tok <= m["max_output"]


def week_cost(m, w=None, thinking=0, peak=True, inquiry_weeks=1):
    w = w or load_workload()
    n = w["calls_per_week"]
    it = input_tokens_per_call(w, inquiry_weeks)
    ot = w["output_tokens_per_call"] + thinking
    return n * call_cost(m, it, ot, peak)


def month_cost(m, w=None, thinking=0, peak=True):
    w = w or load_workload()
    return week_cost(m, w, thinking, peak) * weeks_per_month(w)


def sorted_models(prices=None):
    prices = prices or load_prices()
    return sorted(prices["models"], key=lambda m: ORDER.index(m["provider"]))


def check_conditions(m, w=None, peak=True):
    """팀 조건 세 개를 차례로 본다. 결과: {'조건': True/False/None}. None은 문서로 확인하지 못함."""
    w = w or load_workload()
    c = w["team_conditions"]
    t = m["data"]["trains_on_api"]
    doc = {"no": True, "paid_no_free_yes": True, "unknown": None}.get(t, False)
    return {
        "no_training_documented": doc,
        "context_ok": m["context"] >= c["min_context_tokens"],
        "budget_ok": month_cost(m, w, peak=peak) <= c["monthly_budget_usd"],
    }
