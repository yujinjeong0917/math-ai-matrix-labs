"""AI 도구 실전 2장 검증. `uv run pytest -q tools/ch02_api_caching` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch02_cache as C
import tools_ch02_project as PJ
import tools_ch02_request as RQ

HERE = Path(__file__).parent


# ---------- 가격표 ----------

def test_price_file_has_source_and_date_for_every_entry():
    prices = C.load_prices()
    assert prices["checked"] == "2026-10-10"
    assert C.PRICE_FILE.name == "prices_2026-10-10.json"
    for m in prices["models"]:
        assert m["url"].startswith("https://") and m["url_min_tokens"].startswith("https://")
        assert m["checked"] == "2026-10-10"
    for r in prices["rules"].values():
        assert r["url"].startswith("https://") and r["checked"] == "2026-10-10"


def test_price_multipliers_match_docs():
    # Anthropic: 5분 쓰기 1.25배, 1시간 쓰기 2배, 읽기 0.1배(Opus/Sonnet 5.5는 0.05배)
    for name, read in [("claude-sonnet-5-5", F(1, 20)), ("claude-opus-5-5", F(1, 20)),
                       ("claude-haiku-5-5", F(1, 10)), ("claude-sonnet-4-6", F(1, 10))]:
        m = C.price_of(name)
        assert C.multipliers(m, "5m") == (F(5, 4), read)
        assert C.multipliers(m, "1h") == (F(2), read)
    # OpenAI GPT-5.6 이후: 쓰기 1.25배, 읽기 0.1배(GPT-6.1 Sol은 0.05배)
    assert C.multipliers(C.price_of("gpt-6-astra")) == (F(5, 4), F(1, 10))
    assert C.multipliers(C.price_of("gpt-6.1-sol")) == (F(5, 4), F(1, 20))
    # Gemini: 쓰기 할증 없음(가정), 읽기 0.1배
    assert C.multipliers(C.price_of("gemini-3.8-flash")) == (F(1), F(1, 10))


# ---------- 첫 화면 손계산 ----------

def test_first_screen_hand_numbers():
    P, q = 10_000, 100
    assert F(5, 4) * P + q == 12_600            # 첫 호출(쓰기)
    assert F(1, 10) * P + q == 1_100            # 두 번째부터(읽기)
    assert 2 * (P + q) == 20_200 and 12_600 + 1_100 == 13_700
    assert C.breakeven_calls(F(5, 4), F(1, 10)) == 2


# ---------- 손익분기 ----------

@pytest.mark.parametrize("w,r,n", [(F(5, 4), F(1, 10), 2), (F(2), F(1, 10), 3), (F(5, 4), F(1, 20), 2),
                                   (F(2), F(1, 20), 3), (F(1), F(1, 10), 2), (F(3), F(1, 10), 4)])
def test_breakeven_formula_equals_bruteforce(w, r, n):
    assert C.breakeven_calls(w, r) == n == C.breakeven_bruteforce(w, r)


def test_anthropic_doc_claim_one_read_5m_two_reads_1h():
    # "caching pays off after one cache read for the 5-minute duration (1.25x write),
    #  or after two cache reads for the 1-hour duration (2x write)"
    assert C.breakeven_calls(F(5, 4), F(1, 10)) - 1 == 1
    assert C.breakeven_calls(F(2), F(1, 10)) - 1 == 2


def test_openai_doc_claims():
    # "writing a prefix once and fully reusing it once costs 1.35x ... compared with 2x"
    assert F(5, 4) + F(1, 10) == F(135, 100)
    # M = 1,024, r = 0.1, w = 1.25 -> 102.4 + 1,177.6 / N ; N = 10이면 원래 앞부분이 221토큰 이상
    M, r, w = 1024, F(1, 10), F(5, 4)
    assert M * r == F(1024, 10) and M * (w - r) == F(11776, 10)
    cross = M * r + M * (w - r) / 10
    assert cross == F(22016, 100)
    assert min(L for L in range(1, 2000) if L * 10 > M * (w + 9 * r)) == 221


# ---------- 캐시 흉내 ----------

def blocks(q="a"):
    return [("rules", 3000), ("examples", 5000), ("inq", 24000), (f"q:{q}", 150)]


def test_cache_hit_needs_identical_prefix_and_refreshes_ttl():
    c = C.PrefixCache(ttl_minutes=5, min_tokens=512)
    u1 = c.request(blocks("a"), [1, 2], t=0)
    assert u1 == {"input_tokens": 150, "cache_creation_input_tokens": 32000, "cache_read_input_tokens": 0}
    u2 = c.request(blocks("b"), [1, 2], t=4)
    assert u2 == {"input_tokens": 150, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 32000}
    # 4분에 읽었으니 만료가 9분으로 밀렸다. 8분에 와도 적중.
    assert c.request(blocks("c"), [1, 2], t=8)["cache_read_input_tokens"] == 32000
    # 8분 + 5분 = 13분이 지나 14분에 오면 다시 쓴다.
    assert c.request(blocks("d"), [1, 2], t=14)["cache_creation_input_tokens"] == 32000


def test_partial_prefix_reuse_when_bundle_changes():
    c = C.PrefixCache(ttl_minutes=5, min_tokens=512)
    c.request(blocks(), [1, 2], t=0)
    b = blocks()
    b[2] = ("inq-v2", 24000)          # 문의 묶음만 바뀜: 지시문+예시 8,000은 읽고 묶음 24,000은 새로 쓴다
    u = c.request(b, [1, 2], t=1)
    assert u["cache_read_input_tokens"] == 8000 and u["cache_creation_input_tokens"] == 24000


def test_below_minimum_is_not_cached():
    c = C.PrefixCache(ttl_minutes=5, min_tokens=512)
    for t in range(3):
        u = c.request([("short", 400), (f"q{t}", 150)], [0], t=t)
        assert u == {"input_tokens": 550, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def test_anthropic_doc_example_breakpoint_on_changing_block_never_reads():
    # 문서의 예: 바뀌는 블록(시각+사용자 메시지)에 캐시 지점을 두면 매번 쓰기만 하고 읽지 못한다.
    c = C.PrefixCache(ttl_minutes=5, min_tokens=512)
    for t in range(5):
        u = c.request([("s1", 2000), ("s2", 2000), (f"stamp+msg{t}", 100)], [2], t=t)
        assert u["cache_read_input_tokens"] == 0 and u["cache_creation_input_tokens"] == 4100
    # 캐시 지점을 마지막으로 변하지 않는 블록으로 옮기면 두 번째부터 읽는다.
    c = C.PrefixCache(ttl_minutes=5, min_tokens=512)
    reads = [c.request([("s1", 2000), ("s2", 2000), (f"stamp+msg{t}", 100)], [1], t=t)["cache_read_input_tokens"]
             for t in range(5)]
    assert reads == [0, 4000, 4000, 4000, 4000]


# ---------- 관통 프로젝트 ----------

def test_project_sizes():
    p = PJ.load_project()
    assert PJ.prefix_tokens(p) == 32000 and len(p["sections"]) == 12
    assert PJ.prefix_tokens(p) + p["blocks"][-1]["tokens"] == 32150


def test_failure_scenarios_with_sonnet_5_5():
    p = PJ.load_project()
    m = C.price_of("claude-sonnet-5-5")
    got = {s: PJ.run(p, s, m) for s in PJ.SCENARIOS}
    wr = {s: (r["writes"], r["reads"]) for s, r in got.items()}
    assert wr == {"good": (1, 11), "timestamp_front": (12, 0), "name_front": (3, 9), "question_first": (12, 0),
                  "breakpoint_on_question": (12, 0), "gap10_5m": (12, 0), "gap10_1h": (1, 11),
                  "short_prompt": (0, 0)}
    g = got["good"]
    assert g["input_cost_nocache"] == F(7716, 10000) and g["input_cost_cache"] == F(1188, 10000)
    assert g["output_cost"] == F(84, 1000)
    # 앞에 날짜를 넣거나 10분 간격으로 5분 캐시를 쓰면 캐싱 없음보다 비싸다
    for s in ("timestamp_front", "question_first", "breakpoint_on_question", "gap10_5m"):
        assert got[s]["input_cost_cache"] > got[s]["input_cost_nocache"]
    assert got["gap10_1h"]["input_cost_cache"] == F(1668, 10000)
    assert got["short_prompt"]["input_cost_cache"] == got["short_prompt"]["input_cost_nocache"]


def test_weekly_saving_is_smaller_once_output_is_counted():
    p = PJ.load_project()
    for m in C.load_prices()["models"]:
        r = PJ.run(p, "good", m)
        s_in = 1 - r["input_cost_cache"] / r["input_cost_nocache"]
        s_all = 1 - (r["input_cost_cache"] + r["output_cost"]) / (r["input_cost_nocache"] + r["output_cost"])
        assert 0 < s_all < s_in < 1


# ---------- usage 두 모양 ----------

def test_usage_shapes_give_same_cost_and_double_count_is_wrong():
    m = C.price_of("gpt-6.1-sol")
    a = {"input_tokens": 150, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 32000}
    o = C.to_openai_usage(a)
    assert o["input_tokens"] == 32150 and o["input_tokens_details"]["cached_tokens"] == 32000
    assert C.cost_anthropic_usage(a, m) == C.cost_openai_usage(o, m) == F(35, 10000)
    assert C.cost_openai_double_count(o, m) == F(675, 10000)
    # OpenAI 문서 예: 두 번째 요청 input 15000 = cached 12000 + write 3000
    o2 = {"input_tokens": 15000, "input_tokens_details": {"cached_tokens": 12000, "cache_write_tokens": 3000}}
    assert o2["input_tokens"] - 12000 - 3000 == 0


# ---------- 요청 메시지와 SDK 흉내 ----------

def test_request_example_structure():
    req = RQ.load_example()
    assert list(req) == ["model", "max_tokens", "system", "messages"]
    assert RQ.breakpoints_of(req) == [1, 2]
    RQ.validate(req)
    assert len(RQ.breakpoints_of(req)) <= C.load_prices()["rules"]["anthropic"]["max_breakpoints"]


def test_sdk_like_client_retries_and_headers():
    t = RQ.FakeTransport([429, 529])
    c = RQ.MiniClient(t, env={"ANTHROPIC_API_KEY": "sk-test"})
    r = c.create(**RQ.load_example())
    assert r["attempts"] == 3 and c.waits == [0.5, 1.0]
    h = t.calls[0]["headers"]
    assert h["anthropic-version"] == "2023-06-01" and h["x-api-key"] == "sk-test"
    with pytest.raises(RQ.APIError) as e:
        RQ.MiniClient(RQ.FakeTransport([500, 500, 500]), env={"ANTHROPIC_API_KEY": "k"}).create(**RQ.load_example())
    assert e.value.attempts == 3
    with pytest.raises(RQ.APIError) as e:
        RQ.MiniClient(RQ.FakeTransport([400]), env={"ANTHROPIC_API_KEY": "k"}).create(**RQ.load_example())
    assert e.value.attempts == 1
    with pytest.raises(ValueError):
        RQ.MiniClient(RQ.FakeTransport([]), env={}).create(**RQ.load_example())
    with pytest.raises(ValueError):
        RQ.MiniClient(RQ.FakeTransport([]), env={"ANTHROPIC_API_KEY": "k"}).create(model="x", messages=[])


# ---------- 결과 파일 ----------

def test_results_file_matches_recomputation():
    res = json.loads((HERE / "results" / "ch02.json").read_text(encoding="utf-8"))
    assert res["e0_hand"]["n_star"] == 2 and res["e8_crosscheck"]["mismatches"] == 0
    p = PJ.load_project()
    for row in res["e4_weekly"]["rows"]:
        r = PJ.run(p, "good", C.price_of(row["model"]))
        assert row["cache_input"] == round(float(r["input_cost_cache"]), 6)
        assert row["nocache_input"] == round(float(r["input_cost_nocache"]), 6)


def test_reader_log_example_rows():
    import csv
    m = C.price_of("claude-sonnet-5-5")
    with open(HERE / "checks" / "ch02.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            u = {k: int(row[k]) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
            cost = C.cost_anthropic_usage(u, m, row["TTL"], int(row["output_tokens"]))
            assert round(float(cost), 4) == float(row["계산한 금액(달러)"])


def test_exercises_in_chapter():
    # 연습 1: 1시간 보관(w = 2), 앞부분 10,000 + 질문 100, 토큰 1개 1원
    assert 2 * 10_000 + 100 + 1_100 == 21_200 and 21_200 + 1_100 == 22_300
    # 연습 3: 이름 블록을 질문 바로 앞(캐시 지점 뒤)으로 옮기면 1번 쓰기, 11번 읽기
    p = PJ.load_project()
    c = C.PrefixCache(5, 512)
    us = []
    for i, s in enumerate(p["sections"]):
        b = PJ.base_blocks(p, s)
        who = p["requesters"][i % 3]
        b = b[:3] + [(f"name:{who}", 20)] + b[3:]
        us.append(c.request(b, [1, 2], t=i))
    assert sum(u["cache_creation_input_tokens"] > 0 for u in us) == 1
    assert sum(u["cache_read_input_tokens"] > 0 for u in us) == 11
    assert all(u["input_tokens"] == 170 for u in us)
