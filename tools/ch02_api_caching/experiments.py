"""AI 도구 실전 2장 실험. `uv run python tools/ch02_api_caching/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 앞부분 10,000토큰 + 질문 100토큰, 토큰 1개 = 1원이라 치고 n번 보낼 때
E1 요청 메시지 읽기: data/request_example.json의 칸과 캐시 지점, 토큰 가정값의 합
E2 손익분기 호출 수 n*: 가격표의 모델마다(5분·1시간 쓰기)
E3 문서가 스스로 적은 숫자 다시 계산(Anthropic 1번/2번 읽기, OpenAI 1.35배, OpenAI 채우기 경계 221토큰)
E4 관통 프로젝트 한 주(절 12개)의 비용: 모델별, 캐싱 없음 / 캐싱
E5 캐시가 깨지는 흔한 실패 8가지(Claude Sonnet 5.5 가격)
E6 usage 읽기: Anthropic 모양과 OpenAI 모양, 두 번 세는 실수
E7 SDK 흉내: 재시도와 필수 칸 확인
E8 대조: 무작위 500경우에서 공식(닫힌 식) / 시뮬레이터 / 하나씩 세기

표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch02_cache as C  # noqa: E402
import tools_ch02_project as PJ  # noqa: E402
import tools_ch02_request as RQ  # noqa: E402

OUT = HERE / "results" / "ch02.json"
SEED = 2


def f(x, nd=6):
    """Fraction -> 보기 좋은 소수(문자열 아님)."""
    return round(float(x), nd)


def e0_hand():
    P, q, w, r = 10_000, 100, F(5, 4), F(1, 10)
    rows = []
    for n in range(1, 6):
        no = n * (P + q)
        ca = (w * P + q) + (n - 1) * (r * P + q)
        rows.append({"n": n, "nocache": int(no), "cache": int(ca), "cheaper": "cache" if ca < no else "nocache"})
    return {"prefix": P, "question": q, "first_call_cache": int(w * P + q), "later_call_cache": int(r * P + q),
            "per_call_nocache": P + q, "rows": rows, "n_star": C.breakeven_calls(w, r),
            "bound": f((w - r) / (1 - r), 4)}


def e1_request():
    req = RQ.load_example()
    blocks = RQ.walk_blocks(req)
    p = PJ.load_project()
    return {"top_level_keys": list(req.keys()), "model": req["model"], "max_tokens": req["max_tokens"],
            "blocks": [{"where": w, "role": r, "cache_control": c, "head": h} for w, r, c, h in blocks],
            "breakpoints": RQ.breakpoints_of(req),
            "assumed_tokens": {b["id"]: b["tokens"] for b in p["blocks"]},
            "prefix_tokens": PJ.prefix_tokens(p),
            "input_tokens_per_call": PJ.prefix_tokens(p) + p["blocks"][-1]["tokens"],
            "headers": RQ.MiniClient(RQ.FakeTransport([]), api_key="sk-test").headers() | {"x-api-key": "(키)"}}


def e2_breakeven(prices):
    rows = []
    for m in prices["models"]:
        for ttl in ("5m", "1h"):
            if ttl == "1h" and m["write_1h"] is None:
                continue
            w, r = C.multipliers(m, ttl)
            n = C.breakeven_calls(w, r)
            assert n == C.breakeven_bruteforce(w, r)
            ttl_label = ttl if m["provider"] != "OpenAI" else "30m"
            if m["provider"] == "Google":
                ttl_label = "implicit"
            rows.append({"model": m["model"], "provider": m["provider"], "ttl": ttl_label,
                         "w": f(w, 4), "r": f(r, 4), "bound": f((w - r) / (1 - r), 4), "n_star": n,
                         "ratio_n12": f(C.prefix_cost_ratio(12, w, r), 4),
                         "limit_saving": f(1 - r, 4)})
    return rows


def e3_doc_claims():
    # Anthropic 가격 문서: "caching pays off after one cache read for the 5-minute duration (1.25x write),
    # or after two cache reads for the 1-hour duration (2x write)"
    a5 = C.breakeven_calls(F(5, 4), F(1, 10)) - 1
    a1 = C.breakeven_calls(F(2), F(1, 10)) - 1
    # OpenAI 캐싱 문서: "writing a prefix once and fully reusing it once costs 1.35x ... compared with 2x"
    oa = F(5, 4) + F(1, 10)
    # OpenAI 캐싱 문서: M = 1,024, r = 0.1, w = 1.25 -> 102.4 + 1,177.6 / N, N = 10에서 221토큰
    M, r, w, N = 1024, F(1, 10), F(5, 4), 10
    cross = M * r + M * (w - r) / N
    min_len = int(cross) + 1 if cross != int(cross) else int(cross)
    return {"anthropic_reads_to_payoff_5m": a5, "anthropic_reads_to_payoff_1h": a1,
            "openai_one_write_one_read": f(oa, 4), "openai_no_cache_two": 2,
            "openai_pad_crossover_const": f(M * r, 4), "openai_pad_crossover_coef": f(M * (w - r), 4),
            "openai_pad_crossover_N10": f(cross, 4), "openai_pad_min_original_N10": min_len}


def e4_weekly(prices, p):
    rows = []
    for m in prices["models"]:
        res = PJ.run(p, "good", m)
        no_in, ca_in, out = res["input_cost_nocache"], res["input_cost_cache"], res["output_cost"]
        rows.append({"model": m["model"], "provider": m["provider"],
                     "writes": res["writes"], "reads": res["reads"],
                     "write_tokens": res["total"]["cache_creation_input_tokens"],
                     "read_tokens": res["total"]["cache_read_input_tokens"],
                     "plain_tokens": res["total"]["input_tokens"],
                     "nocache_input": f(no_in), "cache_input": f(ca_in), "output": f(out),
                     "nocache_total": f(no_in + out), "cache_total": f(ca_in + out),
                     "saving_input_pct": f(100 * (1 - ca_in / no_in), 1),
                     "saving_total_pct": f(100 * (1 - (ca_in + out) / (no_in + out)), 1),
                     "year_nocache_total": f(52 * (no_in + out), 2), "year_cache_total": f(52 * (ca_in + out), 2),
                     "assumption": "쓰기 할증·저장 요금 없음, 매번 적중 가정" if m["provider"] == "Google" else ""})
    return {"calls": len(p["sections"]), "prefix_tokens": PJ.prefix_tokens(p),
            "question_tokens": p["blocks"][-1]["tokens"], "output_tokens_per_call": p["output_tokens_per_call"],
            "rows": rows}


def e5_failures(prices, p):
    m = C.price_of("claude-sonnet-5-5", prices)
    rows = []
    for s in PJ.SCENARIOS:
        res = PJ.run(p, s, m)
        rows.append({"scenario": s, "ttl": res["ttl"], "writes": res["writes"], "reads": res["reads"],
                     "write_tokens": res["total"]["cache_creation_input_tokens"],
                     "read_tokens": res["total"]["cache_read_input_tokens"],
                     "plain_tokens": res["total"]["input_tokens"],
                     "input_cost_cache": f(res["input_cost_cache"]), "input_cost_nocache": f(res["input_cost_nocache"]),
                     "ratio_vs_nocache": f(res["input_cost_cache"] / res["input_cost_nocache"], 4)})
    return {"model": m["model"], "rows": rows}


def e6_usage(prices, p):
    m = C.price_of("gpt-6.1-sol", prices)
    us = C.run_calls(PJ.calls(p, "good"), m["ttl_minutes"][0], m["min_cache_tokens"])
    second = us[1]
    oa = C.to_openai_usage(second)
    right_a = C.cost_anthropic_usage(second, m)
    right_o = C.cost_openai_usage(oa, m)
    wrong = C.cost_openai_double_count(oa, m)
    return {"anthropic_shape": second, "openai_shape": oa, "cost_anthropic_rule": f(right_a),
            "cost_openai_rule": f(right_o), "cost_double_count": f(wrong),
            "double_count_ratio": f(wrong / right_o, 3)}


def e7_sdk():
    t1 = RQ.FakeTransport([429, 529], usage={"input_tokens": 150, "cache_creation_input_tokens": 0,
                                             "cache_read_input_tokens": 32000, "output_tokens": 700})
    c1 = RQ.MiniClient(t1, env={"ANTHROPIC_API_KEY": "sk-test"})
    req = RQ.load_example()
    r1 = c1.create(**req)
    t2 = RQ.FakeTransport([500, 500, 500])
    c2 = RQ.MiniClient(t2, env={"ANTHROPIC_API_KEY": "sk-test"})
    try:
        c2.create(**req)
        fail = None
    except RQ.APIError as e:
        fail = {"status": e.status, "attempts": e.attempts}
    t3 = RQ.FakeTransport([400])
    try:
        RQ.MiniClient(t3, env={"ANTHROPIC_API_KEY": "sk-test"}).create(**req)
        bad = None
    except RQ.APIError as e:
        bad = {"status": e.status, "attempts": e.attempts}
    try:
        RQ.MiniClient(RQ.FakeTransport([]), env={"ANTHROPIC_API_KEY": "k"}).create(model="x", messages=[])
        missing = None
    except ValueError as e:
        missing = str(e)
    return {"retry_then_ok": {"attempts": r1["attempts"], "waits": c1.waits},
            "retry_exhausted": fail, "not_retried_400": bad, "missing_field": missing,
            "sent_headers": sorted(t1.calls[0]["headers"].keys())}


def e8_crosscheck(n_cases=500):
    rng = random.Random(SEED)
    bad = 0
    for _ in range(n_cases):
        P = rng.randint(600, 60_000)
        q = rng.randint(10, 2_000)
        n = rng.randint(1, 30)
        w = F(rng.choice([100, 125, 200]), 100)
        r = F(rng.choice([25, 50, 100]), 1000)
        closed = (w * P + q) + (n - 1) * (r * P + q)
        cache = C.PrefixCache(ttl_minutes=5, min_tokens=512)
        sim = F(0)
        loop = F(0)
        for i in range(n):
            u = cache.request([("pre", P), (f"q{i}", q)], [0], t=i)
            sim += u["input_tokens"] + w * u["cache_creation_input_tokens"] + r * u["cache_read_input_tokens"]
            loop += (w * P if i == 0 else r * P) + q
        if not (closed == sim == loop):
            bad += 1
        if C.breakeven_calls(w, r) != C.breakeven_bruteforce(w, r):
            bad += 1
    return {"cases": n_cases, "mismatches": bad}


def main():
    prices = C.load_prices()
    p = PJ.load_project()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": SEED, "price_file": C.PRICE_FILE.name, "checked": prices["checked"]},
        "e0_hand": e0_hand(),
        "e1_request": e1_request(),
        "e2_breakeven": e2_breakeven(prices),
        "e3_doc_claims": e3_doc_claims(),
        "e4_weekly": e4_weekly(prices, p),
        "e5_failures": e5_failures(prices, p),
        "e6_usage": e6_usage(prices, p),
        "e7_sdk": e7_sdk(),
        "e8_crosscheck": e8_crosscheck(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
