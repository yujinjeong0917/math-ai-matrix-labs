"""AI 도구 실전 2장: 관통 프로젝트(주간 고객 문의 요약 보고서)의 호출 시나리오.

한 주 보고서는 절 12개를 따로 부른다. 앞부분(지시문 + 예시 + 문의 묶음)은 같고 마지막 요청만 바뀐다.
시나리오마다 같은 12번의 호출을 만들고, tools_ch02_cache.PrefixCache로 usage를 낸 뒤 금액으로 바꾼다.
"""

import json
from pathlib import Path

import tools_ch02_cache as C

HERE = Path(__file__).parent


def load_project(path=HERE / "data" / "weekly_report.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_blocks(p, section):
    out = []
    for b in p["blocks"]:
        key = b["id"] if b["stable"] else f'{b["id"]}:{section}'
        out.append((key, b["tokens"]))
    return out


def prefix_tokens(p):
    return sum(b["tokens"] for b in p["blocks"] if b["stable"])


def calls(p, scenario):
    """시나리오 이름 -> [(blocks, breakpoints, t분)] 12개."""
    secs = p["sections"]
    gap = p["gap_minutes_manual"] if scenario in ("gap10_5m", "gap10_1h") else p["gap_minutes_script"]
    inj = p["prefix_injection_tokens"]
    out = []
    for i, s in enumerate(secs):
        t = i * gap
        b = base_blocks(p, s)
        bps = [1, 2]                      # 예시 끝, 문의 묶음 끝 (request_example.json과 같은 자리)
        if scenario == "timestamp_front":
            b = [(f"now:2026-10-12 09:{t:02d}", inj)] + b
            bps = [2, 3]
        elif scenario == "name_front":
            who = p["requesters"][i % len(p["requesters"])]
            b = [(f"name:{who}", inj)] + b
            bps = [2, 3]
        elif scenario == "question_first":
            q = b.pop()
            b = [q] + b
            bps = [2, 3]
        elif scenario == "breakpoint_on_question":
            bps = [3]
        elif scenario == "short_prompt":
            b = [("rules-short", p["short_prompt_tokens"]), b[-1]]
            bps = [0]
        out.append((b, bps, t))
    return out


SCENARIOS = ["good", "timestamp_front", "name_front", "question_first", "breakpoint_on_question",
             "gap10_5m", "gap10_1h", "short_prompt"]


def ttl_for(scenario, m):
    if scenario == "gap10_1h":
        return 60, "1h"
    return m["ttl_minutes"][0] if m.get("ttl_minutes") else 5, "5m"


def run(p, scenario, m):
    ttl, kind = ttl_for(scenario, m)
    us = C.run_calls(calls(p, scenario), ttl, m["min_cache_tokens"])
    out_tok = p["output_tokens_per_call"] * len(us)
    tot = C.sum_usage(us)
    total_input = sum(sum(n for _, n in b) for b, _, _ in calls(p, scenario))
    cache_in = sum(C.cost_anthropic_usage(u, m, kind) for u in us)
    nocache_in = C.cost_nocache(total_input, m)
    out_cost = C.cost_nocache(0, m, out_tok)
    return {
        "usages": us, "total": tot, "total_input": total_input, "ttl": kind,
        "writes": sum(1 for u in us if u["cache_creation_input_tokens"] > 0),
        "reads": sum(1 for u in us if u["cache_read_input_tokens"] > 0),
        "input_cost_cache": cache_in, "input_cost_nocache": nocache_in, "output_cost": out_cost,
    }
