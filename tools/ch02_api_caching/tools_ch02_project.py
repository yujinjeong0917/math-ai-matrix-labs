"""AI 도구 실전 2장: 관통 프로젝트(주간 고객 문의 요약 보고서)의 호출 시나리오.

한 주 보고서는 절 12개를 따로 부른다. 앞부분(지시문 + 예시 + 문의 묶음)은 같고 마지막 요청만 바뀐다.
시나리오마다 같은 12번의 호출을 만들고, tools_ch02_cache.PrefixCache로 usage를 낸 뒤 금액으로 바꾼다.
"""

import json
from pathlib import Path

import tools_ch02_cache as C

HERE = Path(__file__).parent


def load_project(path=HERE / "data" / "weekly_report.json"):
    """관통 프로젝트 설정(블록별 토큰 수, 절 목록, 호출 간격 등) JSON을 읽는다."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_blocks(p, section):
    """한 절을 부르는 요청의 블록 목록 [(key, 토큰 수)].
    stable 블록은 절과 무관한 key, 아닌 블록은 절 이름을 붙인 key를 써서 '앞부분이 같은지'를 key 비교로 판정한다.
    """
    out = []
    for b in p["blocks"]:
        # key가 같아야만 캐시가 같은 블록으로 본다. 절마다 바뀌는 블록은 key에 절 이름을 넣어 달라지게 한다.
        key = b["id"] if b["stable"] else f'{b["id"]}:{section}'
        out.append((key, b["tokens"]))
    return out


def prefix_tokens(p):
    """절이 바뀌어도 변하지 않는 앞부분(stable 블록)의 토큰 합."""
    return sum(b["tokens"] for b in p["blocks"] if b["stable"])


def calls(p, scenario):
    """시나리오 이름 -> [(blocks, breakpoints, t분)] 12개."""
    secs = p["sections"]
    # gap10 시나리오는 사람이 손으로 부르는 간격(수동), 나머지는 스크립트 간격. 수치는 data JSON에서 온다.
    gap = p["gap_minutes_manual"] if scenario in ("gap10_5m", "gap10_1h") else p["gap_minutes_script"]
    inj = p["prefix_injection_tokens"]
    out = []
    for i, s in enumerate(secs):
        # t는 보고서 시작 후 경과 시간(분). 캐시 만료(TTL) 판정에 쓰인다.
        t = i * gap
        b = base_blocks(p, s)
        # 캐시 지점은 블록 번호. 아래 변형들은 앞에 블록이 끼거나 순서가 바뀌어 번호가 밀린다(off-by-one 함정).
        bps = [1, 2]                      # 예시 끝, 문의 묶음 끝 (request_example.json과 같은 자리)
        # 나쁜 습관 1: 매번 바뀌는 시각을 맨 앞에 넣으면 앞부분이 매번 달라져 캐시가 안 맞는다.
        if scenario == "timestamp_front":
            b = [(f"now:2026-10-12 09:{t:02d}", inj)] + b
            bps = [2, 3]
        # 나쁜 습관 2: 요청자 이름(호출마다 순환)을 맨 앞에 넣는 경우.
        elif scenario == "name_front":
            who = p["requesters"][i % len(p["requesters"])]
            b = [(f"name:{who}", inj)] + b
            bps = [2, 3]
        # 나쁜 습관 3: 절마다 다른 질문을 맨 앞으로 옮기면 뒤의 큰 블록이 모두 달라진 위치가 된다.
        elif scenario == "question_first":
            q = b.pop()
            b = [q] + b
            bps = [2, 3]
        # 캐시 지점을 매번 바뀌는 질문 블록에 두면 그 끝까지가 쓰기 대상이 되어 읽기가 안 생긴다.
        elif scenario == "breakpoint_on_question":
            bps = [3]
        # 최소 캐시 토큰 수에 못 미치는 짧은 프롬프트: 캐시 지점을 둬도 캐시되지 않는다.
        elif scenario == "short_prompt":
            b = [("rules-short", p["short_prompt_tokens"]), b[-1]]
            bps = [0]
        out.append((b, bps, t))
    return out


SCENARIOS = ["good", "timestamp_front", "name_front", "question_first", "breakpoint_on_question",
             "gap10_5m", "gap10_1h", "short_prompt"]


def ttl_for(scenario, m):
    """시나리오에 쓸 (유지 시간 분, 종류). gap10_1h만 1시간 TTL, 나머지는 모델의 기본(없으면 5분)."""
    if scenario == "gap10_1h":
        return 60, "1h"
    return m["ttl_minutes"][0] if m.get("ttl_minutes") else 5, "5m"


def run(p, scenario, m):
    """한 시나리오를 돌려 usage와 입력·출력 비용(달러)을 낸다. 출력 비용은 캐시와 무관해 따로 센다."""
    ttl, kind = ttl_for(scenario, m)
    us = C.run_calls(calls(p, scenario), ttl, m["min_cache_tokens"])
    out_tok = p["output_tokens_per_call"] * len(us)
    tot = C.sum_usage(us)
    # 캐싱이 없었다면 낸 입력 토큰 합(비교 기준).
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
