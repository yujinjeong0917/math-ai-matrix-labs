"""AI 도구 실전 3장: 도구 호출 고리(tool-use loop)와 관통 프로젝트 계산.

모델은 진짜 모델이 아니라 정해진 대본대로 답하는 가짜(ScriptedModel)다. 네트워크·API 키 없이 돈다.
고리의 모양은 Anthropic Messages API의 도구 호출과 같게 맞췄다.
  1) 요청에 tools(이름·설명·입력 형식)를 싣는다.
  2) 모델이 stop_reason "tool_use"와 tool_use 블록(id, name, input)을 돌려준다.
  3) 우리 프로그램이 그 도구를 실행한다(여기서는 MCP 서버에 tools/call).
  4) 결과를 tool_result 블록(tool_use_id, content, is_error)에 담아 다시 보낸다.
  5) 모델이 stop_reason "end_turn"으로 글을 마친다.
https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview (확인일 2026-10-10)
"""

import json
import zlib
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
ASSUME_FILE = HERE / "data" / "project_assumptions.json"
PRICE_FILE = HERE / "data" / "prices_2026-10-10.json"
WEEK = "2026-W41"
COUNT_SECTIONS = ["불만 상위 5", "이번 주 새로 생긴 유형", "다음 주 할 일"]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mcp_tools_to_api(tools):
    """MCP tools/list의 도구 정의를 Messages API의 tools 칸 모양으로 바꾼다(inputSchema → input_schema)."""
    return [{"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]} for t in tools]


class ScriptedModel:
    """보고서 절 이름을 보고 어떤 도구를 부를지 정해진 대로 고르는 가짜 모델."""

    def __init__(self, wrong_first=()):
        self.wrong_first = set(wrong_first)   # 처음에 없는 유형 이름을 대는 절(실패 흉내)
        self.calls = 0

    def create(self, tools, messages, section):
        self.calls += 1
        names = {t["name"] for t in tools}
        last = messages[-1]
        results = [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"] \
            if isinstance(last["content"], list) else []
        n_prev = sum(1 for m in messages if m["role"] == "assistant")
        tid = f"toolu_{zlib.crc32(section.encode()) % 10_000:04d}_{n_prev + 1}"   # 고정된 가짜 id
        if results and not results[-1].get("is_error"):
            data = json.loads(results[-1]["content"])
            n = data.get("count", data.get("total"))
            return {"role": "assistant", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": f"[{section}] 이번 주 {n}건을 읽고 절을 썼어요."}]}
        if section in COUNT_SECTIONS:
            assert "count_by_category" in names
            call = {"name": "count_by_category", "input": {"week": WEEK}}
        else:
            cat = section
            if section in self.wrong_first and not results:
                cat = section + "요청"              # 없는 유형 이름. 서버가 isError로 고칠 길을 알려 준다
            call = {"name": "list_inquiries", "input": {"week": WEEK, "category": cat}}
        return {"role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": tid, **call}]}


def run_section(client, model, api_tools, section):
    """보고서 절 하나를 도구 호출 고리로 쓴다. 기록(로그)을 돌려준다."""
    messages = [{"role": "user", "content": f"보고서 '{section}' 절을 써 줘. 필요하면 도구로 문의를 읽어."}]
    steps = []
    while True:
        resp = model.create(api_tools, messages, section)
        messages.append({"role": "assistant", "content": resp["content"]})
        if resp["stop_reason"] == "end_turn":
            steps.append({"kind": "final"})
            break
        results = []
        for block in resp["content"]:
            if block["type"] != "tool_use":
                continue
            r = client.call_tool(block["name"], block["input"])     # 실행은 프로그램이, MCP 서버에서
            text = r["content"][0]["text"]
            sc = r.get("structuredContent") or {}
            steps.append({"kind": "tool", "name": block["name"], "input": block["input"],
                          "is_error": r["isError"], "chars": len(text),
                          "items": len(sc.get("items", [])) if isinstance(sc, dict) else 0})
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": text,
                            "is_error": r["isError"]})
        messages.append({"role": "user", "content": results})
    return {"section": section, "requests": sum(1 for m in messages if m["role"] == "assistant"),
            "steps": steps, "messages": messages}


# ---------- 토큰·비용 계산(가정값은 data/project_assumptions.json) ----------

def result_tokens(step, a):
    if step["is_error"]:
        return a["error_result_tokens"]
    if step["name"] == "list_inquiries":
        return step["items"] * a["tokens_per_inquiry"]
    return a["count_result_tokens"]


def section_requests(log, a):
    """절 하나에서 보낸 요청마다 (입력 토큰, 출력 토큰). 대화가 쌓이며 다시 보내는 몫까지 센다."""
    fixed = a["rules"] + a["examples"] + a["tool_definitions"] + a["tool_system_prompt"]
    out, carried = [], 0
    tools = [s for s in log["steps"] if s["kind"] == "tool"]
    for i in range(log["requests"]):
        is_last = i == log["requests"] - 1
        inp = fixed + a["question"] + carried
        out.append((inp, a["output_final"] if is_last else a["tool_use_block"]))
        if not is_last:
            carried += a["tool_use_block"] + result_tokens(tools[i], a)
    return out


def week_cost(logs, a, price, cache):
    """한 주(절 12개)의 입력·출력 토큰과 금액(달러). cache=True면 고정 앞부분을 5분 캐시로 맡긴다고 본다."""
    fixed = a["rules"] + a["examples"] + a["tool_definitions"] + a["tool_system_prompt"]
    reqs = [r for lg in logs for r in section_requests(lg, a)]
    tin = sum(i for i, _ in reqs)
    tout = sum(o for _, o in reqs)
    M = 1_000_000
    if not cache:
        cin = F(tin) * F(str(price["input"])) / M
    else:
        plain = tin - fixed * len(reqs)
        cin = (F(fixed) * F(str(price["write_5m"])) + F(fixed * (len(reqs) - 1)) * F(str(price["read"]))
               + F(plain) * F(str(price["input"]))) / M
    cout = F(tout) * F(str(price["output"])) / M
    return {"requests": len(reqs), "input_tokens": tin, "output_tokens": tout,
            "input_cost": cin, "output_cost": cout, "total_cost": cin + cout}


def copy_paste_cost(a, price, n_sections, cache):
    """2장 방식(문의 120건을 붙여 넣은 앞부분 32,000토큰을 절마다 보냄)의 한 주 비용."""
    prefix = a["rules"] + a["examples"] + a["inquiry_bundle"]
    M = 1_000_000
    tin = n_sections * (prefix + a["question"])
    if not cache:
        cin = F(tin) * F(str(price["input"])) / M
    else:
        cin = (F(prefix) * F(str(price["write_5m"])) + F(prefix * (n_sections - 1)) * F(str(price["read"]))
               + F(n_sections * a["question"]) * F(str(price["input"]))) / M
    cout = F(n_sections * a["output_final"]) * F(str(price["output"])) / M
    return {"requests": n_sections, "input_tokens": tin, "output_tokens": n_sections * a["output_final"],
            "input_cost": cin, "output_cost": cout, "total_cost": cin + cout}
