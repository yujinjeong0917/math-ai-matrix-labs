"""AI 도구 실전 3장 실험. `uv run python tools/ch03_tools_mcp/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 앱 3개 × 도구 4개, 짝마다 잇기 N×M과 표준 N+M
E1 도구 호출 JSON 읽기: data/tool_call_example.json의 칸
E2 현재 세대(2026-07-28) 왕복: server/discover → tools/list → tools/call, 진짜 자식 프로세스로
E3 옛 세대(2025-11-25) 왕복: initialize → notifications/initialized → tools/list → tools/call,
   그리고 옛 서버(--legacy-only)를 만난 클라이언트의 되돌아가기
E4 오류: _meta 빠짐, 모르는 버전, 모르는 도구, 잘못된 인자, 모르는 메서드, 깨진 JSON, stdout·stderr, 종료
E5 설정 JSON: Claude Desktop 예와 Claude Code(.mcp.json) 예를 읽고, 그 설정으로 서버를 띄워 왕복
E6 연결 수 표와 바뀜이 번지는 범위
E7 관통 프로젝트: 보고서 절 12개를 도구 호출 고리 + MCP로 쓰기, 2장(붙여 넣기)과 요청 수·토큰·금액 비교
E8 대조: 연결 수 공식과 하나씩 세기, 도구가 센 건수와 파일을 직접 센 건수

표준 라이브러리만 쓴다. 네트워크·API 키 없이 돈다. 서버는 sys.executable로 띄운다.
"""

import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch03_client as CL  # noqa: E402
import tools_ch03_config as CF  # noqa: E402
import tools_ch03_count as CN  # noqa: E402
import tools_ch03_data as D  # noqa: E402
import tools_ch03_loop as LP  # noqa: E402

OUT = HERE / "results" / "ch03.json"
SERVER = str(HERE / "tools_ch03_server.py")
LABS = HERE.parent.parent           # 실습 저장소 맨 위


def f(x, nd=6):
    return round(float(x), nd)


def server(*extra):
    return CL.StdioClient([sys.executable, SERVER, *extra])


def e0_hand():
    a = LP.load_json(LP.ASSUME_FILE)
    n, m = len(a["apps"]), len(a["tools"])
    return {"apps": a["apps"], "tools": a["tools"], "n": n, "m": m,
            "pairwise": CN.pairwise(n, m), "standard": CN.standard(n, m), "gap": CN.gap(n, m),
            "add_two_tools": {"m": m + 2, "pairwise": CN.pairwise(n, m + 2), "standard": CN.standard(n, m + 2)}}


def e1_tool_call():
    ex = LP.load_json(HERE / "data" / "tool_call_example.json")
    r1, resp1, m2, resp2 = ex["request_1"], ex["response_1"], ex["request_2_new_message"], ex["response_2"]
    tu = resp1["content"][0]
    tr = m2["content"][0]
    return {"request_keys": list(r1.keys()), "tool_keys": list(r1["tools"][0].keys()),
            "response_stop_reason": resp1["stop_reason"], "tool_use": tu,
            "tool_result_matches_id": tr["tool_use_id"] == tu["id"],
            "final_stop_reason": resp2["stop_reason"]}


def e2_modern():
    with server() as c:
        era, ver, info = c.connect()
        tools = c.list_tools()
        listed = c.request("tools/list")
        r = c.call_tool("count_by_category", {"week": D.WEEK})
        rc = c.close()
        lines = c.transcript
    return {"era": era, "version": ver, "server_info": info,
            "methods": [json.loads(l)["method"] for d, l in lines if d == "→"],
            "tool_names": [t["name"] for t in tools],
            "list_result_type": listed["resultType"], "ttlMs": listed["ttlMs"], "cacheScope": listed["cacheScope"],
            "counts": r["structuredContent"]["counts"], "total": r["structuredContent"]["total"],
            "is_error": r["isError"], "returncode": rc,
            "first_line_sent": lines[0][1], "lines_sent": sum(1 for d, _ in lines if d == "→"),
            "lines_received": sum(1 for d, _ in lines if d == "←")}


def e3_legacy():
    # (가) 옛 세대 클라이언트가 두 세대를 다 받는 서버에 initialize로 붙는다
    with server() as c:
        c.era = "legacy"
        init = c.request("initialize", {"protocolVersion": CL.LEGACY, "capabilities": {},
                                        "clientInfo": {"name": "old-client", "version": "1"}}, meta=False)
        c.notify("notifications/initialized")
        tools = c.list_tools()
        r = c.call_tool("list_inquiries", {"week": D.WEEK, "category": "환불"})
        c.close()
        methods_a = [json.loads(l)["method"] for d, l in c.transcript if d == "→"]
        has_result_type = "resultType" in json.loads(c.transcript[-1][1])["result"]
    # (나) 두 세대 클라이언트가 옛 서버를 만나 되돌아간다
    with server("--legacy-only") as c2:
        era, ver, _ = c2.connect()
        n = c2.call_tool("list_inquiries", {"week": D.WEEK, "category": "환불"})["structuredContent"]["count"]
        c2.close()
        probe = json.loads(c2.transcript[1][1])
        methods_b = [json.loads(l)["method"] for d, l in c2.transcript if d == "→"]
    return {"legacy_client": {"version": init["protocolVersion"], "methods": methods_a,
                              "tool_names": [t["name"] for t in tools],
                              "refund_count": r["structuredContent"]["count"],
                              "raw_result_has_resultType": has_result_type},
            "fallback": {"probe_error": probe["error"], "era": era, "version": ver, "methods": methods_b,
                         "refund_count": n}}


def e4_errors():
    out = {}
    with server() as c:
        c.era = "modern"
        meta = c.meta()

        def raw(msg):
            c.send(msg)
            return c._read(5)
        out["missing_meta"] = raw({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["error"]
        bad = dict(meta)
        bad.pop("io.modelcontextprotocol/clientCapabilities")
        out["missing_caps"] = raw({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": bad}})["error"]
        old = c.meta("2099-01-01")
        out["unsupported_version"] = raw({"jsonrpc": "2.0", "id": 3, "method": "tools/list",
                                          "params": {"_meta": old}})["error"]
        out["unknown_tool"] = raw({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                   "params": {"name": "delete_all", "arguments": {}, "_meta": meta}})["error"]
        r = raw({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "list_inquiries", "arguments": {"week": D.WEEK, "category": "환불요청"},
                            "_meta": meta}})["result"]
        out["bad_argument"] = {"isError": r["isError"], "text": r["content"][0]["text"],
                               "resultType": r["resultType"]}
        r = raw({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                 "params": {"name": "get_inquiry", "arguments": {"id": "#9999"}, "_meta": meta}})["result"]
        out["not_found"] = {"isError": r["isError"], "text": r["content"][0]["text"]}
        out["unknown_method"] = raw({"jsonrpc": "2.0", "id": 7, "method": "sampling/createMessage",
                                     "params": {"_meta": meta}})["error"]
        c.proc.stdin.write("{이건 JSON이 아니에요\n")
        c.proc.stdin.flush()
        out["parse_error"] = c._read(5)
        c.notify("notifications/cancelled", {"requestId": 99})   # 알림에는 답이 없다
        out["initialize_modern_version"] = raw({"jsonrpc": "2.0", "id": 8, "method": "initialize",
                                                "params": {"protocolVersion": CL.MODERN}})["error"]
        rc = c.close()
        out["stdout_all_json_one_line"] = all(l.startswith("{") and json.loads(l) for l in c.stdout_lines)
        out["stdout_lines"] = len(c.stdout_lines)
        out["stderr_lines"] = c.stderr_lines
        out["returncode_after_stdin_closed"] = rc
    return out


def e5_config():
    desk = HERE / "data" / "claude_desktop_config.example.json"
    code = HERE / "data" / "mcp.example.json"
    d = CF.read_config(desk, "desktop")
    k_none = CF.read_config(code, "code", env={})
    k_env = CF.read_config(code, "code", env={"CLAUDE_PROJECT_DIR": "/Users/you/math-ai-matrix-labs"})
    broken = CF.read_config({"mcpServers": {"x": {"url": "https://example.com/mcp"}}}, "code")
    # 설정대로 띄워 보기: 명령만 지금 파이썬으로, 자리표시 경로는 이 저장소 경로로 바꾼다
    spec = d["inquiries"]
    argv = [sys.executable] + [a.replace("/Users/you/math-ai-matrix-labs", str(LABS)) for a in spec["argv"][1:]]
    env = dict(os.environ, **{k: v.replace("/Users/you/math-ai-matrix-labs", str(LABS)) for k, v in spec["env"].items()})
    with CL.StdioClient(argv, env=env) as c:
        era, _, _ = c.connect()
        n = len(c.list_tools())
        c.close()
    spec2 = k_none["inquiries"]
    with CL.StdioClient([sys.executable] + spec2["argv"][1:], env=dict(os.environ, **spec2["env"]), cwd=LABS) as c:
        era2, _, _ = c.connect()
        n2 = c.call_tool("count_by_category", {"week": D.WEEK})["structuredContent"]["total"]
        c.close()
    return {"desktop": d, "code_no_env": k_none, "code_with_env": k_env, "url_without_type": broken,
            "desktop_lines": len(CF.line_by_line(desk)), "code_lines": len(CF.line_by_line(code)),
            "launched_from_desktop_config": {"era": era, "tools": n},
            "launched_from_code_config": {"era": era2, "total": n2, "cwd": "실습 저장소 맨 위"}}


def e6_connections():
    t = CN.table()
    return {"table": t, "ripple_3x4": CN.ripple(3, 4),
            "ties": [[r["n"], r["m"]] for r in t if r["gap"] == 0],
            "standard_more": [[r["n"], r["m"]] for r in t if r["gap"] < 0],
            "n3_m6": {"pairwise": CN.pairwise(3, 6), "standard": CN.standard(3, 6)},
            "examples": [{"n": n, "m": m, "pairwise": CN.pairwise(n, m), "standard": CN.standard(n, m),
                          "gap": CN.gap(n, m)} for n, m in [(1, 4), (2, 2), (3, 4), (3, 6), (5, 10), (10, 20)]]}


def e7_project():
    a = LP.load_json(LP.ASSUME_FILE)
    price = LP.load_json(LP.PRICE_FILE)
    logs = []
    with server() as c:
        c.connect()
        api_tools = LP.mcp_tools_to_api(c.list_tools())
        model = LP.ScriptedModel(wrong_first=["환불"])
        for s in a["sections"]:
            logs.append(LP.run_section(c, model, api_tools, s))
        c.close()
    rows = []
    for lg in logs:
        tools = [s for s in lg["steps"] if s["kind"] == "tool"]
        rows.append({"section": lg["section"], "requests": lg["requests"],
                     "tool_calls": [f'{t["name"]}({t["input"].get("category", "")})' for t in tools],
                     "errors": sum(t["is_error"] for t in tools),
                     "items_read": sum(t["items"] for t in tools),
                     "tokens_in": [i for i, _ in LP.section_requests(lg, a)]})
    tool_nc = LP.week_cost(logs, a, price, cache=False)
    tool_c = LP.week_cost(logs, a, price, cache=True)
    cp_nc = LP.copy_paste_cost(a, price, len(a["sections"]), cache=False)
    cp_c = LP.copy_paste_cost(a, price, len(a["sections"]), cache=True)

    def pack(x):
        return {k: (f(v) if not isinstance(v, int) else v) for k, v in x.items()}
    return {"rows": rows, "items_read_total": sum(r["items_read"] for r in rows),
            "tool_requests": tool_nc["requests"], "copy_paste_requests": cp_nc["requests"],
            "tool_nocache": pack(tool_nc), "tool_cache": pack(tool_c),
            "copy_paste_nocache": pack(cp_nc), "copy_paste_cache": pack(cp_c),
            "fixed_prefix_tool": a["rules"] + a["examples"] + a["tool_definitions"] + a["tool_system_prompt"],
            "refund_messages": [m["content"] if isinstance(m["content"], str) else m["content"]
                                for m in logs[1]["messages"]],
            "model_calls": model.calls}


def e8_crosscheck():
    bad = 0
    for n in range(1, 11):
        for m in range(1, 11):
            apps, tools = [f"a{i}" for i in range(n)], [f"t{j}" for j in range(m)]
            if CN.pairwise_by_listing(apps, tools) != CN.pairwise(n, m):
                bad += 1
            if CN.standard_by_listing(apps, tools) != CN.standard(n, m):
                bad += 1
            if CN.pairwise(n, m) - CN.standard(n, m) != CN.gap(n, m):
                bad += 1
    data = D.load()
    direct = Counter(i["category"] for i in data["items"])
    with server() as c:
        c.connect()
        via = c.call_tool("count_by_category", {"week": D.WEEK})["structuredContent"]["counts"]
        via_list = {cat: c.call_tool("list_inquiries", {"week": D.WEEK, "category": cat})["structuredContent"]["count"]
                    for cat in data["categories"]}
        c.close()
    return {"pairs_checked": 100, "mismatches": bad, "direct": dict(direct), "via_count_tool": via,
            "via_list_tool": via_list, "agree": dict(direct) == via == via_list,
            "data_file_regenerates": D.build() == data}


def main():
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_current": CL.MODERN, "spec_legacy": CL.LEGACY, "checked": "2026-10-10",
                 "price_file": LP.PRICE_FILE.name, "data_file": D.DATA_FILE.name, "seed": D.SEED},
        "e0_hand": e0_hand(),
        "e1_tool_call": e1_tool_call(),
        "e2_modern": e2_modern(),
        "e3_legacy": e3_legacy(),
        "e4_errors": e4_errors(),
        "e5_config": e5_config(),
        "e6_connections": e6_connections(),
        "e7_project": e7_project(),
        "e8_crosscheck": e8_crosscheck(),
    }
    # 기계마다 다른 값(경로)은 결과 파일에 남기지 않는다
    for k in ("stderr_lines",):
        res["e4_errors"][k] = [l.replace(str(HERE), "<ch03>") for l in res["e4_errors"][k]]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
