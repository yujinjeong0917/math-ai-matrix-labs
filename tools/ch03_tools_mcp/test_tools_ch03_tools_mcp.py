"""AI 도구 실전 3장 검증. `uv run pytest -q tools/ch03_tools_mcp` 로 실행한다.

서버를 진짜 자식 프로세스로 띄워 stdio로 왕복한다(네트워크·API 키 없음).
experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 다시 계산한다.
"""

import json
import os
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch03_client as CL
import tools_ch03_config as CF
import tools_ch03_count as CN
import tools_ch03_data as D
import tools_ch03_loop as LP

HERE = Path(__file__).parent
SERVER = str(HERE / "tools_ch03_server.py")
LABS = HERE.parent.parent


def spawn(*extra, **kw):
    return CL.StdioClient([sys.executable, SERVER, *extra], timeout=5.0, **kw)


def raw(c, msg):
    c.send(msg)
    return c._read(5)


# ---------- 연결 수 ----------

def test_first_screen_hand_numbers():
    assert CN.pairwise(3, 4) == 12 and CN.standard(3, 4) == 7
    assert CN.pairwise(3, 6) == 18 and CN.standard(3, 6) == 9


@pytest.mark.parametrize("n", range(1, 11))
def test_gap_formula_and_listing(n):
    for m in range(1, 11):
        apps, tools = list(range(n)), list(range(m))
        assert CN.pairwise_by_listing(apps, tools) == n * m
        assert CN.standard_by_listing(apps, tools) == n + m
        assert n * m - (n + m) == (n - 1) * (m - 1) - 1 == CN.gap(n, m)


def test_tie_only_at_two_by_two_and_standard_costs_more_when_one_side_is_one():
    t = CN.table(6, 6)
    assert [(r["n"], r["m"]) for r in t if r["gap"] == 0] == [(2, 2)]
    assert all(r["n"] == 1 or r["m"] == 1 for r in t if r["gap"] < 0)
    assert CN.gap(1, 4) == -1          # 앱이 하나뿐이면 표준이 연결을 하나 더 만든다


def test_ripple():
    r = CN.ripple(3, 4)
    assert r["tool_changes"] == {"pairwise": 3, "standard": 1}
    assert r["app_changes"] == {"pairwise": 4, "standard": 1}


# ---------- 데이터 ----------

def test_data_file_is_reproducible_and_has_120():
    d = D.load()
    assert d == D.build()
    assert len(d["items"]) == 120 == sum(D.COUNTS.values())
    assert Counter(i["category"] for i in d["items"]) == Counter(D.COUNTS)
    assert all(i["received"].startswith("2026-10-") and "05" <= i["received"][8:10] <= "11" for i in d["items"])


# ---------- 현재 세대 왕복 ----------

def test_modern_roundtrip_over_real_stdio():
    with spawn() as c:
        era, ver, info = c.connect()
        assert (era, ver) == ("modern", "2026-07-28") and info["name"] == "inquiries-mcp"
        listed = c.request("tools/list")
        assert listed["resultType"] == "complete"
        assert listed["ttlMs"] == 300_000 and listed["cacheScope"] in ("public", "private")
        names = [t["name"] for t in listed["tools"]]
        assert names == ["count_by_category", "list_inquiries", "get_inquiry"]
        assert [t["name"] for t in c.list_tools()] == names            # 같은 순서
        r = c.call_tool("count_by_category", {"week": "2026-W41"})
        assert r["resultType"] == "complete" and r["isError"] is False
        assert r["structuredContent"]["total"] == 120
        assert json.loads(r["content"][0]["text"]) == r["structuredContent"]
        assert c.close() == 0
    sent = [json.loads(l) for d, l in c.transcript if d == "→"]
    assert all("io.modelcontextprotocol/protocolVersion" in m["params"]["_meta"] for m in sent)
    assert all("initialize" != m["method"] for m in sent)


def test_resources_and_prompts_minimal():
    with spawn() as c:
        c.connect()
        res = c.request("resources/list")["resources"]
        body = c.request("resources/read", {"uri": res[0]["uri"]})
        assert len(json.loads(body["contents"][0]["text"])) == 120
        p = c.request("prompts/get", {"name": "weekly_section", "arguments": {"category": "환불"}})
        assert "환불" in p["messages"][0]["content"]["text"]
        c.close()


# ---------- 옛 세대 왕복과 되돌아가기 ----------

def test_legacy_initialize_roundtrip():
    with spawn() as c:
        c.era = "legacy"
        init = c.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                        "clientInfo": {"name": "old", "version": "1"}}, meta=False)
        assert init["protocolVersion"] == "2025-11-25" and "tools" in init["capabilities"]
        c.notify("notifications/initialized")
        assert len(c.list_tools()) == 3
        r = c.call_tool("list_inquiries", {"week": "2026-W41", "category": "환불"})
        assert r["structuredContent"]["count"] == 18
        c.close()
    last = json.loads(c.transcript[-1][1])["result"]
    assert "resultType" not in last          # 옛 세대 결과에는 resultType이 없다(클라이언트가 complete로 읽음)


def test_dual_era_client_falls_back_on_legacy_only_server():
    with spawn("--legacy-only") as c:
        era, ver, _ = c.connect()
        assert (era, ver) == ("legacy", "2025-11-25")
        assert [json.loads(l)["method"] for d, l in c.transcript if d == "→"][:3] == \
            ["server/discover", "initialize", "notifications/initialized"]
        assert c.call_tool("get_inquiry", {"id": "#0001"})["structuredContent"]["category"] == "환불"
        c.close()


def test_modern_error_does_not_fall_back():
    with spawn() as c:
        c.era = None
        c.version = "2099-01-01"
        err = raw(c, {"jsonrpc": "2.0", "id": 1, "method": "server/discover",
                      "params": {"_meta": c.meta("2099-01-01")}})["error"]
        assert err["code"] == -32022 and err["data"] == {"supported": ["2026-07-28", "2025-11-25"],
                                                          "requested": "2099-01-01"}
        c.close()


# ---------- 오류와 stdio 규칙 ----------

def test_errors_and_stdout_purity():
    with spawn() as c:
        c.era = "modern"
        meta = c.meta()
        assert raw(c, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["error"]["code"] == -32602
        m2 = {k: v for k, v in meta.items() if k != "io.modelcontextprotocol/clientCapabilities"}
        assert raw(c, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {"_meta": m2}})["error"]["code"] == -32602
        e = raw(c, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}, "_meta": meta}})["error"]
        assert e["code"] == -32602 and "Unknown tool" in e["message"]
        r = raw(c, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "list_inquiries", "arguments": {"week": "2026-W41", "category": "환불요청"},
                               "_meta": meta}})["result"]
        assert r["isError"] is True and "환불" in r["content"][0]["text"]
        r = raw(c, {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "list_inquiries", "arguments": {"week": "2026-W40", "category": "환불"},
                               "_meta": meta}})["result"]
        assert r["isError"] is True
        assert raw(c, {"jsonrpc": "2.0", "id": 6, "method": "ping", "params": {"_meta": meta}})["error"]["code"] == -32601
        c.proc.stdin.write("not json\n")
        c.proc.stdin.flush()
        assert c._read(5) == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        c.notify("notifications/cancelled", {"requestId": 1})
        assert raw(c, {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": meta}})["id"] == 7
        assert c.close() == 0
    assert all(json.loads(l) for l in c.stdout_lines)
    assert all("\n" not in l for l in c.stdout_lines)
    assert any("stdin closed" in l for l in c.stderr_lines)


def test_timeout_instead_of_hanging():
    with spawn() as c:
        c.era = "modern"
        c.notify("notifications/cancelled", {"requestId": 1})   # 답이 없는 알림
        with pytest.raises(TimeoutError):
            c._read(0.3)
        c.close()


# ---------- 설정 JSON ----------

def test_config_files_read_and_launch():
    desk = CF.read_config(HERE / "data" / "claude_desktop_config.example.json", "desktop")
    spec = desk["inquiries"]
    assert spec["argv"][0] == "/usr/bin/python3" and spec["argv"][1].startswith("/")
    argv = [sys.executable] + [a.replace("/Users/you/math-ai-matrix-labs", str(LABS)) for a in spec["argv"][1:]]
    env = dict(os.environ, **{k: v.replace("/Users/you/math-ai-matrix-labs", str(LABS)) for k, v in spec["env"].items()})
    with CL.StdioClient(argv, env=env) as c:
        assert c.connect()[0] == "modern" and len(c.list_tools()) == 3
        c.close()
    code = CF.read_config(HERE / "data" / "mcp.example.json", "code", env={})
    assert code["inquiries"]["argv"][1] == "./tools/ch03_tools_mcp/tools_ch03_server.py"
    assert code["notion"] == {"type": "http", "url": "https://mcp.notion.com/mcp", "warnings": []}
    with CL.StdioClient([sys.executable] + code["inquiries"]["argv"][1:],
                        env=dict(os.environ, **code["inquiries"]["env"]), cwd=LABS) as c:
        c.connect()
        assert c.call_tool("count_by_category", {"week": "2026-W41"})["structuredContent"]["total"] == 120
        c.close()


def test_config_expansion_rules():
    w = []
    assert CF.expand("${A:-x}/y", {}, w) == "x/y" and w == []
    assert CF.expand("${A:-x}/y", {"A": "/p"}, w) == "/p/y"
    assert CF.expand("${B}", {}, w) == "${B}" and w == ["환경 변수 B가 없어요"]
    bad = CF.read_config({"mcpServers": {"x": {"url": "https://e.com/mcp"}}}, "code")
    assert "skipped" in bad["x"]
    alias = CF.read_config({"mcpServers": {"x": {"type": "streamable-http", "url": "https://e.com/mcp"}}}, "code")
    assert alias["x"]["type"] == "http"
    # Desktop 쪽은 변수를 바꾸지 않는다고 본다(문서에서 확인하지 못함)
    d = CF.read_config({"mcpServers": {"x": {"command": "${HOME}/a"}}}, "desktop")
    assert d["x"]["argv"] == ["${HOME}/a"]


# ---------- 도구 호출 고리와 관통 프로젝트 ----------

def test_tool_call_example_shape():
    ex = LP.load_json(HERE / "data" / "tool_call_example.json")
    tu = ex["response_1"]["content"][0]
    assert ex["response_1"]["stop_reason"] == "tool_use" and tu["type"] == "tool_use"
    assert ex["request_2_new_message"]["content"][0]["tool_use_id"] == tu["id"]
    assert set(ex["request_1"]["tools"][0]) == {"name", "description", "input_schema"}


def run_week(wrong_first=("환불",)):
    a = LP.load_json(LP.ASSUME_FILE)
    with spawn() as c:
        c.connect()
        api_tools = LP.mcp_tools_to_api(c.list_tools())
        logs = [LP.run_section(c, LP.ScriptedModel(wrong_first), api_tools, s) for s in a["sections"]]
        c.close()
    return a, logs


def test_project_week_numbers():
    a, logs = run_week()
    price = LP.load_json(LP.PRICE_FILE)
    assert [lg["requests"] for lg in logs] == [2, 3] + [2] * 10
    items = sum(s["items"] for lg in logs for s in lg["steps"] if s["kind"] == "tool")
    assert items == 120
    nc = LP.week_cost(logs, a, price, cache=False)
    ca = LP.week_cost(logs, a, price, cache=True)
    assert nc["requests"] == 25 and nc["input_tokens"] == 245_980 and nc["output_tokens"] == 9_050
    assert nc["input_cost"] == F(245_980 * 2, 10**6)
    fixed = 3000 + 5000 + 400 + 286
    assert ca["input_cost"] == (F(fixed) * F(5, 2) + F(24 * fixed) * F(1, 10) + F(245_980 - 25 * fixed) * 2) / 10**6
    cp = LP.copy_paste_cost(a, price, 12, cache=True)
    assert float(cp["input_cost"]) == pytest.approx(0.1188)       # 2장 결과와 같은 값
    assert LP.copy_paste_cost(a, price, 12, cache=False)["input_tokens"] == 385_800


def test_project_without_mistake_needs_24_requests():
    a, logs = run_week(wrong_first=())
    assert sum(lg["requests"] for lg in logs) == 24


def test_section_tokens_by_hand():
    a, logs = run_week()
    assert LP.section_requests(logs[0], a) == [(8836, 50), (8836 + 50 + 28 * 200, 700)]
    assert LP.section_requests(logs[1], a)[1][0] == 8836 + 50 + 40


# ---------- 결과 파일 ----------

def test_results_file_matches_recomputation():
    res = json.loads((HERE / "results" / "ch03.json").read_text(encoding="utf-8"))
    assert res["meta"]["spec_current"] == "2026-07-28" and res["meta"]["checked"] == "2026-10-10"
    assert res["e0_hand"]["pairwise"] == 12 and res["e0_hand"]["standard"] == 7
    assert res["e7_project"]["tool_nocache"]["input_tokens"] == 245_980
    assert res["e7_project"]["tool_cache"]["total_cost"] == pytest.approx(0.190721, abs=1e-6)
    assert res["e8_crosscheck"]["mismatches"] == 0 and res["e8_crosscheck"]["agree"] is True
    assert res["e4_errors"]["unsupported_version"]["code"] == -32022


def test_reader_log_example_rows():
    rows = (HERE / "checks" / "ch03.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("날짜,")
    assert len(rows) >= 3
