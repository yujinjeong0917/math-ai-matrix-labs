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
    """서버 자식 프로세스를 띄운다. timeout 5초로 서버 멈춤 시 테스트가 끝없이 기다리지 않는다."""
    return CL.StdioClient([sys.executable, SERVER, *extra], timeout=5.0, **kw)


def raw(c, msg):
    """클라이언트 보정(_meta 자동 첨부, 오류 예외 변환) 없이 메시지를 보내고 응답 한 줄을 그대로 읽는다."""
    c.send(msg)
    return c._read(5)


# ---------- 연결 수 ----------

def test_first_screen_hand_numbers():
    """첫 화면 손계산: 앱 3 x 도구 4 = 12 대 3 + 4 = 7, 도구 6개면 18 대 9."""
    assert CN.pairwise(3, 4) == 12 and CN.standard(3, 4) == 7
    assert CN.pairwise(3, 6) == 18 and CN.standard(3, 6) == 9


@pytest.mark.parametrize("n", range(1, 11))
def test_gap_formula_and_listing(n):
    """공식 N x M, N + M, 차이 식을 목록으로 하나씩 센 값과 대조한다(n, m = 1~10 전부)."""
    for m in range(1, 11):
        apps, tools = list(range(n)), list(range(m))
        assert CN.pairwise_by_listing(apps, tools) == n * m
        assert CN.standard_by_listing(apps, tools) == n + m
        assert n * m - (n + m) == (n - 1) * (m - 1) - 1 == CN.gap(n, m)


def test_tie_only_at_two_by_two_and_standard_costs_more_when_one_side_is_one():
    """gap=0은 2x2에서만 나오고, gap<0(표준이 더 많음)은 한쪽이 1일 때뿐이다."""
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
    """저장된 JSON이 고정 시드로 다시 만든 것과 같고, 120건·유형별 건수가 맞다. 날짜는 10월 5~11일 안(문자열 비교)."""
    d = D.load()
    assert d == D.build()
    assert len(d["items"]) == 120 == sum(D.COUNTS.values())
    assert Counter(i["category"] for i in d["items"]) == Counter(D.COUNTS)
    assert all(i["received"].startswith("2026-10-") and "05" <= i["received"][8:10] <= "11" for i in d["items"])


# ---------- 현재 세대 왕복 ----------

def test_modern_roundtrip_over_real_stdio():
    """진짜 자식 프로세스와 현재 세대로 왕복한다. 보낸 요청 모두에 _meta 버전이 있고 initialize가 없어야 한다."""
    with spawn() as c:
        era, ver, info = c.connect()
        assert (era, ver) == ("modern", "2026-07-28") and info["name"] == "inquiries-mcp"
        listed = c.request("tools/list")
        assert listed["resultType"] == "complete"
        # 300,000ms = 5분(서버가 목록에 붙이는 캐시 힌트).
        assert listed["ttlMs"] == 300_000 and listed["cacheScope"] in ("public", "private")
        names = [t["name"] for t in listed["tools"]]
        assert names == ["count_by_category", "list_inquiries", "get_inquiry"]
        assert [t["name"] for t in c.list_tools()] == names            # 같은 순서
        r = c.call_tool("count_by_category", {"week": "2026-W41"})
        assert r["resultType"] == "complete" and r["isError"] is False
        assert r["structuredContent"]["total"] == 120
        assert json.loads(r["content"][0]["text"]) == r["structuredContent"]
        # stdin을 닫으면 서버가 스스로 종료하고 종료 코드 0이어야 한다.
        assert c.close() == 0
    sent = [json.loads(l) for d, l in c.transcript if d == "→"]
    assert all("io.modelcontextprotocol/protocolVersion" in m["params"]["_meta"] for m in sent)
    assert all("initialize" != m["method"] for m in sent)


def test_resources_and_prompts_minimal():
    """자료 목록·읽기, 프롬프트 틀 가져오기의 최소 동작."""
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
    """옛 세대 순서(initialize, notifications/initialized, 이후 요청)로 왕복한다. 환불 18건은 data의 COUNTS."""
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
    """옛 서버를 만나면 discover가 실패하고 initialize로 되돌아간다. 보낸 메서드 순서로 확인한다. #0001의 유형은 고정 시드 데이터에서 나온 값."""
    with spawn("--legacy-only") as c:
        era, ver, _ = c.connect()
        assert (era, ver) == ("legacy", "2025-11-25")
        assert [json.loads(l)["method"] for d, l in c.transcript if d == "→"][:3] == \
            ["server/discover", "initialize", "notifications/initialized"]
        assert c.call_tool("get_inquiry", {"id": "#0001"})["structuredContent"]["category"] == "환불"
        c.close()


def test_modern_error_does_not_fall_back():
    """현재 세대 오류 코드(-32022)는 옛 서버라는 뜻이 아니므로 되돌아가기 대상이 아니다. 오류 data에 서버가 아는 버전 목록이 담긴다."""
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
    """오류 코드를 하나씩 확인한다. -32602 잘못된 파라미터, -32601 없는 메서드, -32700 파싱 오류. 인자 오류는 error가 아니라 isError 결과."""
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
        # 현재 세대 경로에는 ping이 없어(옛 세대에서만 처리) 없는 메서드 오류가 난다.
        assert raw(c, {"jsonrpc": "2.0", "id": 6, "method": "ping", "params": {"_meta": meta}})["error"]["code"] == -32601
        c.proc.stdin.write("not json\n")
        c.proc.stdin.flush()
        assert c._read(5) == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        # 알림에는 응답이 없다. 바로 뒤 요청의 응답 id가 7이면 알림에 대한 응답이 끼지 않은 것.
        c.notify("notifications/cancelled", {"requestId": 1})
        assert raw(c, {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": meta}})["id"] == 7
        assert c.close() == 0
    # stdout 모든 줄이 JSON 하나씩이어야 한다(사양: stdout에는 메시지만).
    assert all(json.loads(l) for l in c.stdout_lines)
    assert all("\n" not in l for l in c.stdout_lines)
    assert any("stdin closed" in l for l in c.stderr_lines)


def test_timeout_instead_of_hanging():
    """답이 없을 때 TimeoutError가 나는지 확인한다(0.3초만 기다림)."""
    with spawn() as c:
        c.era = "modern"
        c.notify("notifications/cancelled", {"requestId": 1})   # 답이 없는 알림
        with pytest.raises(TimeoutError):
            c._read(0.3)
        c.close()


# ---------- 설정 JSON ----------

def test_config_files_read_and_launch():
    """설정 JSON 예시를 읽고, 그 설정대로 서버를 띄워 연결한다. 자리표시 경로는 이 저장소 경로로 치환한다."""
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
    """${VAR:-기본값} 규칙, 없는 변수는 경고와 원문 유지, url만 있고 type 없으면 건너뜀, 별칭, desktop은 치환 안 함."""
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
    """도구 호출 JSON 예시의 칸과 id 대응(tool_use.id = tool_result.tool_use_id)을 확인한다."""
    ex = LP.load_json(HERE / "data" / "tool_call_example.json")
    tu = ex["response_1"]["content"][0]
    assert ex["response_1"]["stop_reason"] == "tool_use" and tu["type"] == "tool_use"
    assert ex["request_2_new_message"]["content"][0]["tool_use_id"] == tu["id"]
    assert set(ex["request_1"]["tools"][0]) == {"name", "description", "input_schema"}


def run_week(wrong_first=("환불",)):
    """12개 절을 실제 서버로 돌린다. 기본은 환불 절에서 한 번 틀리는 시나리오."""
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
    # 요청 수: 배송 2, 환불 3(오류 1번 포함), 나머지 10개 절 2씩. 합 25.
    assert [lg["requests"] for lg in logs] == [2, 3] + [2] * 10
    items = sum(s["items"] for lg in logs for s in lg["steps"] if s["kind"] == "tool")
    # 12개 절이 읽은 문의 합: 유형 9개 절의 합이 120(건수 절 3개는 목록을 읽지 않음).
    assert items == 120
    nc = LP.week_cost(logs, a, price, cache=False)
    ca = LP.week_cost(logs, a, price, cache=True)
    # 토큰 기대값은 data/project_assumptions.json의 가정값으로 손계산한 값.
    assert nc["requests"] == 25 and nc["input_tokens"] == 245_980 and nc["output_tokens"] == 9_050
    assert nc["input_cost"] == F(245_980 * 2, 10**6)
    # 고정 앞부분 = 지시문 + 예시 + 도구 정의 + 도구 시스템 프롬프트(가정값).
    fixed = 3000 + 5000 + 400 + 286
    assert ca["input_cost"] == (F(fixed) * F(5, 2) + F(24 * fixed) * F(1, 10) + F(245_980 - 25 * fixed) * 2) / 10**6
    cp = LP.copy_paste_cost(a, price, 12, cache=True)
    # float 비교라 approx로 허용 오차를 둔다. 0.1188은 2장 캐싱 입력 비용(2장 테스트의 값).
    assert float(cp["input_cost"]) == pytest.approx(0.1188)       # 2장 결과와 같은 값
    assert LP.copy_paste_cost(a, price, 12, cache=False)["input_tokens"] == 385_800


def test_project_without_mistake_needs_24_requests():
    """틀리지 않으면 12개 절 x 2요청 = 24. 오류 한 번이 요청 1번을 더한다."""
    a, logs = run_week(wrong_first=())
    assert sum(lg["requests"] for lg in logs) == 24


def test_section_tokens_by_hand():
    """절 하나의 요청별 (입력, 출력) 토큰을 손으로 맞춘다. 8836 = 고정 8,686 + 질문 150, 두 번째 요청은 50(도구 호출 블록) + 28건 x 200토큰이 더해진다."""
    a, logs = run_week()
    assert LP.section_requests(logs[0], a) == [(8836, 50), (8836 + 50 + 28 * 200, 700)]
    assert LP.section_requests(logs[1], a)[1][0] == 8836 + 50 + 40


# ---------- 결과 파일 ----------

def test_results_file_matches_recomputation():
    """저장된 결과 파일의 핵심 숫자가 현재 코드와 같은지 본다(float 비교는 abs=1e-6 허용)."""
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
