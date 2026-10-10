"""AI 도구 실전 5장 검증. `uv run pytest -q tools/ch05_notion_figma` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
from pathlib import Path

import tools_ch05_data as D
import tools_ch05_mcp as M
import tools_ch05_roundtrip as R

HERE = Path(__file__).parent


# ---------- 사실 파일 ----------

def test_access_file_has_url_and_date_for_every_fact():
    acc = D.load_access()
    assert acc["checked"] == "2026-10-10"
    assert (HERE / "data" / "mcp_access_2026-10-10.json").exists()
    n = 0
    for part in ("mcp_spec", "notion", "figma"):
        for f in acc[part]["facts"]:
            assert f["url"].startswith("https://") and f["checked"] == "2026-10-10", f
            n += 1
    assert n >= 20
    assert acc["mcp_spec"]["version"] == "2026-07-28"


def test_rows_limit_matches_access_file():
    acc = D.load_access()
    rows = next(f for f in acc["notion"]["facts"] if f["key"] == "rows_limit")
    assert "100" in rows["value"] and M.MAX_ROWS == 100


# ---------- 데이터 ----------

def test_data_regenerates_identically():
    assert D.dump(D.build_inquiries()) == (D.DATA / "notion_inquiries.json").read_text(encoding="utf-8")
    assert D.dump(D.build_figma()) == (D.DATA / "figma_file.json").read_text(encoding="utf-8")


def test_data_shape():
    db = D.load_inquiries()["data_sources"]["고객 문의"]
    assert len(db["rows"]) == 120 + 18
    week = [r for r in db["rows"] if D.WEEK_START <= r["properties"]["접수일"][:10] <= D.WEEK_END]
    assert len(week) == 120 == sum(D.TYPE_COUNTS.values())
    inj = [r for r in db["rows"] if "AI에게" in r["properties"]["본문"]]
    assert [r["id"] for r in inj] == [D.INJECTION_ID]


# ---------- MCP 메시지 모양 ----------

def test_tools_list_and_call_shape():
    s = M.MiniServer()
    c = M.Client(s)
    tools = c.list_tools()["result"]["tools"]
    assert [t["name"] for t in tools] == ["query_inquiries", "get_screen_texts"]
    for t in tools:
        assert set(t) == {"name", "description", "inputSchema"} and t["inputSchema"]["type"] == "object"
    r = c.call("query_inquiries", start=D.WEEK_START, end=D.WEEK_END)["result"]
    assert r["isError"] is False and r["resultType"] == "complete"
    assert json.loads(r["content"][0]["text"]) == r["structuredContent"]
    assert len(r["structuredContent"]["rows"]) == 100 and r["structuredContent"]["next_cursor"] == "100"


def test_bad_input_is_tool_error_not_protocol_error():
    s = M.MiniServer()
    resp = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "query_inquiries", "arguments": {"start": "10/05", "end": D.WEEK_END}}})
    assert "error" not in resp and resp["result"]["isError"] is True
    resp = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "query_inquiries", "arguments": {"end": D.WEEK_END}}})
    assert resp["result"]["isError"] is True


def test_write_tool_absent_when_read_only():
    resp = M.MiniServer().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "set_text", "arguments": {"node_id": "1:302", "characters": "x"}}})
    assert resp["error"]["code"] == -32602


# ---------- 왕복 ----------

def test_roundtrip_numbers():
    r = R.roundtrip(M.MiniServer())
    assert r["rows"] == 120 and r["notion_calls"] == 2 and r["figma_calls"] == 4
    top = r["summary"][0]
    assert (top["유형"], top["건수"], top["비율_pct"]) == ("배송", 22, 18.3)
    assert sum(x["건수"] for x in r["summary"]) == 120
    got = {q["phrase"]: (q["count"], q["status"]) for q in r["checklist"]}
    assert got == {"3영업일 안에 처리돼요": (7, "관련 화면에 있음"),
                   "일시적인 오류가 발생했습니다": (6, "Figma에 없음"),
                   "30,000원 이상 무료 배송": (5, "다른 화면에만 있음"),
                   "쿠폰 적용": (4, "관련 화면에 있음")}
    elsewhere = next(q for q in r["checklist"] if q["phrase"] == "30,000원 이상 무료 배송")
    assert elsewhere["found"][0]["node_id"] == "1:602"


def test_all_once_strategy_same_checklist_fewer_calls():
    a = R.roundtrip(M.MiniServer())
    b = R.roundtrip(M.MiniServer(), strategy="all_once")
    assert b["figma_calls"] == 1 and a["checklist"] == b["checklist"]


def test_roundtrip_does_not_write():
    s = M.MiniServer(allow_write=True)
    R.roundtrip(s)
    assert s.writes == []


# ---------- 주입 장난감 ----------

def test_injection_scenarios():
    ro_naive = R.run_agent(False, "naive")
    assert ro_naive["notion_writes"] == ro_naive["figma_writes"] == 0 and ro_naive["blocked_no_tool"] == 121
    rw_naive = R.run_agent(True, "naive")
    assert rw_naive["notion_writes"] == 80 and rw_naive["figma_writes"] == 1
    assert rw_naive["figma_changes"][0] == {"node_id": "1:302", "before": "50,000원 이상 구매하면 무료 배송",
                                            "after": "전 상품 무료 배송"}
    for aw in (False, True):
        sep = R.run_agent(aw, "separate")
        assert sep["notion_writes"] == sep["figma_writes"] == sep["attempted_calls"] == 0
        assert [f["id"] for f in sep["flagged_for_human"]] == [D.INJECTION_ID]


def test_user_task_alone_has_no_instructions():
    assert R.parse_instructions(R.USER_TASK) == []


# ---------- 크기 ----------

def test_sizes():
    s = R.sizes()
    assert s["text_nodes"] == 20 and s["all_nodes"] == 32
    assert s["text_only_chars"] < s["full_tree_chars"] and 0.1 < s["ratio"] < 0.3
