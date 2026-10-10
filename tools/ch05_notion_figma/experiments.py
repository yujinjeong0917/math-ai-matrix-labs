"""AI 도구 실전 5장 실험. `uv run python tools/ch05_notion_figma/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 100행씩 읽으면 몇 번 부르나, 화면마다 부를 때와 한 번에 부를 때 Figma 호출 수
E1 도구 정의와 tools/call 요청·응답 한 줄씩 읽기용 예시, 입력이 틀렸을 때 isError
E2 왕복: 이번 주 문의 요약 -> 인용 문구 -> 화면 문구 점검 목록
E3 같은 글자 정보를 노드 트리 전체로 줄 때와 글자 노드만 줄 때의 글자 수
E4 프롬프트 주입 장난감: 읽기 전용 / 쓰기 도구 연결 x 순진한 루프 / 구별하는 루프
E5 Figma 호출 한도(View·Collab 좌석 한 달 20번)와 주간 보고서 호출 수
E6 결정성: 데이터 재생성과 왕복 두 번이 같은가

표준 라이브러리만 쓴다. 네트워크·API 키가 필요 없고 결과는 매번 같다.
"""

import json
import math
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch05_data as D  # noqa: E402
import tools_ch05_mcp as M  # noqa: E402
import tools_ch05_roundtrip as R  # noqa: E402

OUT = HERE / "results" / "ch05.json"


def e0_hand():
    rows = sum(D.TYPE_COUNTS.values())
    return {"week_rows": rows, "max_rows_per_call": M.MAX_ROWS, "notion_calls": math.ceil(rows / M.MAX_ROWS),
            "quoted_frames": 3, "figma_calls_per_screen": 3 + 1, "figma_calls_all_once": 1}


def e1_messages():
    s = M.MiniServer()
    c = M.Client(s)
    listed = c.list_tools()["result"]["tools"]
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
           "params": {"name": "query_inquiries",
                      "arguments": {"start": D.WEEK_START, "end": D.WEEK_END, "type": "환불"}}}
    resp = s.handle(req)
    sc = resp["result"]["structuredContent"]
    bad = s.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "query_inquiries", "arguments": {"start": "10/05", "end": D.WEEK_END}}})
    unknown = s.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "update_inquiry_status", "arguments": {"id": "inq-0057", "status": "답변 완료"}}})
    return {"tool_names_read_only": [t["name"] for t in listed],
            "tool_names_with_write": [t["name"] for t in M.MiniServer(allow_write=True).tools()],
            "query_tool_definition": listed[0],
            "example_request": req,
            "example_rows": len(sc["rows"]), "example_first_row": sc["rows"][0], "example_next_cursor": sc["next_cursor"],
            "bad_date": bad["result"], "write_tool_when_read_only": unknown}


def e2_roundtrip():
    r = R.roundtrip(M.MiniServer())
    r.pop("row_ids")
    r["quoted_inquiries"] = sum(q["count"] for q in r["checklist"])
    r["status_counts"] = {}
    for q in r["checklist"]:
        r["status_counts"][q["status"]] = r["status_counts"].get(q["status"], 0) + 1
    return r


def e4_injection():
    out = []
    for aw in (False, True):
        for mode in ("naive", "separate"):
            out.append(R.run_agent(aw, mode))
    week = M.MiniServer().rows
    week = [x for x in week if D.WEEK_START <= x["properties"]["접수일"][:10] <= D.WEEK_END]
    status = {}
    for x in week:
        status[x["properties"]["상태"]] = status.get(x["properties"]["상태"], 0) + 1
    return {"injection_row": D.INJECTION_ID, "week_status_before": status, "scenarios": out}


def e5_rate(e0):
    acc = D.load_access()
    limit = next(f["per_month"] for f in acc["figma"]["facts"] if f["key"] == "rate_view_collab")
    runs_per_month = F(52, 12)
    per = F(e0["figma_calls_per_screen"]) * runs_per_month
    once = F(e0["figma_calls_all_once"]) * runs_per_month
    return {"view_collab_limit_per_month": limit, "runs_per_month": round(float(runs_per_month), 4),
            "per_screen_calls_per_month": round(float(per), 2), "all_once_calls_per_month": round(float(once), 2),
            "per_screen_runs_until_limit": limit // e0["figma_calls_per_screen"],
            "all_once_runs_until_limit": limit // e0["figma_calls_all_once"],
            "per_screen_twice_a_week": round(float(2 * per), 2)}


def e6_determinism():
    same_data = (D.dump(D.build_inquiries()) == (D.DATA / "notion_inquiries.json").read_text(encoding="utf-8")
                 and D.dump(D.build_figma()) == (D.DATA / "figma_file.json").read_text(encoding="utf-8"))
    a = R.roundtrip(M.MiniServer())
    b = R.roundtrip(M.MiniServer())
    return {"data_regenerates_identically": same_data, "roundtrip_twice_identical": a == b}


def main():
    e0 = e0_hand()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": D.SEED, "access_file": "mcp_access_2026-10-10.json", "checked": D.load_access()["checked"],
                 "week": [D.WEEK_START, D.WEEK_END]},
        "e0_hand": e0,
        "e1_messages": e1_messages(),
        "e2_roundtrip": e2_roundtrip(),
        "e3_sizes": R.sizes(),
        "e4_injection": e4_injection(),
        "e5_rate": e5_rate(e0),
        "e6_determinism": e6_determinism(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
