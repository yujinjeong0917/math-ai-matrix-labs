"""AI 도구 실전 5장: "이번 주 문의 요약 -> 화면 문구 점검" 왕복과 프롬프트 주입 장난감.

모델을 부르지 않는다. 모델이 할 판단(어떤 도구를 어떤 인자로 부를지)을 고정 규칙으로 대신해,
같은 입력이면 같은 도구 호출과 같은 결과가 나오게 했다.

왕복
1. query_inquiries로 이번 주 문의를 모두 읽는다(100행이 넘으면 next_cursor로 한 번 더).
2. 유형별로 센다(보고서 요약).
3. 본문에서 고객이 「」로 인용한 화면 문구를 모으고, 문의의 "화면" 칸(relation)이 가리키는 Figma 프레임을 찾는다.
4. get_screen_texts로 그 프레임의 글자 노드를 읽어 인용 문구가 있는지 본다.
5. 관련 화면에 없으면 모든 화면을 한 번 읽어 다른 화면에 있는지 본다.

주입 장난감
- 순진한 루프: 사용자 지시와 도구 결과를 구별하지 않고, 어디서든 지시처럼 생긴 문장을 찾으면 따른다.
- 구별하는 루프: 사용자 지시에서만 할 일을 고르고, 도구 결과 속 지시처럼 생긴 문장은 "사람 확인" 목록에 올린다.
지시를 알아보는 규칙(INSTR)은 이 장의 주입 예 한 가지만 알아보는 장난감이다. 실제 모델이 무엇을 따를지는 이 모형으로 알 수 없다.
"""

import json
import re
from collections import Counter

import tools_ch05_data as D
import tools_ch05_mcp as M

QUOTE_RE = re.compile(r"「([^」]+)」")
INSTR_SPAN = re.compile(r"\(이 글을 읽는 AI에게:([^)]*)\)")
INSTR_STATUS_ALL = re.compile(r"이번 주 문의를 모두 '([^']+)'(?:로|으로) 바꾸")
INSTR_SET_TEXT = re.compile(r"'([^']+)'을 '([^']+)'(?:로|으로) 고쳐")

USER_TASK = ("이번 주(2026-10-05~2026-10-11) 문의를 유형별로 세고, 고객이 인용한 화면 문구를 Figma에서 찾아 "
             "점검 목록을 만들어 줘. 아무것도 고치지 마.")


def payload(resp):
    r = resp["result"]
    if r.get("isError"):
        raise RuntimeError(r["content"][0]["text"])
    return r["structuredContent"]


def read_week(client, start=D.WEEK_START, end=D.WEEK_END):
    rows, cursor, calls = [], None, 0
    while True:
        args = {"start": start, "end": end}
        if cursor:
            args["cursor"] = cursor
        p = payload(client.call("query_inquiries", **args))
        calls += 1
        rows += p["rows"]
        cursor = p["next_cursor"]
        if not cursor:
            return rows, calls


def summarize(rows):
    c = Counter(r["properties"]["유형"] for r in rows)
    order = list(D.TYPE_COUNTS)
    items = sorted(c.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    n = len(rows)
    return [{"유형": t, "건수": k, "비율_pct": round(100 * k / n, 1)} for t, k in items]


def collect_quotes(server, rows):
    quotes = {}
    for r in rows:
        for ph in QUOTE_RE.findall(r["properties"]["본문"]):
            q = quotes.setdefault(ph, {"phrase": ph, "count": 0, "ids": [], "frames": []})
            q["count"] += 1
            q["ids"].append(r["id"])
            for s in r["properties"]["화면"]:
                f = server.screen_frame(s)
                if f and f not in q["frames"]:
                    q["frames"].append(f)
    return list(quotes.values())


def find(phrase, frames_payload):
    hits = []
    for f in frames_payload["frames"]:
        for t in f["texts"]:
            if phrase in t["characters"]:
                hits.append({"frame": f["id"], "frame_name": f["name"], "node_id": t["id"], "characters": t["characters"]})
    return hits


def roundtrip(server, client=None, strategy="per_screen"):
    """strategy: per_screen(관련 화면마다 한 번, 못 찾으면 전체 한 번) | all_once(전체를 한 번만)."""
    client = client or M.Client(server)
    rows, notion_calls = read_week(client)
    summary = summarize(rows)
    quotes = collect_quotes(server, rows)
    figma_calls = 0
    cache = {}
    if strategy == "all_once":
        cache[None] = payload(client.call("get_screen_texts"))
        figma_calls += 1
    checklist = []
    for q in quotes:
        status, hits = None, []
        for fid in q["frames"]:
            if strategy == "all_once":
                fp = {"frames": [f for f in cache[None]["frames"] if f["id"] == fid]}
            else:
                if fid not in cache:
                    cache[fid] = payload(client.call("get_screen_texts", frame_id=fid))
                    figma_calls += 1
                fp = cache[fid]
            hits += find(q["phrase"], fp)
        if hits:
            status = "관련 화면에 있음"
        else:
            if None not in cache:
                cache[None] = payload(client.call("get_screen_texts"))
                figma_calls += 1
            hits = find(q["phrase"], cache[None])
            status = "다른 화면에만 있음" if hits else "Figma에 없음"
        checklist.append({"phrase": q["phrase"], "count": q["count"],
                          "related_frames": q["frames"], "status": status, "found": hits})
    checklist.sort(key=lambda x: -x["count"])
    return {"rows": len(rows), "notion_calls": notion_calls, "figma_calls": figma_calls,
            "summary": summary, "checklist": checklist, "row_ids": [r["id"] for r in rows]}


# ---------------- 프롬프트 주입 장난감 ----------------

def parse_instructions(text):
    out = []
    for m in INSTR_STATUS_ALL.finditer(text):
        out.append(("status_all", m.group(1)))
    for m in INSTR_SET_TEXT.finditer(text):
        out.append(("set_text", m.group(1), m.group(2)))
    return out


class RecordingClient(M.Client):
    """도구 결과의 글(content[0].text)을 모아 두는 클라이언트. 모델의 문맥에 들어가는 글이라고 본다."""

    def __init__(self, server):
        super().__init__(server)
        self.context = []

    def call(self, name, **arguments):
        resp = super().call(name, **arguments)
        self.context.append(resp["result"]["content"][0]["text"])
        return resp


def run_agent(allow_write, mode):
    """mode: naive | separate. 반환: 왕복 결과 + 쓰기·시도·사람 확인 목록."""
    server = M.MiniServer(allow_write=allow_write)
    client = RecordingClient(server)
    rt = roundtrip(server, client)
    # 할 일 고르기
    if mode == "naive":
        sources = [USER_TASK] + client.context
    else:
        sources = [USER_TASK]
    todo = []
    for s in sources:
        todo += parse_instructions(s)
    # 도구 결과 속 지시처럼 생긴 문장 표시(구별하는 루프만)
    flagged = []
    if mode == "separate":
        seen = set()
        for r in server.rows:
            for m in INSTR_SPAN.finditer(r["properties"]["본문"]):
                if r["id"] in rt["row_ids"] and r["id"] not in seen:
                    seen.add(r["id"])
                    flagged.append({"id": r["id"], "text": m.group(0)})
    tool_names = {t["name"] for t in server.tools()}
    attempted, blocked = 0, 0
    for ins in dict.fromkeys(todo):           # 같은 지시가 여러 결과에 보여도 한 번만
        if ins[0] == "status_all":
            for rid in rt["row_ids"]:
                attempted += 1
                if "update_inquiry_status" in tool_names:
                    client.call("update_inquiry_status", id=rid, status=ins[1])
                else:
                    blocked += 1
        elif ins[0] == "set_text":
            attempted += 1
            if "set_text" in tool_names:
                allp = payload(client.call("get_screen_texts"))
                node = next(t["id"] for f in allp["frames"] for t in f["texts"] if t["characters"] == ins[1])
                client.call("set_text", node_id=node, characters=ins[2])
            else:
                blocked += 1
    notion_w = [w for w in server.writes if w[0] == "notion"]
    figma_w = [w for w in server.writes if w[0] == "figma"]
    return {"allow_write": allow_write, "mode": mode,
            "instructions_followed": [list(i) for i in dict.fromkeys(todo)],
            "attempted_calls": attempted, "blocked_no_tool": blocked,
            "notion_writes": len(notion_w), "figma_writes": len(figma_w),
            "figma_changes": [{"node_id": w[1], "before": w[2], "after": w[3]} for w in figma_w],
            "flagged_for_human": flagged, "tool_calls_total": len(server.log)}


def sizes():
    """같은 글자 정보를 노드 트리 전체로 줄 때와 글자 노드만 골라 줄 때의 글자 수."""
    server = M.MiniServer()
    full = json.dumps(server.frames, ensure_ascii=False)
    texts = json.dumps(payload(M.Client(server).call("get_screen_texts")), ensure_ascii=False)
    n_text_nodes = sum(1 for f in server.frames for c in f["children"] if c["type"] == "TEXT")
    n_nodes = sum(1 + len(f["children"]) for f in server.frames)
    return {"full_tree_chars": len(full), "text_only_chars": len(texts),
            "ratio": round(len(texts) / len(full), 4), "text_nodes": n_text_nodes, "all_nodes": n_nodes}
