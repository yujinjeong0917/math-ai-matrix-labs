"""AI 도구 실전 5장: MCP 모양의 도구 서버 흉내 (프로세스 안에서만 돈다, 네트워크 없음).

MCP 명세(2026-07-28판)의 tools/list, tools/call 메시지 모양을 따른다.
- 도구 정의: name, description, inputSchema
- 결과: resultType "complete", content[{type: "text", text}], structuredContent, isError
- 입력이 틀리면 JSON-RPC 오류가 아니라 isError: true 결과로 돌려준다(도구 실행 오류)

읽기 도구 두 개
- query_inquiries : Notion "고객 문의" 데이터 소스에서 기간·유형으로 행을 읽는다. 한 번에 최대 100행, 넘으면 next_cursor.
  (Notion MCP notion-query-data-sources의 rows 모드가 "100-row maximum per query"라는 점을 흉내 냈다)
- get_screen_texts : Figma 파일에서 프레임(화면) 하나 또는 전체의 글자 노드만 골라 id, name, characters로 준다.

쓰기 도구 두 개 (기본은 꺼져 있음, 프롬프트 주입 실험에서만 켠다)
- update_inquiry_status : 문의 행의 상태 칸을 바꾼다
- set_text : 글자 노드의 문구를 바꾼다
"""

import copy
import json
import re

import tools_ch05_data as D

PROTOCOL = "2026-07-28"
MAX_ROWS = 100
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

READ_TOOLS = [
    {"name": "query_inquiries",
     "description": "고객 문의 데이터베이스에서 접수일이 기간 안에 있는 문의를 읽어요. 한 번에 최대 100행이고, 더 있으면 next_cursor를 줘요.",
     "inputSchema": {"type": "object",
                     "properties": {"start": {"type": "string", "description": "시작일 YYYY-MM-DD (포함)"},
                                    "end": {"type": "string", "description": "끝일 YYYY-MM-DD (포함)"},
                                    "type": {"type": "string", "description": "유형 하나만 고를 때"},
                                    "cursor": {"type": "string"}},
                     "required": ["start", "end"], "additionalProperties": False}},
    {"name": "get_screen_texts",
     "description": "Figma 파일에서 화면(프레임) 하나의 글자 노드만 골라 줘요. frame_id를 비우면 모든 화면을 줘요.",
     "inputSchema": {"type": "object",
                     "properties": {"frame_id": {"type": "string", "description": "예: 1:20"}},
                     "additionalProperties": False}},
]
WRITE_TOOLS = [
    {"name": "update_inquiry_status",
     "description": "문의 하나의 상태를 바꿔요.",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "status": {"type": "string"}},
                     "required": ["id", "status"], "additionalProperties": False}},
    {"name": "set_text",
     "description": "글자 노드 하나의 문구를 바꿔요.",
     "inputSchema": {"type": "object", "properties": {"node_id": {"type": "string"}, "characters": {"type": "string"}},
                     "required": ["node_id", "characters"], "additionalProperties": False}},
]


def _ok(data):
    return {"resultType": "complete",
            "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
            "structuredContent": data, "isError": False}


def _err(msg):
    return {"resultType": "complete", "content": [{"type": "text", "text": msg}], "isError": True}


def text_nodes(frame):
    return [{"id": c["id"], "name": c["name"], "characters": c["characters"]}
            for c in frame["children"] if c["type"] == "TEXT"]


class MiniServer:
    """Notion 쪽과 Figma 쪽 도구를 한 서버에 모은 흉내. 실제로는 서버가 둘(Notion MCP, Figma MCP)이다."""

    def __init__(self, allow_write=False, inquiries=None, figma=None):
        self.db = copy.deepcopy(inquiries or D.load_inquiries())
        self.figma = copy.deepcopy(figma or D.load_figma())
        self.allow_write = allow_write
        self.log = []          # (도구 이름, 인자)
        self.writes = []       # 실제로 바뀐 것

    # ---- 데이터 접근 ----
    @property
    def rows(self):
        return self.db["data_sources"]["고객 문의"]["rows"]

    @property
    def frames(self):
        return self.figma["document"]["children"][0]["children"]

    def screen_frame(self, screen_id):
        for r in self.db["data_sources"]["화면 목록"]["rows"]:
            if r["id"] == screen_id:
                return r["properties"]["figma_frame"]
        return None

    # ---- MCP 메시지 ----
    def tools(self):
        return READ_TOOLS + (WRITE_TOOLS if self.allow_write else [])

    def handle(self, req):
        """JSON-RPC 요청 dict 하나 -> 응답 dict 하나."""
        rid, method, params = req.get("id"), req.get("method"), req.get("params", {})
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"resultType": "complete", "tools": self.tools()}}
        if method != "tools/call":
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        name, args = params.get("name"), params.get("arguments", {})
        if name not in {t["name"] for t in self.tools()}:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        self.log.append((name, args))
        bad = self._valid(name, args)
        result = _err(bad) if bad else getattr(self, "_t_" + name)(**args)
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _valid(self, name, args):
        schema = next(t for t in self.tools() if t["name"] == name)["inputSchema"]
        for k in schema.get("required", []):
            if k not in args:
                return f"'{k}' 칸이 필요해요."
        for k in args:
            if k not in schema["properties"]:
                return f"'{k}' 칸은 이 도구에 없어요."
        for k in ("start", "end"):
            if k in args and not DATE_RE.match(str(args[k])):
                return f"'{k}'는 YYYY-MM-DD 모양이어야 해요. 받은 값: {args[k]}"
        return None

    # ---- 도구 본체 ----
    def _t_query_inquiries(self, start, end, type=None, cursor=None):
        hits = [r for r in self.rows if start <= r["properties"]["접수일"][:10] <= end
                and (type is None or r["properties"]["유형"] == type)]
        off = int(cursor) if cursor else 0
        page = hits[off: off + MAX_ROWS]
        nxt = str(off + MAX_ROWS) if off + MAX_ROWS < len(hits) else None
        return _ok({"rows": page, "next_cursor": nxt})

    def _t_get_screen_texts(self, frame_id=None):
        fr = [f for f in self.frames if frame_id is None or f["id"] == frame_id]
        if not fr:
            return _err(f"프레임 {frame_id}를 찾지 못했어요.")
        return _ok({"frames": [{"id": f["id"], "name": f["name"], "texts": text_nodes(f)} for f in fr]})

    def _t_update_inquiry_status(self, id, status):
        for r in self.rows:
            if r["id"] == id:
                if r["properties"]["상태"] != status:
                    self.writes.append(("notion", id, r["properties"]["상태"], status))
                    r["properties"]["상태"] = status
                return _ok({"id": id, "상태": status})
        return _err(f"문의 {id}가 없어요.")

    def _t_set_text(self, node_id, characters):
        for f in self.frames:
            for c in f["children"]:
                if c["id"] == node_id and c["type"] == "TEXT":
                    self.writes.append(("figma", node_id, c["characters"], characters))
                    c["characters"] = characters
                    return _ok({"node_id": node_id, "characters": characters})
        return _err(f"글자 노드 {node_id}가 없어요.")


class Client:
    """요청 번호를 붙여 tools/call을 보내는 작은 클라이언트."""

    def __init__(self, server):
        self.server = server
        self.next_id = 1

    def call(self, name, **arguments):
        req = {"jsonrpc": "2.0", "id": self.next_id, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}
        self.next_id += 1
        return self.server.handle(req)

    def list_tools(self):
        req = {"jsonrpc": "2.0", "id": self.next_id, "method": "tools/list", "params": {}}
        self.next_id += 1
        return self.server.handle(req)
