"""AI 도구 실전 3장: 표준 라이브러리로 만든 아주 작은 MCP 서버(stdio).

문의 파일 하나를 읽고 도구 3개, 자료 1개, 프롬프트 틀 1개를 내놓는다.
- stdin에서 한 줄에 JSON-RPC 메시지 하나를 읽고, stdout에는 한 줄에 응답 하나만 쓴다(그 밖의 글은 stderr).
- 두 세대를 모두 받는다(dual-era).
  · 현재 개정판 2026-07-28(modern): 요청마다 params._meta에 버전과 클라이언트 능력을 싣는다. server/discover로 시작할 수 있다.
  · 이전 개정판 2025-11-25(legacy): initialize → notifications/initialized로 연결을 먼저 맺는다.
- `--legacy-only`를 주면 옛 서버처럼 굴어서(server/discover를 모름) 클라이언트의 되돌아가기(fallback)를 시험할 수 있다.

실행: python tools_ch03_server.py [--data 경로] [--legacy-only]   (경로는 환경 변수 INQUIRY_DATA로도 줄 수 있다)
사양: https://modelcontextprotocol.io/specification/2026-07-28 (확인일 2026-10-10)
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
SUPPORTED = [MODERN, LEGACY]
SERVER_INFO = {"name": "inquiries-mcp", "version": "0.3.0"}
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
UNSUPPORTED_VERSION = -32022           # 2026-07-28에서 새로 정한 번호(초안의 -32004에서 바뀜)
DEFAULT_DATA = Path(__file__).parent / "data" / "inquiries_2026-W41.json"


def log(*a):
    print("[inquiries-mcp]", *a, file=sys.stderr, flush=True)


class InquiryServer:
    def __init__(self, data_path, legacy_only=False):
        self.data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.categories = self.data["categories"]
        self.legacy_only = legacy_only
        self.legacy_version = None     # initialize로 정한 옛 세대 버전(프로세스 하나에만 유효)
        self.legacy_ready = False

    # ---------- 내놓는 것: 도구, 자료, 프롬프트 틀 ----------
    def tools(self):
        week = {"type": "string", "description": "ISO 주. 예: 2026-W41"}
        return [   # 순서를 고정한다(사양: 같은 목록은 같은 순서로 → 프롬프트 캐시 적중에 유리)
            {"name": "count_by_category", "title": "유형별 문의 건수",
             "description": "한 주의 고객 문의를 유형별로 세어 돌려줘요.",
             "inputSchema": {"type": "object", "properties": {"week": week}, "required": ["week"],
                             "additionalProperties": False}},
            {"name": "list_inquiries", "title": "유형별 문의 목록",
             "description": "한 주의 고객 문의 가운데 한 유형만 골라 접수 시각 순으로 돌려줘요.",
             "inputSchema": {"type": "object",
                             "properties": {"week": week,
                                            "category": {"type": "string", "enum": list(self.categories)}},
                             "required": ["week", "category"], "additionalProperties": False}},
            {"name": "get_inquiry", "title": "문의 한 건",
             "description": "문의 번호(예: #0007)로 한 건을 돌려줘요.",
             "inputSchema": {"type": "object", "properties": {"id": {"type": "string", "pattern": "^#[0-9]{4}$"}},
                             "required": ["id"], "additionalProperties": False}},
        ]

    def resources(self):
        return [{"uri": f"inquiries://{self.data['week']}/all", "name": f"{self.data['week']} 문의 전체",
                 "mimeType": "application/json"}]

    def prompts(self):
        return [{"name": "weekly_section", "title": "주간 보고서 한 절",
                 "description": "유형 하나로 보고서 한 절을 쓰게 하는 틀",
                 "arguments": [{"name": "category", "required": True}]}]

    # ---------- 도구 실행 ----------
    def _check_args(self, tool, args):
        schema = tool["inputSchema"]
        if not isinstance(args, dict):
            return "arguments는 객체여야 해요."
        for k in schema.get("required", []):
            if k not in args:
                return f"'{k}' 값이 빠졌어요."
        for k, v in args.items():
            prop = schema["properties"].get(k)
            if prop is None:
                return f"'{k}'는 받지 않는 값이에요. 받는 값: {', '.join(schema['properties'])}"
            if not isinstance(v, str):
                return f"'{k}'는 글자여야 해요."
            if "enum" in prop and v not in prop["enum"]:
                return f"'{v}'는 없는 유형이에요. 쓸 수 있는 값: {', '.join(prop['enum'])}"
        if "week" in args and args["week"] != self.data["week"]:
            return f"{args['week']} 문의는 없어요. 이 서버에는 {self.data['week']}만 있어요."
        return None

    def run_tool(self, name, args):
        """(구조화 결과, 오류 문장) 중 하나를 돌려준다."""
        tool = next(t for t in self.tools() if t["name"] == name)
        err = self._check_args(tool, args)
        if err:
            return None, err
        items = self.data["items"]
        if name == "count_by_category":
            c = Counter(i["category"] for i in items)
            return {"week": self.data["week"], "total": len(items),
                    "counts": {k: c.get(k, 0) for k in self.categories}}, None
        if name == "list_inquiries":
            sel = [i for i in items if i["category"] == args["category"]]
            return {"week": self.data["week"], "category": args["category"], "count": len(sel), "items": sel}, None
        if name == "get_inquiry":
            hit = [i for i in items if i["id"] == args["id"]]
            if not hit:
                return None, f"{args['id']} 문의를 찾지 못했어요."
            return hit[0], None
        raise AssertionError(name)

    # ---------- JSON-RPC 처리 ----------
    def _modern_result(self, body):
        return {"resultType": "complete", **body, "_meta": {META_SERVER: SERVER_INFO}}

    def handle(self, msg):
        """메시지 하나를 받아 응답 dict(또는 알림이면 None)를 돌려준다."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return _err(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "Invalid Request")
        mid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
        if "id" not in msg:                       # 알림: 답하지 않는다
            if method == "notifications/initialized" and self.legacy_version:
                self.legacy_ready = True
                log("legacy session ready", self.legacy_version)
            return None
        meta = params.get("_meta") or {}
        if method == "initialize":
            return self._initialize(mid, params)
        if META_VERSION in meta and not self.legacy_only:
            return self._modern(mid, method, params, meta)
        if self.legacy_version:                   # initialize를 마친 옛 세대 연결
            return self._legacy(mid, method, params)
        if self.legacy_only:
            if method == "server/discover":
                return _err(mid, METHOD_NOT_FOUND, f"Method not found: {method}")
            return _err(mid, INVALID_REQUEST, "Server not initialized")
        return _err(mid, INVALID_PARAMS, f"Missing required _meta field: {META_VERSION}")

    def _initialize(self, mid, params):
        asked = params.get("protocolVersion")
        if not self.legacy_only and asked == MODERN:
            # 현재 세대에는 initialize가 없다. 받는 버전을 알려 준다(사양 권장).
            return _err(mid, METHOD_NOT_FOUND, f"initialize is not part of {MODERN}; supported: {SUPPORTED}")
        self.legacy_version = LEGACY              # 옛 세대에서 아는 버전은 하나뿐이라 늘 이것으로 답한다
        log("initialize", asked, "->", self.legacy_version)
        return _ok(mid, {"protocolVersion": self.legacy_version,
                         "capabilities": {"tools": {"listChanged": False}, "resources": {}, "prompts": {}},
                         "serverInfo": SERVER_INFO,
                         "instructions": "주간 고객 문의를 읽는 도구예요."})

    def _modern(self, mid, method, params, meta):
        if meta.get(META_VERSION) != MODERN:
            return _err(mid, UNSUPPORTED_VERSION, "Unsupported protocol version",
                        {"supported": SUPPORTED, "requested": meta.get(META_VERSION)})
        if META_CAPS not in meta:
            return _err(mid, INVALID_PARAMS, f"Missing required _meta field: {META_CAPS}")
        if method == "server/discover":
            return _ok(mid, {"resultType": "complete", "supportedVersions": SUPPORTED,
                             "capabilities": {"tools": {"listChanged": False}, "resources": {}, "prompts": {}},
                             "_meta": {META_SERVER: SERVER_INFO},
                             "instructions": "주간 고객 문의를 읽는 도구예요.", "ttlMs": 3_600_000,
                             "cacheScope": "private"})
        body = self._common(mid, method, params)
        if "error" in body:
            return body
        cacheable = method in ("tools/list", "resources/list", "prompts/list", "resources/read")
        extra = {"ttlMs": 300_000, "cacheScope": "private"} if cacheable else {}
        return _ok(mid, self._modern_result({**body["result"], **extra}))

    def _legacy(self, mid, method, params):
        if not self.legacy_ready and method != "ping":
            log("warning: request before notifications/initialized")
        if method == "ping":
            return _ok(mid, {})
        return self._common(mid, method, params)

    def _common(self, mid, method, params):
        if method == "tools/list":
            return _ok(mid, {"tools": self.tools()})
        if method == "tools/call":
            name = params.get("name")
            if name not in [t["name"] for t in self.tools()]:
                return _err(mid, INVALID_PARAMS, f"Unknown tool: {name}")
            data, err = self.run_tool(name, params.get("arguments", {}))
            if err:   # 도구 실행 오류는 결과 안에 isError로 담는다(모델이 읽고 고칠 수 있게)
                return _ok(mid, {"content": [{"type": "text", "text": err}], "isError": True})
            text = json.dumps(data, ensure_ascii=False)
            return _ok(mid, {"content": [{"type": "text", "text": text}], "structuredContent": data,
                             "isError": False})
        if method == "resources/list":
            return _ok(mid, {"resources": self.resources()})
        if method == "resources/read":
            uri = params.get("uri")
            if uri != self.resources()[0]["uri"]:
                return _err(mid, INVALID_PARAMS, f"Resource not found: {uri}")
            return _ok(mid, {"contents": [{"uri": uri, "mimeType": "application/json",
                                           "text": json.dumps(self.data["items"], ensure_ascii=False)}]})
        if method == "prompts/list":
            return _ok(mid, {"prompts": self.prompts()})
        if method == "prompts/get":
            cat = (params.get("arguments") or {}).get("category")
            if params.get("name") != "weekly_section" or cat not in self.categories:
                return _err(mid, INVALID_PARAMS, "Unknown prompt or category")
            return _ok(mid, {"messages": [{"role": "user", "content": {
                "type": "text", "text": f"이번 주 '{cat}' 문의를 읽고 보고서 한 절을 다섯 줄 안으로 써 줘."}}]})
        return _err(mid, METHOD_NOT_FOUND, f"Method not found: {method}")


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message, data=None):
    e = {"code": code, "message": message}
    if data is not None:
        e["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": e}


def serve(stdin, stdout, server):
    for line in stdin:                       # stdin이 닫히면(EOF) 반복이 끝나고 서버도 끝난다
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out = _err(None, PARSE_ERROR, "Parse error")
        else:
            out = server.handle(msg)
        if out is not None:
            stdout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")  # 한 줄, 줄바꿈 없음
            stdout.flush()
    log("stdin closed, bye")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    legacy_only = "--legacy-only" in argv
    data = os.environ.get("INQUIRY_DATA") or str(DEFAULT_DATA)
    if "--data" in argv:
        data = argv[argv.index("--data") + 1]
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    log("start", "legacy-only" if legacy_only else "dual-era", data)
    serve(sys.stdin, sys.stdout, InquiryServer(data, legacy_only))


if __name__ == "__main__":
    main()
