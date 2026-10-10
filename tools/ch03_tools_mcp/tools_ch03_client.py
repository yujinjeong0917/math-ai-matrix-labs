"""AI 도구 실전 3장: 표준 라이브러리로 만든 아주 작은 MCP 클라이언트(stdio).

서버 프로그램을 자식 프로세스로 띄우고, 한 줄에 JSON 하나씩 주고받는다.
- connect(): 먼저 현재 세대(2026-07-28) 방식으로 server/discover를 보내 본다.
  · 결과가 오면 현재 세대 서버. 이후 요청마다 _meta에 버전과 능력을 싣는다.
  · 현재 세대 오류(-32022 등)가 오면 현재 세대 서버. 서버가 받는 버전을 골라 다시 보낸다.
  · 그 밖의 오류거나 답이 없으면 옛 세대 서버. initialize → notifications/initialized로 되돌아간다.
- 읽을 때마다 시간 제한을 둬서 서버가 멈춰도 테스트가 끝없이 기다리지 않는다.
"""

import json
import queue
import subprocess
import threading

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
MODERN_ERRORS = {-32020, -32021, -32022}    # 2026-07-28이 정한 오류 번호
CLIENT_INFO = {"name": "weekly-report-script", "version": "0.3.0"}


class RPCError(Exception):
    def __init__(self, error):
        super().__init__(f'{error["code"]}: {error["message"]}')
        self.code, self.message, self.data = error["code"], error["message"], error.get("data")


class StdioClient:
    def __init__(self, command, env=None, timeout=5.0, cwd=None):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, cwd=cwd, text=True, encoding="utf-8", bufsize=1)
        self.timeout = timeout
        self.next_id = 1
        self.era = None
        self.version = None
        self.transcript = []                 # ("→" 또는 "←", 한 줄 글)
        self.stdout_lines = []               # stdout에서 읽은 모든 줄(사양 위반 검사용)
        self._q = queue.Queue()
        self._err_lines = []
        threading.Thread(target=self._pump, args=(self.proc.stdout, self._q), daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump(self, stream, q):
        for line in stream:
            q.put(line.rstrip("\n"))
        q.put(None)

    def _pump_err(self):
        for line in self.proc.stderr:
            self._err_lines.append(line.rstrip("\n"))

    @property
    def stderr_lines(self):
        return list(self._err_lines)

    # ---------- 보내고 받기 ----------
    def send(self, msg):
        line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
        assert "\n" not in line
        self.transcript.append(("→", line))
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read(self, timeout):
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("서버가 시간 안에 답하지 않았어요")
        if line is None:
            raise EOFError("서버가 끝났어요")
        self.stdout_lines.append(line)
        self.transcript.append(("←", line))
        return json.loads(line)

    def request(self, method, params=None, meta=True, timeout=None):
        mid = self.next_id
        self.next_id += 1
        params = dict(params or {})
        if meta and self.era != "legacy":
            params["_meta"] = self.meta()
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params:
            msg["params"] = params
        self.send(msg)
        while True:
            resp = self._read(self.timeout if timeout is None else timeout)
            if resp.get("id") == mid:
                break                           # 알림 등 다른 메시지는 건너뛴다
        if "error" in resp:
            raise RPCError(resp["error"])
        result = resp["result"]
        result.setdefault("resultType", "complete")   # 옛 서버는 resultType이 없다 → complete로 읽는다(사양)
        return result

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        self.send(msg)

    def meta(self, version=None):
        return {"io.modelcontextprotocol/protocolVersion": version or self.version or MODERN,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
                "io.modelcontextprotocol/clientCapabilities": {}}

    # ---------- 연결 ----------
    def connect(self, probe_timeout=1.0):
        """세대를 알아내고 (era, version, 서버 정보)를 돌려준다."""
        try:
            d = self.request("server/discover", timeout=probe_timeout)
            self.era = "modern"
            self.version = MODERN if MODERN in d["supportedVersions"] else d["supportedVersions"][0]
            return self.era, self.version, d["_meta"]["io.modelcontextprotocol/serverInfo"]
        except RPCError as e:
            if e.code in MODERN_ERRORS:
                self.era = "modern"
                self.version = next(v for v in e.data["supported"] if v >= MODERN)
                return self.era, self.version, None
        except TimeoutError:
            pass
        self.era = "legacy"                      # 옛 세대로 되돌아간다
        r = self.request("initialize", {"protocolVersion": LEGACY, "capabilities": {},
                                        "clientInfo": CLIENT_INFO}, meta=False)
        self.version = r["protocolVersion"]
        self.notify("notifications/initialized")
        return self.era, self.version, r["serverInfo"]

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call_tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self, timeout=5.0):
        """stdin을 닫고 서버가 스스로 끝나길 기다린다. 늦으면 강제로 끝낸다(사양의 종료 순서)."""
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            rc = self.proc.wait(timeout=timeout)
        return rc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.close()
