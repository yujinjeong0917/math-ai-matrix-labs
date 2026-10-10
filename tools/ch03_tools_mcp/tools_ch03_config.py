"""AI 도구 실전 3장: MCP 설정 JSON(mcpServers 블록) 읽기.

Claude Desktop(claude_desktop_config.json)과 Claude Code(.mcp.json)는 둘 다 "mcpServers" 아래에
서버 이름 → {command, args, env} 를 적는다. 다른 점(확인일 2026-10-10):
- Claude Code는 "type"("stdio", "http" 등)을 적을 수 있고, type이 없으면 stdio로 읽는다.
  url이 있는데 type이 없으면 설정 오류로 건너뛴다.
  https://code.claude.com/docs/en/mcp
- Claude Code는 command, args, env, url, headers 안의 ${VAR}, ${VAR:-기본값}을 환경 변수로 바꾼다.
  변수가 없고 기본값도 없으면 ${VAR} 글자를 그대로 둔다(경고만 낸다).
- Claude Desktop 문서의 예에는 type이 없고 경로는 절대 경로로 쓰라고 한다. 변수 바꾸기는 문서에서 찾지 못해 하지 않는다.
  https://modelcontextprotocol.io/docs/develop/connect-local-servers
"""

import json
import re
from pathlib import Path

VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(s, env, warnings):
    def rep(m):
        name, default = m.group(1), m.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        warnings.append(f"환경 변수 {name}가 없어요")
        return m.group(0)
    return VAR.sub(rep, s)


def read_config(path_or_dict, client, env=None):
    """설정을 읽어 서버마다 실행 정보를 돌려준다. client는 "desktop" 또는 "code"."""
    cfg = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    env = dict(env or {})
    out = {}
    for name, entry in cfg["mcpServers"].items():
        warnings = []
        typ = entry.get("type")
        if client == "code":
            if typ is None and "url" in entry:
                out[name] = {"skipped": f'MCP server "{name}" has a "url" but no "type"'}
                continue
            typ = typ or "stdio"
            if typ == "streamable-http":
                typ = "http"                  # 사양 이름도 별칭으로 받는다
            fix = (lambda s: expand(s, env, warnings))
        else:
            typ = "stdio"                     # Desktop 문서의 예는 로컬(stdio) 서버만 보여 준다
            fix = (lambda s: s)
        if typ != "stdio":
            out[name] = {"type": typ, "url": fix(entry.get("url", "")), "warnings": warnings}
            continue
        if "command" not in entry:
            out[name] = {"skipped": f'MCP server "{name}" has no "command"'}
            continue
        out[name] = {"type": "stdio",
                     "argv": [fix(entry["command"])] + [fix(a) for a in entry.get("args", [])],
                     "env": {k: fix(v) for k, v in entry.get("env", {}).items()},
                     "warnings": warnings}
    return out


def line_by_line(path):
    """설정 파일의 줄마다 (줄 번호, 글)을 돌려준다. 본문 표와 맞춰 보는 데 쓴다."""
    return [(i, l.rstrip("\n")) for i, l in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1)]
