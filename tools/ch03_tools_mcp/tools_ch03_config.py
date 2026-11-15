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

# 그룹 1: 변수 이름, 그룹 2: ':-' 뒤의 기본값(없으면 None). 기본값이 빈 문자열이면 빈 문자열로 바뀐다.
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(s, env, warnings):
    """문자열 안의 ${VAR}, ${VAR:-기본값}을 환경 변수로 바꾼다. 못 바꾸면 경고를 쌓고 원문을 둔다."""
    def rep(m):
        name, default = m.group(1), m.group(2)
        # 환경 변수가 있으면 값이 기본값보다 우선한다.
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
            # code 클라이언트만 변수 바꾸기를 한다(위 docstring의 근거).
            fix = (lambda s: expand(s, env, warnings))
        else:
            typ = "stdio"                     # Desktop 문서의 예는 로컬(stdio) 서버만 보여 준다
            # desktop은 변수 바꾸기를 하지 않는다(문서에서 찾지 못해서, 문서 확인 필요).
            fix = (lambda s: s)
        # stdio가 아닌 서버(http 등)는 실행 인자 대신 url만 해석한다.
        if typ != "stdio":
            out[name] = {"type": typ, "url": fix(entry.get("url", "")), "warnings": warnings}
            continue
        if "command" not in entry:
            out[name] = {"skipped": f'MCP server "{name}" has no "command"'}
            continue
        out[name] = {"type": "stdio",
                     # argv[0]은 실행 파일, 나머지는 인자. env 값도 같은 규칙으로 바꾼다.
                     "argv": [fix(entry["command"])] + [fix(a) for a in entry.get("args", [])],
                     "env": {k: fix(v) for k, v in entry.get("env", {}).items()},
                     "warnings": warnings}
    return out


def line_by_line(path):
    """설정 파일의 줄마다 (줄 번호, 글)을 돌려준다. 본문 표와 맞춰 보는 데 쓴다."""
    return [(i, l.rstrip("\n")) for i, l in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1)]
