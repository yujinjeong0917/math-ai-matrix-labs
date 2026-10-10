# 3장. 모델은 왜 내 파일을 못 볼까, 도구마다 따로 이으면 무엇이 터질까 (`ch03_tools_mcp`)

웹 챕터: `math-ai-matrix/tracks/tools/03-tools-mcp.html`

관통 프로젝트(가상의 작은 팀이 만드는 "주간 고객 문의 요약 보고서")에서, 2장까지 사람이 매주 복사해 붙여 넣던 문의 120건을 모델이 도구로 직접 읽게 바꾼다. 네트워크·API 키 없이, 표준 라이브러리 파이썬만으로 재현한다.

1. 도구 호출 왕복(요청의 `tools` → 응답의 `tool_use` → 프로그램이 실행 → `tool_result`)을 직접 만든 JSON으로 한 줄씩 읽고, 대본대로 답하는 가짜 모델로 고리를 돌린다.
2. stdio JSON-RPC MCP 서버·클라이언트를 만들어 진짜 자식 프로세스로 왕복한다. 현재 개정판 2026-07-28(`server/discover` → `tools/list` → `tools/call`, 요청마다 `_meta`)과 이전 개정판 2025-11-25(`initialize` → `notifications/initialized` → `tools/list` → `tools/call`)를 둘 다 받는다.
3. Claude Desktop(`claude_desktop_config.json`)과 Claude Code(`.mcp.json`)의 `mcpServers` 설정 예를 읽고, 그 설정대로 서버를 띄워 본다.
4. 앱 N개 × 도구 M개를 짝마다 이을 때(N×M)와 공통 규약으로 이을 때(N+M)의 연결 수, 한쪽이 바뀔 때 고칠 곳의 수를 센다.

## 실행
```bash
uv run python tools/ch03_tools_mcp/experiments.py   # results/ch03.json 갱신
uv run pytest -q tools/ch03_tools_mcp
```
서버만 따로 띄워 손으로 한 줄씩 보내 볼 수도 있다(한 줄에 JSON 하나, 끝내려면 Ctrl-D).
```bash
uv run python tools/ch03_tools_mcp/tools_ch03_server.py
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}
```

## 파일
| 파일 | 역할 |
|---|---|
| `tools_ch03_server.py` | MCP 서버(stdio, 두 세대). 도구 3개(`count_by_category`, `list_inquiries`, `get_inquiry`), 자료 1개, 프롬프트 틀 1개. `--legacy-only`로 옛 서버 흉내 |
| `tools_ch03_client.py` | MCP 클라이언트. `server/discover`로 세대를 알아내고, 옛 서버면 `initialize`로 되돌아간다. 읽기마다 시간 제한 |
| `tools_ch03_loop.py` | 도구 호출 고리(가짜 모델), MCP 도구 정의 → API `tools` 칸 변환, 한 주 토큰·금액 계산 |
| `tools_ch03_config.py` | `mcpServers` 설정 읽기, Claude Code식 `${VAR}`, `${VAR:-기본값}` 바꾸기 |
| `tools_ch03_count.py` | N×M, N+M, 차이 (N−1)(M−1)−1, 바뀜이 번지는 범위 |
| `tools_ch03_data.py` | 가상 문의 120건 만들기(고정 시드 3) |
| `experiments.py` | E0~E8 |
| `data/inquiries_2026-W41.json` | 가상 문의 120건(유형 9가지, 건수는 가정값) |
| `data/tool_call_example.json` | 직접 만든 도구 호출 왕복 예 |
| `data/claude_desktop_config.example.json`, `data/mcp.example.json` | 직접 만든 설정 예(경로는 자리표시) |
| `data/project_assumptions.json` | 토큰 가정값(도구 사용 시스템 프롬프트 286토큰만 공식 문서 값) |
| `data/prices_2026-10-10.json` | 가격표(Claude Sonnet 5.5, 2장과 같은 값). 값이 바뀌면 날짜가 다른 새 파일을 만든다 |
| `checks/ch03.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- 가짜 모델은 절 이름을 보고 부를 도구를 정해진 대로 고른다. 진짜 모델이 도구를 언제, 어떤 인자로 부를지는 재현하지 않는다.
- 서버는 사양의 일부만 담았다. 인증(HTTP용), Streamable HTTP, `subscriptions/listen`, 여러 번 오가는 요청(MRTR), 페이지 나누기, 진행 알림은 없다. 2026-07-28에서 폐기 예정이 된 roots·sampling·logging은 만들지 않았다.
- 토큰 수는 세지 않은 가정값이다(문의 1건 200토큰은 2장의 24,000 ÷ 120).
- 설정 예의 실행 시험은 명령을 지금 파이썬(`sys.executable`)으로, 자리표시 경로를 이 저장소 경로로 바꿔서 돌린다.

확인일 2026-10-10. 사양: https://modelcontextprotocol.io/specification/2026-07-28
