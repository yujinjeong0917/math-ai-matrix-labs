# 5장. Notion·Figma를 AI와 이으면 무엇이 쉬워지고 무엇이 그대로일까 (`ch05_notion_figma`)

웹 챕터: `math-ai-matrix/tracks/tools/05-notion-figma.html`

관통 프로젝트(가상의 작은 팀이 만드는 "주간 고객 문의 요약 보고서")에서, 고객 문의는 Notion 데이터베이스에, 앱 화면 문구는 Figma 파일에 있다고 두고 "이번 주 문의 요약 → 고객이 인용한 화면 문구 점검" 왕복을 네트워크·API 키 없이 재현한다.

1. Notion 데이터 소스와 Figma 노드 트리를 흉내 낸 로컬 JSON(고정 시드 5)을 만든다.
2. MCP 명세(2026-07-28판)의 `tools/list`·`tools/call` 모양을 따른 도구 두 개 `query_inquiries`, `get_screen_texts`를 프로세스 안 서버에 붙인다.
3. 이번 주 문의 120건을 100행씩 두 번 읽어 유형별로 세고, 본문에서 「」로 인용된 화면 문구를 Figma 글자 노드에서 찾아 "관련 화면에 있음 / 다른 화면에만 있음 / Figma에 없음"으로 나눈다.
4. 문의 본문 하나에 숨은 지시(프롬프트 주입 예)를 넣고, 도구 결과를 지시로 읽는 순진한 루프와 데이터로만 읽는 루프를 읽기 전용 / 쓰기 도구 연결 두 조건에서 비교한다. 자세한 방어는 8장에서 다룬다.

## 실행
```bash
uv run python tools/ch05_notion_figma/experiments.py   # results/ch05.json 갱신
uv run pytest -q tools/ch05_notion_figma
python tools/ch05_notion_figma/tools_ch05_data.py      # data/ 다시 만들기(바이트까지 같아야 함)
```
표준 라이브러리만 쓴다. 실제 Notion·Figma·모델을 부르지 않는다.

## 파일
| 파일 | 역할 |
|---|---|
| `tools_ch05_data.py` | 가상 문의 138건(이번 주 120 + 지난주 18), 화면 목록 6개, Figma 프레임 6개 생성기 |
| `tools_ch05_mcp.py` | MCP 모양의 도구 서버 흉내(MiniServer)와 클라이언트. 읽기 도구 2개, 실험용 쓰기 도구 2개 |
| `tools_ch05_roundtrip.py` | 요약 → 점검 왕복, 프롬프트 주입 장난감, 크기 비교 |
| `experiments.py` | E0~E6 |
| `data/notion_inquiries.json` | "고객 문의"(속성: 제목, 접수일, 유형, 상태, 화면 relation, 본문)와 "화면 목록" |
| `data/figma_file.json` | Figma 노드 트리 흉내(id, name, type, characters, absoluteBoundingBox, fills, style) |
| `data/mcp_access_2026-10-10.json` | Notion MCP·Figma MCP·MCP 명세에서 확인한 사실. 항목마다 출처 URL과 확인일. 바뀌면 날짜가 다른 새 파일을 만든다 |
| `checks/ch05.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- 모델을 부르지 않는다. 어떤 도구를 부를지는 고정 규칙이 대신한다. 실제 모델은 다른 순서로 부르거나 더 많이 부를 수 있다.
- 주입 장난감의 "지시 알아보기" 규칙은 이 장의 예문 한 가지만 알아본다. 실제 모델이 어떤 문장을 따를지는 이 모형으로 알 수 없고, 구별하는 루프도 완전한 방어가 아니다(8장).
- 실제 Notion MCP·Figma MCP의 도구 이름과 응답 모양은 이 흉내와 다르다. 같은 것은 "MCP 메시지 모양"과 "읽기 한 번에 100행" 같은 확인한 조건뿐이다.
- 인용 문구 찾기는 글자 그대로 포함되는지만 본다. 띄어쓰기나 표현이 조금 다른 인용은 놓친다.

확인일 2026-10-10.
