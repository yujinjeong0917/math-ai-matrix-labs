# 4장. 문서는 왜 서로 연결돼야 하나 (`ch04_notion_relations`)

웹 챕터: `math-ai-matrix/tracks/productivity/04-notion-relations.html`

Notion을 띄우지 않고, 관통 프로젝트의 결정 10개·할 일 20개·스펙 4개를 두 가지로 만든 파이썬 모형에서 재현한다.

1. 서로 링크 없는 일반 페이지(결정 페이지 + 할 일 체크리스트 + 스펙 페이지)로 질문 5개에 답하면 어디서 막히나.
2. 같은 연결을 relation으로 두고 필터·rollup·수식으로 답하면 무엇이 바뀌나. 사람이 연결을 빠뜨리는 비율(기록률 r)을 같게 두고 비교한다.
3. 빈칸 검사가 빠진 연결을 얼마나 잡나, 할 일 경로로 화면 영향을 추론하면 왜 틀리나(Codd의 연결 함정), 손으로 적은 개수는 얼마나 어긋나나.

## 실행
```bash
uv run python productivity/ch04_notion_relations/experiments.py   # results/ch04.json, data/*.json·md 갱신
uv run pytest -q productivity/ch04_notion_relations
```
표준 라이브러리만 쓴다. NumPy·PyTorch로 옮길 계산이 없어, 같은 질문을 순수 파이썬(선택·조인·묶어 세기)과
sqlite3(SQL), Notion 수식 모양 rollup 세 방식으로 구현해 무작위 DB 500개에서 대조한다.

## 파일
| 파일 | 역할 |
|---|---|
| `productivity_ch04_data.py` | 결정·할 일·스펙 합성 데이터, 잃을 수 있는 연결 44개 |
| `productivity_ch04_pages.py` | 일반 페이지 만들기와 찾는 사람의 규칙(읽은 줄 수) |
| `productivity_ch04_relational.py` | relation 표, 관계 대수 세 가지, rollup, 질문 5개, 빈칸 검사, sqlite3 대응 구현 |
| `experiments.py` | E0~E7 |
| `specs/notion_schema.md` | 독자가 Notion에서 만들 데이터베이스 4개 |
| `specs/question_rules.md` | 질문·찾는 규칙·채점 규칙(실험 전에 정함) |
| `checks/ch04.csv` | 독자 기록 양식(예시 행은 합성 모형 값) |
| `data/` | 프로젝트 데이터(JSON), 기록률 1.0일 때의 일반 페이지 모습(Markdown) |

## 한계
- 결과는 합성 데이터와 찾는 규칙이 정한다. 장부 계산 확인이지 실제 팀 측정이 아니다.
- 시간은 잴 수 없어 읽은 줄 수·보기의 행 수로 바꿔 셌다.
- 빈칸 검사 뒤에는 드러난 연결을 사람이 정확히 다시 채운다고 가정했다.
- Notion의 실제 동작(계산 종류, 내보내기)은 확인일(2026-10-08)의 도움말 기준이다.

확인일 2026-10-08.
