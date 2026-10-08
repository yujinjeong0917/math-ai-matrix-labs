# 3장. 개발자에게 넘길 때 무엇이 새나 (`ch03_handoff`)

웹 챕터: `math-ai-matrix/tracks/productivity/03-handoff.html`

Figma를 띄우지 않고, 화면 4개를 "화면 x 상태 x 전환" 표로 만든 파이썬 모형에서 세 가지를 재현한다.

1. 이미지 + 메신저 설명만으로 넘기면 개발자가 질문 20개 중 몇 개에 답하지 못하고, 몇 개를 틀리게 만드나.
2. 체크리스트 + 프로토타입, Dev Mode(기본 상태만 그린 파일 / 체크리스트를 채운 파일)로 넘기면 무엇이 바뀌나.
3. 프로토타입을 그래프로 보고 닿지 않는 상태, 막다른 상태, 돌아갈 길 없는 오류를 찾는 검사.

## 실행
```bash
uv run python productivity/ch03_handoff/experiments.py   # results/ch03.json, data/*.json 갱신
uv run pytest -q productivity/ch03_handoff
```
표준 라이브러리만 쓴다. NumPy·PyTorch로 옮길 계산이 없어, 대신 같은 검사를 두 방식으로 구현해 대조한다
(그래프 검사: 너비 우선 탐색 vs 워셜 전이 폐포, 완결성: 칸 세기 vs 합 공식).

## 파일
| 파일 | 역할 |
|---|---|
| `productivity_ch03_model.py` | 화면·상태·전환, 질문 20개, 넘겨주는 방식 5가지, 채점, 질문 왕복, 토큰 이름 vs 값 복사 |
| `productivity_ch03_checks.py` | 프로토타입 그래프 검사 두 구현, WCAG 대비 계산, 터치 영역 검사 |
| `experiments.py` | E1~E7 |
| `specs/handoff_checklist.md` | 독자가 Figma에서 채울 체크리스트 |
| `specs/question_rules.md` | 질문·추측·채점 규칙(실험 전에 정함) |
| `checks/ch03.csv` | 독자 기록 양식(예시 행은 합성 모형 값) |
| `data/` | 질문 20개, 전환 18개(JSON) |

## 한계
- 결과는 패키지에 무엇이 들었는지와 추측 규칙이 정한다. 장부 계산 확인이지 실제 팀 측정이 아니다.
- 이미지에서 글자 크기를 읽지 않는다고 둔 것, 추측 기본값은 모형의 가정이다.
- 넘기기 준비 시간은 모형으로 잴 수 없어 "디자이너가 만든 것의 개수"로 바꿨다.
- Dev Mode 요금·좌석 이름은 확인일(2026-10-08) 기준이다.

확인일 2026-10-08.
