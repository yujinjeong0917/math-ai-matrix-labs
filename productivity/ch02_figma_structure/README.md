# 2장. 디자인 파일을 주고받던 시절 (`ch02_figma_structure`)

웹 챕터: `math-ai-matrix/tracks/productivity/02-figma-structure.html`

Figma를 띄우지 않고, 디자인 파일을 "레이어와 속성의 모음"으로 단순화한 파이썬 모형에서
세 가지를 재현한다.

1. 두 사람이 사본을 고친 뒤 파일째 저장하면 무엇이 사라지나(합치는 단위: 파일 / 레이어 / 속성 / 잠금).
2. 버튼을 화면마다 그린 사본과 main component 1개의 변경 비용, override·detach가 남기는 어긋남.
3. 화면 폭을 바꿨을 때 고정 좌표와 auto layout의 차이.

## 실행
```bash
uv run python productivity/ch02_figma_structure/experiments.py   # results/ch02.json, data/*.json 갱신
uv run pytest -q productivity/ch02_figma_structure
```
표준 라이브러리만 쓴다. NumPy·PyTorch 대응 구현은 이 장에서는 의미가 없어 두지 않았고,
대신 같은 규칙을 두 방식으로 구현해 대조한다(속성 단위 LWW: 연산 기록 재생 vs 레지스터 합치기,
auto layout: 엔진 vs 닫힌 식).

## 파일
| 파일 | 역할 |
|---|---|
| `productivity_ch02_layout.py` | auto layout 최소 구현(hug·fill·fixed·min/max·padding·gap), 고정 좌표로 얼리기, 넘침 검사 |
| `productivity_ch02_project.py` | 화면 4개, main component·instance·override·detach, 핑퐁 파일, 명세 검사, 라이브러리 사용률 |
| `productivity_ch02_sync.py` | 두 사람 편집 기록과 합치는 방식 4가지, 이론값 |
| `experiments.py` | E1~E7 비교 실험 |
| `specs/figma_file_spec.md` | 독자가 Figma에서 따를 파일 구조 명세 |
| `checks/ch02.csv` | 독자 기록 양식(예시 행은 합성 실험 값) |
| `data/` | 두 합성 파일의 구조(JSON) |

## 한계
- 글자 폭은 한글 14px, 영숫자 8px, 공백 4px로 둔 모형이다.
- auto layout은 가로·세로 쌓기만 구현했다(wrap, grid, 정렬, Auto 간격 없음).
- override한 속성이 main 변경을 받지 않는다는 동작은 도움말에 직접 적혀 있지 않은 모형의 가정이다 `[확인 필요]`.
- 합치기 모형은 실제 Figma 서버의 삭제·부모 변경·확인 전 값 처리 등을 다루지 않는다.

확인일 2026-10-08.
