# 5장. 스프레드시트 사고는 왜 반복되나 (`ch05_sheets`)

웹 챕터: `math-ai-matrix/tracks/productivity/05-sheets.html`

Google Sheets를 띄우지 않고, 관통 프로젝트(동네 스터디룸 예약)의 2주 운영 데이터(합성, 지점 20곳, 예약 2,460건)로
스프레드시트 사고 여섯 가지를 파이썬 모형에서 재현한다.

1. 자유 입력 시트: 손으로 적은 범위 수식, 아래에 붙이는 행, 자동 형식 변환, 행 수 상한이 있는 형식으로 저장.
2. 명세 통합 문서(`specs/sheets_workbook_spec.md`): raw(붙여 넣기만, 보호) -> clean(수식으로만) -> pivot -> metrics, checks C1~C9.
3. 같은 입력에서 두 방식이 사고를 결과 전에 잡는지 비교하고, 명세도 못 잡는 경우를 따로 센다.

## 실행
```bash
OMP_NUM_THREADS=2 uv run python productivity/ch05_sheets/experiments.py   # results/ch05.json, data/*.csv 갱신
uv run pytest -q productivity/ch05_sheets
```
표준 라이브러리와 NumPy만 쓴다. 피벗(묶어 세기)은 dict, sqlite3, NumPy `bincount` 세 방식으로 구현해 무작위 데이터 500개에서 대조한다.
PyTorch로 옮길 계산이 없어 쓰지 않았다.

## 파일
| 파일 | 역할 |
|---|---|
| `productivity_ch05_data.py` | 합성 예약 데이터, 지저분한 사본, 이벤트 로그, 첫 화면 손계산 표 |
| `productivity_ch05_sheet.py` | 자유 시트 모형: 범위 수식과 행 추가, 자동 형식 변환 흉내, 65,536행 저장 |
| `productivity_ch05_workbook.py` | 명세 통합 문서: raw 보호, clean, 피벗 세 구현, 지표, checks |
| `experiments.py` | E0~E10과 요약 |
| `specs/sheets_workbook_spec.md` | 독자가 Google Sheets에서 만들 통합 문서 |
| `checks/ch05.csv` | 독자 기록 양식(예시 행은 합성 모형 값) |
| `data/` | `bookings.csv`(깨끗한 원본), `bookings_messy.csv`(표기가 섞인 사본), `branches.csv` |

## 한계
- 결과는 합성 데이터와 모형 규칙이 정한다. 범위가 행 추가에 어떻게 반응하는지, 자동 변환 규칙은 실제 도구를 단순화한 가정이다.
- 재계산 시간은 파이썬 모형의 시간이지 Google Sheets의 시간이 아니다.
- 65,536행은 Excel 97-2003(.xls) 형식의 한도다. Google Sheets의 한도는 셀 2천만 개 또는 100MB다(도움말).

확인일 2026-10-08.
