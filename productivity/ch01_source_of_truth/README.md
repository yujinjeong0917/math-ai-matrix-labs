# 1장. 최신본이 두 개면 아무도 최신본을 모른다 (`ch01_source_of_truth`)

웹 챕터: `math-ai-matrix/tracks/productivity/01-source-of-truth.html`

Figma·Notion·Google Sheets를 띄우지 않고, 관통 프로젝트(동네 스터디룸 예약)의 값 3개(시간당 요금, 운영 시간, 취소 규정)를
세 자리(기획 문서, 디자인 시안, 운영 시트)에 적어 둔 파이썬 모형에서 두 방식을 비교한다.

1. 복사 방식: 값이 세 자리에 따로 있다. 변경 카드 8장을 받은 사람이 자기 자리 한 곳만 고친다.
2. 원본 등록부 방식: 값마다 원본 자리 하나. 고치는 건 원본에서만. 링크·임베드로 이은 자리는 따라오고, 손으로 옮기는 자리는 등록부에 적는다.

같은 카드로 어긋난 칸, 진짜 값이 어디에도 없는 값, 한 번도 진짜였던 적 없는 칸, 다수결·파일 수정 시각으로 골랐을 때의 결과,
변경 1건당 고칠 자리 수를 센다. 그다음 무작위 카드 8장 x 시드 5,000개로 q(손으로 옮길 때 성공할 확률)를 바꿔 가며 잰다.

## 실행
```bash
OMP_NUM_THREADS=2 uv run python productivity/ch01_source_of_truth/experiments.py   # results/ch01.json, data/*.json, checks/ch01.csv 갱신
uv run pytest -q productivity/ch01_source_of_truth
```
표준 라이브러리와 NumPy만 쓴다. 일반 모형(값 하나를 k곳에, 변경 m번)은 식, 순수 파이썬 몬테카를로, NumPy 몬테카를로 세 가지로
구현해 대조한다. PyTorch로 옮길 계산이 없어 쓰지 않았다. 노트북 CPU에서 10초 안에 끝난다.

## 파일
| 파일 | 역할 |
|---|---|
| `productivity_ch01_project.py` | 값·자리·카드 8장, 원본 자리와 잇는 방법, 결정 규칙 3줄, 원본 등록부 15항목 |
| `productivity_ch01_sim.py` | 복사·등록부 실행, 어긋난 칸 세기, 나중에 고르는 규칙 3개, 무작위 카드, 일반 모형, 등록부 점검 |
| `productivity_ch01_np.py` | 일반 모형의 NumPy 구현 |
| `experiments.py` | E0~E7 |
| `specs/sot_register.md` | 독자가 채울 원본 등록부와 점검표 |
| `checks/ch01.csv` | 독자 기록 양식(예시 행은 합성 모형 결과) |
| `data/` | `cards.json`(카드 8장과 초기값), `register.json`(등록부) |

## 한계
- 카드 8장은 "받은 사람이 자기 자리만 고친다"는 규칙을 일부러 지킨 교육용 시나리오다. 실제 팀은 몇 곳은 맞춰 준다.
  그래서 무작위 카드와 q를 바꿔 가며 다시 쟀다.
- q^(k-1) 모형은 자리마다 독립으로 놓친다는 단순화다. 함께 놓치는 경우(correlated)를 따로 쟀다.
- 원본 자리가 막히는 경우(권한, 장애)는 다루지 않는다. 권한 규칙은 6장에서 다룬다.

확인일 2026-10-10.
