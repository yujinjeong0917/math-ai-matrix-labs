# 1장. 같은 질문인데 왜 모델마다 답과 값이 다를까 (`ch01_models`)

웹 챕터: `math-ai-matrix/tracks/tools/01-models.html`

관통 프로젝트(가상의 작은 팀이 만드는 "주간 고객 문의 요약 보고서")의 첫 단계다. Claude·GPT·Gemini·DeepSeek·Grok의 공식 문서를 2026-10-10에 읽고 옮긴 비교표로, 같은 일(한 번에 입력 32,150토큰, 출력 700토큰, 한 주 12번)을 맡길 때 한 달에 얼마가 드는지 네트워크·API 키 없이 계산한다.

1. 비교표(`data/prices_2026-10-10.json`): 문맥 창, 최대 출력, 입력·출력 단가, 가중치 공개 여부와 라이선스, 데이터 처리(학습 사용, 보관, 지역), 학습 기준일. 칸마다 출처 URL과 확인일.
2. 계산기(`tools_ch01_cost.py`): 긴 프롬프트 구간(Haiku 5.5 100,000 초과, OpenAI 272K 초과, Grok 200K 이상), DeepSeek 피크/비피크(한국 시각으로 바꿔 판정), 생각 토큰을 출력으로 세기.
3. 답이 갈리는 이유 장난감(`tools_ch01_answers.py`): 온도를 넣은 소프트맥스와 고정 시드 뽑기, 점수표가 다른 두 모델의 1등.
4. 팀 조건(문서에 "유료 API 입력을 학습에 쓰지 않는다"가 있는가, 문맥 창, 한 달 예산 10달러)으로 거르기. 순위를 매기지 않는다.

## 실행
```bash
uv run python tools/ch01_models/experiments.py   # results/ch01.json 갱신
uv run pytest -q tools/ch01_models
```
표준 라이브러리만 쓴다(fractions, math, random). 실제 API를 부르지 않는다.

## 파일
| 파일 | 역할 |
|---|---|
| `tools_ch01_cost.py` | 가격표·일 크기 읽기, 단가 규칙, 한 번·한 주·한 달 비용, 팀 조건 |
| `tools_ch01_answers.py` | 온도를 넣은 소프트맥스, 탐욕 선택, 고정 시드 뽑기 |
| `experiments.py` | E0~E8 |
| `data/prices_2026-10-10.json` | 비교표. 값이 바뀌면 날짜가 다른 새 파일을 만든다 |
| `data/workload.json` | 블록별 토큰 수(2장과 같은 가정값), 한 주 호출 수, 생각 토큰 가정값, 팀 조건 |
| `checks/ch01.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 가격을 바꿀 때
`data/prices_YYYY-MM-DD.json`을 새로 만들고 `tools_ch01_cost.PRICE_FILE`을 바꾼 뒤 실험과 테스트를 다시 돌린다. Gemini 3.8 Flash는 2027-01-01에 단가가 오른다고 공지되어 있다.

## 한계
- 토큰 수는 세지 않고 정한 가정값이다. 같은 글도 회사마다 토크나이저가 달라 토큰 수가 다르다.
- 답의 품질(한국어 요약이 얼마나 맞는지)은 계산하지 않는다. 팀이 문의 20건으로 직접 시험한다는 조건만 적었다.
- DeepSeek의 API 입력 학습 사용·보관 기간, DeepSeek·Grok의 생각 토큰 단가, Grok 4.7 최대 출력, Gemini 3.8 Flash·DeepSeek 학습 기준일은 문서에서 찾지 못했다 [확인 필요]. 계산은 생각 토큰을 출력 단가로 센다고 가정했다.
- DeepSeek 피크 판정은 중국 공휴일을 무시한다. OpenAI의 "272K"는 272,000토큰으로 읽었다.
- 장난감의 점수표는 직접 정한 값이고 실제 모델의 점수가 아니다.

확인일 2026-10-10.
