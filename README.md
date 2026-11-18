# math-ai-matrix-labs

[수학 × AI 매트릭스](https://math-ai-matrix-pi.vercel.app) 학습 트랙의 실습 코드예요. 웹 챕터에서 계산한 숫자를 직접 다시 만들어 볼 수 있게, 장마다 폴더 하나에 코드·테스트·README를 두었어요. 데이터는 모두 코드 안에서 만드는 합성 데이터라 같은 시드에서는 같은 결과가 나와요.

## 어디서 시작하나요

1. 웹 챕터를 읽다가 "실습 코드" 링크를 누르면 그 장 폴더로 와요. 폴더 README의 `웹 챕터` 링크로 되돌아갈 수 있어요.
2. 트랙 전체를 훑으려면 아래 표에서 트랙 폴더를 열어요. 트랙 폴더 README가 장 목록과 웹 링크를 모아 둬요.

## 트랙 한눈에

| 트랙 폴더 | 이름 | 장 수 | 필요한 것 | 웹 |
|---|---|---|---|---|
| [`llm/`](llm/) | 밑바닥부터 만드는 LLM | 13 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/llm/01-bigram.html) |
| [`rl/`](rl/) | 밑바닥부터 만드는 강화학습 | 13 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/rl/00-map.html) |
| [`eval/`](eval/) | 모델 평가 | 9 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/eval/01-mae-rmse.html) |
| [`algorithms/`](algorithms/) | 그림으로 익히는 알고리즘 | 10 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/algorithms/01-complexity.html) |
| [`pipelines/`](pipelines/) | ML 파이프라인 구축 | 9 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/01-reproduce.html) |
| [`productivity/`](productivity/) | 협업 도구가 어긋날 때 | 6 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/productivity/01-source-of-truth.html) |
| [`tools/`](tools/) | AI 도구 실전 | 8 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/tools/01-models.html) |
| [`webapp/`](webapp/) | 1인 개발 앱 출시 실전 | 13 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/webapp/01-everything-visible.html) |
| [`finetune/`](finetune/) | 내 데이터로 LLM 파인튜닝하기 | 10 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/finetune/01-when-finetune.html) |
| [`math/`](math/) | AI를 읽기 위한 수학 | 13 | `uv` 환경 필요(numpy·torch·pytest) | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/math/00-reading-formulas.html) |
| [`aws/`](aws/) | AWS 배포·운영 | 15 | 1/15장만 `uv` 환경 필요 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/aws/00-local-to-service.html) |
| [`or/`](or/) | 경영과학 | 16 | 1/16장만 `uv` 환경 필요 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/or/00-forecast-vs-decision.html) |
| [`causal/`](causal/) | 인과추론 | 16 | 1/16장만 `uv` 환경 필요 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/causal/00-prediction-intervention-counterfactual.html) |
| [`agent-build/`](agent-build/) | 에이전트 만들기 | 12 | 1/12장만 `uv` 환경 필요 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/agent-build/00-refund-workflow.html) |
| [`waiting-ux/`](waiting-ux/) | 기다림을 설계하다 | 6 | 표준 라이브러리만 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/waiting-ux/00-perceived-vs-actual.html) |
| [`reference-study/`](reference-study/) | 디자이너의 시선으로 레퍼런스 읽기 | 6 | 표준 라이브러리만 | [웹 트랙](https://math-ai-matrix-pi.vercel.app/tracks/reference-study/00-first-impression.html) |

## 실행

```bash
# `uv` 환경(numpy·torch·pytest)이 필요한 장이 있는 트랙: llm, rl, eval, algorithms, pipelines, productivity, tools, webapp, finetune, math, aws, or, causal, agent-build
uv sync                      # Python 3.12, 버전은 uv.lock으로 고정
uv run python scripts/run_all_tests.py   # 모든 장 테스트를 장 폴더마다 따로 실행

# 표준 라이브러리만 쓰는 트랙: waiting-ux, reference-study (설치 없이 장 폴더에서)
cd aws/ch11_retry_duplicates && python3 -m unittest -q
python3 scripts/run_all_tests.py aws or causal agent-build   # 이 트랙들만 한 번에
```

## 폴더마다 같은 구성

- `README.md`: 웹 챕터 링크, 실행 명령, 파일 안내(머리 블록은 `scripts/sync_chapter_readmes.py`가 갱신해요).
- `*_np.py` / `*_torch.py`: NumPy 최소 구현과 PyTorch 대응 구현(해당 장만).
- `experiments.py`: 비교 실험. 결과는 `results/*.json`에 저장해요.
- `test_*.py`: 수치 일치와 수식 성질 검증.

## 이 저장소가 확인하지 못하는 것

- 실측 시간은 기기마다 달라요. 계산량과 테스트 결과는 같아야 해요.
- AWS·외부 서비스 장(`aws`, `pipelines` 일부)은 계정 없이 도는 계산·검증 코드예요. 실제 서비스 호출, 요금, 한도는 측정하지 않아요.

## 이 README를 다시 만드는 법

```bash
python3 scripts/update_readme.py          # 루트·트랙 README
python3 scripts/sync_chapter_readmes.py   # 장 README 머리 블록
```
