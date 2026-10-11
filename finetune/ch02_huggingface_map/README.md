# 2장. Hugging Face 지도: Hub, transformers, datasets, PEFT, TRL은 각각 무엇을 하나 (`ch02_huggingface_map`)

웹 챕터: `math-ai-matrix/tracks/finetune/02-huggingface-map.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 쓰는 작은 모델을 만드는 이야기)에서, 학습에 쓸 공개 모델 저장소를 열어 파일을 한 줄씩 읽는다. 네트워크 없이 `data/`에 넣어 둔 실제 파일(커밋 해시로 고정)과 직접 만든 예시 파일만 읽는다.

1. 가중치 파일(safetensors) 네 개의 헤더를 `struct`로 읽어 텐서 339개, 숫자 7,615,616,512개, 15,231,233,024바이트를 세고 index.json·Hub API·파일 크기와 맞춘다.
2. 모델 카드(README.md) 머리말, config.json, generation_config.json, tokenizer_config.json을 읽고 기본(base) 모델과 대화용(instruct) 모델을 비교한다.
3. chat_template의 도구 없는 갈래를 표준 라이브러리로 다시 짜서 jinja2로 렌더링한 글자와 한 글자도 다르지 않은지 대조한다.
4. 학습 한 번의 단계마다 맡는 라이브러리와, PyPI 의존 관계의 층을 정리한다.

## 실행
```bash
uv run python finetune/ch02_huggingface_map/experiments.py   # results/ch02.json 갱신
uv run pytest -q finetune/ch02_huggingface_map
```

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch02_hfmap.py` | 카드 머리말 파서, config 읽기·비교, safetensors 헤더 읽기·쓰기, Qwen2.5 템플릿 렌더러, 의존 관계 층 |
| `experiments.py` | E0~E7 |
| `data/qwen2.5-7b/` | Qwen/Qwen2.5-7B(커밋 d1497293…)의 config.json, generation_config.json, tokenizer_config.json, model.safetensors.index.json, README 앞부분 발췌, 가중치 파일 네 개의 헤더만(값 없음) |
| `data/qwen2.5-7b-instruct/` | Qwen/Qwen2.5-7B-Instruct(커밋 a09a3545…)의 같은 설정 파일과 README 발췌 |
| `data/our_adapter_example/` | 직접 만든 예시: 우리 팀이 올릴 어댑터의 모델 카드와 adapter_config.json |
| `data/messages_example.json` | 직접 만든 예시 대화(문의 한 건과 답변) |
| `data/render_fixture_2026-10-11.json` | jinja2 3.1.6으로 렌더링한 글자와 tokenizers 0.23.3으로 한 번 센 토큰 수. 실습은 다시 세지 않고 읽기만 한다 |
| `data/sources_2026-10-11.json` | 출처 URL, 커밋 해시, 파일 크기, 라이브러리 버전·역할 인용, PyPI 의존 관계 |
| `results/ch02.json` | 실험 결과(웹 챕터 숫자의 출처) |
| `checks/ch02.csv` | 독자 기록 양식(예시 행은 이 장의 값) |

## 규칙과 가정
- 예로 든 모델은 추천이나 순위가 아니다. 설정 파일과 가중치 헤더가 공개돼 있어 숫자를 맞춰 볼 수 있어서 골랐고, 5장과 같은 모델이다.
- 카드 머리말 파서는 이 장의 파일에 나오는 `키: 값`과 `- 항목` 목록만 읽는다.
- 템플릿 렌더러는 `tools`가 없고 `tool_calls`·`tool` 메시지도 없는 갈래만 옮겼다. 다른 갈래는 `NotImplementedError`.
- 토큰 수는 라이브러리로 한 번 잰 값이다. tokenizer.json(약 7MB)은 넣지 않았다.
- 학습에서 손실을 매길 범위(55토큰)는 TRL v1.15.0의 `qwen2_5_training.jinja`가 `{% generation %}`으로 감싸는 범위('\n' + 답변 + '<|im_end|>\n')에서 시작하는 토큰을 센 값이다.

## 한계
- 가중치 값은 하나도 읽지 않는다. 헤더(이름, 형식, 모양, 위치)만 읽는다.
- 라이브러리를 실제로 불러 학습하지 않는다(8장에서 Colab으로 돌린다).

확인일 2026-10-11.
