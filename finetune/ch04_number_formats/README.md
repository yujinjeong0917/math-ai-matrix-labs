# 4장. fp32, fp16, bf16: 숫자 형식이 학습을 망가뜨리는 순간 (`ch04_number_formats`)

웹 챕터: `math-ai-matrix/tracks/finetune/04-number-formats.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 우리 말투로 쓰는 작은 모델을 만든다)에서 학습 설정의 `fp16`·`bf16` 칸이 무엇을 바꾸는지, GPU 없이 표준 라이브러리로 재현한다.

1. 숫자를 부호·지수·가수 칸으로 나누고(비트 분해기), 칸 수만으로 최댓값 65,504, 가장 작은 비정규수 2^-24, eps를 계산한다.
2. 넘침(300 × 300 = 90,000 → inf, exp(12) → inf, softmax가 NaN)과 사라짐(기울기 1e-8 → 0)을 재현한다.
3. loss scaling을 장난감 손실 ½(wx − y)²에 손으로 적용하고, GradScaler 규칙(넘치면 건너뛰고 절반, 2000번 무사하면 두 배)을 흉내 낸다.
4. bf16은 범위는 넓지만 간격이 거칠어서, 1에 1e-4를 1,000번 더해도 1로 남는 문제와 fp32 주 가중치를 확인한다.

## 실행
```bash
uv run python finetune/ch04_number_formats/experiments.py   # results/ch04.json 갱신
uv run pytest -q finetune/ch04_number_formats
```
E0~E7은 표준 라이브러리만 쓴다(`struct`, `fractions`, `random`). E8과 대조 테스트 두 개는 numpy·torch가 있으면 돌고, 없으면 건너뛴다.

## 흉내 규칙
- fp16: `struct`의 `'e'`(IEEE 754 binary16). `struct`는 너무 큰 값에서 OverflowError를 내지만 GPU 연산은 inf를 내므로 ±inf로 바꾼다.
- fp32: `struct`의 `'f'`. 같은 방식으로 넘치면 ±inf.
- bf16: fp32 비트의 위 16비트. 반올림은 PyTorch `round_to_nearest_even`과 같은 식(NaN → 0x7FC0, 아니면 `u + ((u >> 16) & 1) + 0x7FFF`를 16비트 민다). 비교용 자르기(`mode="trunc"`)도 있다.
- 연산 한 번은 파이썬 float(fp64)로 하고 목표 형식으로 한 번 반올림한다. fp64의 가수가 충분히 길어 덧셈·곱셈·나눗셈 한 번은 목표 형식에서 직접 계산한 것과 같은 값이 된다. fp32 주 가중치도 매 덧셈 뒤에 `'f'`로 반올림한다.

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch04_formats.py` | 형식 한계값(spec), 반올림(f32·f16·bf16), 비트 분해(decompose) |
| `ft_ch04_scaling.py` | 넘침 예, 장난감 기울기와 loss scaling, 가상 기울기 개수 세기, DynamicScaler(GradScaler 흉내), 주 가중치 |
| `experiments.py` | E0~E8 |
| `data/sources_2026-10-11.json` | 사실마다 출처 URL과 확인일 |
| `checks/ch04.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 한계
- E4의 기울기 10,000개는 1e-10 ~ 1e-1 로그 균등으로 뽑은 가상 값이다. 실제 모델의 기울기 분포가 아니다.
- 기울기를 "계산한 뒤 fp16으로 한 번 저장"한다고 본다. 실제 GPU 커널은 중간 누적을 fp32로 하는 등 커널마다 다르다.
- DynamicScaler는 GradScaler의 규칙(건너뛰기, backoff, growth)만 옮겼다. 옵티마이저 여러 개, 기울기 자르기와의 순서(unscale_) 같은 부분은 없다.

확인일 2026-10-11.
