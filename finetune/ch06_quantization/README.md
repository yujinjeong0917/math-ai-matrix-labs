# 6장. 최소한의 양자화: int8, 4bit, NF4가 숫자를 줄이는 법 (`ch06_quantization`)

웹 챕터: `math-ai-matrix/tracks/finetune/06-quantization.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 쓰는 작은 모델을 만드는 이야기)에서, 기본 모델의 가중치를 8비트·4비트로 줄였다가 되돌릴 때 얼마나 틀리는지를 GPU·네트워크 없이 재현한다.

1. 숫자 4개 `[0.8, -2.0, 0.1, 0.4]`를 absmax int8, 가르치기용 균일 4bit(-7~7), bitsandbytes fp4, NF4로 줄였다 되돌린다.
2. 튀는 값 하나가 한 덩어리 전체를 망가뜨리는 것과, 블록으로 나누면 그 블록만 망가지는 것을 본다.
3. 합성 가중치 65,536개(평균 0, 표준편차 0.02, 시드 6)로 방식·블록 크기별 오차, 칸 사용률, 두 번 양자화(단순 모형), 7B 가중치 크기를 잰다.
4. NF4 표를 bitsandbytes 방식으로 다시 만들어 라이브러리의 표와 대조한다.

## 실행
```bash
uv run python finetune/ch06_quantization/experiments.py   # results/ch06.json 갱신
uv run pytest -q finetune/ch06_quantization
```
표준 라이브러리만 쓴다(`statistics.NormalDist`로 정규분포 분위수, `struct 'f'`로 fp32 저장).

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch06_quant.py` | 줄이기·되돌리기(균일, 표), NF4 표 다시 만들기, 두 번 양자화 단순 모형, 비트·크기 계산, 오차 재기 |
| `experiments.py` | E0~E8 |
| `data/specs_2026-10-11.json` | NF4·fp4·int4 표(bitsandbytes 0.50.2 소스), transformers `BitsAndBytesConfig` 기본값, 논문 숫자. 출처 URL과 확인일 |
| `results/ch06.json` | 실험 결과(웹 챕터 숫자의 출처) |
| `checks/ch06.csv` | 독자 기록 양식(예시 행은 모형 값) |

## 규칙과 가정
- 정수 코드로 바꿀 때는 파이썬 `round()`를 쓴다. 딱 가운데(.5)는 짝수 쪽으로 간다. 표에서 고를 때는 가장 가까운 값, 거리가 같으면 앞 값.
- 블록마다 absmax를 fp32로 저장한다고 본다.
- 합성 가중치와 튀는 값(약 1,000개마다 하나를 ±0.5로) 설정은 정한 값이다. 실제 모델 가중치가 아니다.
- 두 번 양자화는 단순 모형이다. bitsandbytes는 2단계에서 8비트 동적 표를 쓰는데, 여기서는 평균을 뺀 뒤 균일 int8(±127)로 줄였다.
- int8은 LLM.int8() 논문의 absmax 방식(×127)을 블록 단위로 쓴 모형이다. bitsandbytes의 `quantize_blockwise`(8비트 동적 표)나 LLM.int8()의 벡터별 양자화·튀는 값 분리와는 다르다.

## 한계
- 되돌린 오차만 잰다. 그 오차가 모델의 답이나 손실을 얼마나 바꾸는지는 실제 모델로 재야 한다(8·9장).
- 크기 계산은 양자화하는 가중치만 셌다. 양자화하지 않고 두는 층, 활성값, LoRA, 옵티마이저 상태는 빠져 있다.

확인일 2026-10-11.
