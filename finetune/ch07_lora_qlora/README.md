# 7장. LoRA와 QLoRA: 전체 대신 작은 행렬만 학습하기 (`ch07_lora_qlora`)

웹 챕터: `math-ai-matrix/tracks/finetune/07-lora-qlora.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 우리 말투로 쓰는 모델 만들기)에서, 5장의 예시 모델 `Qwen/Qwen2.5-7B`(추천이 아니라 예시)에 LoRA를 붙일 때 학습하는 숫자가 몇 개이고 메모리가 얼마인지, QLoRA로 4비트에 얼리면 5장의 "4.28GB + α"가 실제로 얼마가 되는지 GPU·네트워크 없이 계산한다.

1. 손계산: 4×4 변화량(16개)을 B(4×1)·A(1×4) 8개로 적고, ΔWx와 B(Ax)가 같은지 본다.
2. 선형층 하나(d_in → d_out)에 랭크 r을 붙이면 r(d_in + d_out)개. q,v(PEFT의 qwen2 기본값) / q,k,v,o / MLP / all-linear별, r = 1~64별로 센다.
3. 배율 α/r과 rsLoRA α/√r. ±1 무작위 칸으로 (BA)x 크기가 √r로 자라는 것을 본다.
4. 특이값(3, 1, 0.1)을 미리 아는 4×4 행렬로 낮은 랭크 근사 오차(에카르트-영), 거듭제곱법, B = 0에서 출발한 기울기 하강을 대조한다.
5. 합치기 (W0 + sBA)x = W0x + sB(Ax)를 분수로 확인한다.
6. LoRA(얼린 bf16)와 QLoRA 합계. QLoRA는 5장 미리 보기 3.81GB에서 출발하는 사다리로: 4비트가 안 되는 층(임베딩, lm_head, 편향, 정규화) 되돌리기, 6장 블록 상수, LoRA 몫, 5장 활성값.
7. 페이징 옵티마이저 장난감: 길이를 늘릴 때 16GB에 들어가는 한계가 옵티마이저 상태만큼만 늘어남을 본다.

## 실행
```bash
uv run python finetune/ch07_lora_qlora/experiments.py   # results/ch07.json 갱신
uv run pytest -q finetune/ch07_lora_qlora
```
표준 라이브러리만 쓴다(`fractions`, `struct`, `random` 고정 시드 7·11, `math`). 1GB = 10^9바이트.

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch07_lora.py` | LoRA 파라미터 수, 기본 모델 나누기, NF4 비트, QLoRA 사다리, 배율, 페이징 장난감, 작은 행렬 장난감 |
| `experiments.py` | E0~E8 |
| `data/specs_2026-10-11.json` | 논문 인용, PEFT 0.21.2·transformers 5.19.0·bitsandbytes 0.50.2 기본값과 소스 주소, 5·6장에서 이어받은 값 |
| `data/qwen2.5-7b_config.json` | 5장 실습의 config.json 원본을 복사(Hub, 2026-10-11) |
| `results/ch07.json` | 실험 결과(웹 챕터 숫자의 출처) |
| `checks/ch07.csv` | 독자 기록 양식. `print_trainable_parameters()`와 `torch.cuda.max_memory_allocated()`를 계산값과 나란히 적는다(예시 행은 모형 값) |

## 가정
- 대표 설정: 선형층 NF4 + 두 번 양자화(6장 4.126953비트), 나머지 bf16, LoRA all-linear r = 16, PEFT 기본(`autocast_adapter_dtype=True`)대로 어댑터 fp32 + AdamW(숫자 하나당 4 + 4 + 8 = 16바이트), 길이 1,024, 묶음 1, 체크포인팅.
- 논문처럼 어댑터를 bf16으로 두면 2 + 2 + 8 = 12바이트로 센다.
- 4비트가 되는 것은 층 안의 `nn.Linear` 가중치뿐이다(transformers 5.19.0 소스). 임베딩(`nn.Embedding`)과 `lm_head`, 편향, 정규화는 원래 형식으로 남는다.
- `prepare_model_for_kbit_training`을 쓰면 남는 층이 fp32로 올라간다고 보고 따로 계산했다(PEFT 0.21.2 소스).
- 페이징 장난감은 "옵티마이저 상태만 CPU로 내보낼 수 있다"는 단순 모형이다. 실제 페이징은 GPU가 모자랄 때만 일어나고 속도 비용은 상황에 따라 다르다.

## 한계
- 활성값은 5장의 논문 식 추정(GPT 모양 가정)을 그대로 썼다. LoRA 경로가 더 저장하는 작은 활성값(Ax 등)과 드롭아웃 마스크, 출력 칸(logits, fp32로 묶음 1에 약 0.623GB), CUDA 몫, 캐시 할당기 여유분은 넣지 않았다.
- LoRA가 품질을 얼마나 지키는지(랭크·붙이는 곳에 따른 성능)는 계산하지 않았다. 논문 표를 인용만 했다. 실제 결과는 8·9장에서 잰다.
- 작은 행렬 장난감은 특이값을 미리 정한 4×4 행렬이다. 실제 학습에서 변화량 ΔW가 낮은 랭크라는 것은 LoRA 논문의 가설·측정이고 이 실습이 보이는 것이 아니다.

확인일 2026-10-11.
