# 5장. GPU 메모리 계산: 7B 모델에는 몇 GB가 필요할까 (`ch05_gpu_memory`)

웹 챕터: `math-ai-matrix/tracks/finetune/05-gpu-memory.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 우리 말투로 쓰는 모델 만들기)에서, 공개 가중치 모델 하나(예: `Qwen/Qwen2.5-7B`, Apache-2.0, 추천이 아니라 예시)를 학습할 때 GPU 메모리가 어디에 얼마나 드는지 GPU·네트워크 없이 계산한다.

1. 숫자 하나의 바이트 수를 `struct`로 잰다(fp32 `'f'` 4바이트, fp16 `'e'` 2바이트, bf16은 fp32 비트의 위쪽 16비트를 잘라 2바이트).
2. `config.json`으로 파라미터 수를 닫힌 식과 텐서 339개 합으로 세고, safetensors 값 7,615,616,512와 대조한다.
3. 학습 방식별 파라미터 하나당 바이트(ZeRO 논문 16, Hugging Face 문서 18, 8비트 Adam 12, 추론 2, 얼린 4비트 0.5)로 모델 상태 메모리를 계산한다.
4. 활성값을 Korthikanti 외(2022) 식 `sbh(34 + 5as/h)`로 추정하고, 논문의 항목별 내역 합과 대조한다. 체크포인팅을 앞뒤로 지나가는 흉내로 재고 식과 대조한다.
5. 기울기 누적이 큰 묶음 한 번과 같은 기울기를 내는지 분수로 확인한다.

## 실행
```bash
uv run python finetune/ch05_gpu_memory/experiments.py   # results/ch05.json 갱신
uv run pytest -q finetune/ch05_gpu_memory
```
표준 라이브러리만 쓴다(`struct`, `fractions`). 1GB = 10^9바이트.

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch05_memory.py` | 바이트 수, 파라미터 세기, 학습 방식별 바이트, 활성값 식과 내역, 체크포인팅 흉내, 기울기 누적 장난감 |
| `experiments.py` | E0~E6 |
| `data/specs_2026-10-11.json` | 사양표. 항목마다 출처 URL과 확인일. 값이 바뀌면 날짜가 다른 새 파일을 만든다 |
| `data/qwen2.5-7b_config.json` | Hugging Face Hub에서 내려받은 config.json 원본(2026-10-11) |
| `data/scenario.json` | 관통 프로젝트 설정 가정값(길이 1,024토큰, 묶음 4, 실제 묶음 16) |
| `checks/ch05.csv` | 독자 기록 양식. 계산값과 `torch.cuda.max_memory_allocated()`를 나란히 적는다(예시 행은 모형 값) |

## 한계
- 활성값 식은 GPT 모양 블록(MLP 폭 4h, GeLU, 드롭아웃, 16비트 활성값)을 가정한 논문 식이다. Qwen2.5처럼 SwiGLU·RMSNorm·드롭아웃 없음·더 넓은 MLP(18,944 = 약 5.3h)인 모델에는 근사일 뿐이고, 실제 값은 측정해야 한다.
- 출력 칸(logits, 길이 × 어휘 수), CUDA 컨텍스트, PyTorch 캐시 할당기의 여유분, 순간적으로 생기는 임시 텐서는 합계에 넣지 않았다. 출력 칸만 fp32로 잡아도 묶음 1에 약 0.623GB다.
- LoRA·QLoRA 줄은 기본 모델 몫만 센 미리 보기다. 4비트 보정 상수는 6장, LoRA 행렬과 그 옵티마이저 상태는 7장에서 더한다.
- 체크포인팅 시간 비용 4/3배는 "뒤로 가기가 앞으로 가기의 2배"라는 가정에서 나온 값이다.
- 무료 Colab GPU 종류는 Colab FAQ에 적혀 있지 않다([확인 필요]). T4(16GB)는 예로만 썼다.

확인일 2026-10-11.
