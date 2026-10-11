# 10장. 내보내고 띄우기: 어댑터 합치기, GGUF, Ollama·vLLM (`ch10_export_serve`)

웹 챕터: `math-ai-matrix/tracks/finetune/10-export-serve.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 우리 말투로 쓰는 모델 만들기)의 마지막 장. 8장에서 4비트판 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` 위에 학습한 LoRA 어댑터(모든 선형층, r = 16)를 어떻게 내보내고 띄울지를 GPU·네트워크 없이 계산한다. 예시 모델 `Qwen/Qwen2.5-7B-Instruct`(Apache-2.0)는 추천이 아니라 예시다.

1. 손계산: W' = W0 + (α/r)BA. 7장 예(α = 16, r = 8, 배율 2)로 합친 표 [[7, −2], [12, −3]]와 옆길 계산이 모두 (12, 21)인지 본다.
2. 저장 크기: 어댑터 3개를 따로 두기(15.23GB + 3 × 80.7MB) vs 합친 모델 3벌.
3. 4비트로 학습한 어댑터 합치기 장난감: NF4(6장, 블록 64)로 얼린 8×64 행렬 위의 변화 sBA를 (A) 따로 두기, (B) 4비트를 되돌린 W에 bf16으로 합치기, (C) 원래 bf16 W에 합치기, (D) 합친 뒤 다시 NF4로 담기로 비교한다. 변화 크기 rho는 가정값이라 0.1%~10%로 훑는다.
4. GGUF 크기: ggml 블록 바이트와 llama.cpp b11552의 텐서별 형식 규칙(`use_more_bits`)으로 Qwen2.5-7B-Instruct의 F16·Q8_0·Q6_K·Q5_K_M·Q5_0·Q4_K_M·Q4_0·Q3_K_M·Q2_K 크기를 계산하고 Qwen 공식 GGUF 파일 크기와 대조한다(차이 448바이트 이하).
5. 띄우기: KV 캐시 토큰당 57,344바이트, 내 컴퓨터(GGUF + 문맥 4,096토큰), vLLM 메모리 예산(L4 24GB·T4 16GB, gpu_memory_utilization 0.92), 메모리 대역폭으로 본 토큰/초 상한, AWS 온디맨드 가격으로 본 한 달 비용.

## 실행
```bash
uv run python finetune/ch10_export_serve/experiments.py   # results/ch10.json 갱신
uv run pytest -q finetune/ch10_export_serve
```
표준 라이브러리만 쓴다(`fractions`, `struct`, `math`, `random` 고정 시드 10). 1GB = 10^9바이트.

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch10_export.py` | 합치기 손계산, bf16 반올림(가장 가까운 짝수·버림), NF4, 합치기 장난감, GGUF 형식 규칙과 크기, KV 캐시, vLLM 예산, 속도 상한, 비용, Modelfile·모델 카드 읽기 |
| `experiments.py` | E0~E7 |
| `data/specs_2026-10-11.json` | PEFT 0.21.2, transformers 5.19.0, Unsloth, huggingface_hub 2.2.0, llama.cpp b11552, Ollama 0.40.2, vLLM 0.31.0, NVIDIA L4·T4, AWS 가격(2026-10-09 발행 파일)의 값과 출처 주소 |
| `data/gguf_sizes_2026-10-11.json` | Qwen 공식 GGUF 저장소의 파일 크기(바이트) |
| `data/scenario.json` | 1장에서 이어받은 월 3,000건과 이 장의 가정값(답변 평균 200토큰, 어댑터 3개, 장난감 설정) |
| `data/Modelfile.example`, `data/model_card_example.md` | 한 줄씩 읽는 예시. 테스트가 형식을 확인한다 |
| `data/qwen2.5-7b-instruct_config.json`, `data/qwen2.5-1.5b-instruct_config.json` | 8장 실습에서 복사한 config.json |
| `results/ch10.json` | 실험 결과(웹 챕터 숫자의 출처) |
| `checks/ch10.csv` | 독자 기록 양식. 합친 파일 크기, GGUF 크기, 토큰/초를 계산값 옆에 적는다(예시 행은 계산값) |
| `colab/ch10_merge_gguf_serve.ipynb` | 오너가 돌릴 노트북(출력 칸 비움). 어댑터 변화 크기 재기 → 합치기 두 가지 → 모델 카드·비공개 Hub → GGUF → llama-server → Ollama → vLLM 여러 어댑터 |

## 가정
- 어댑터를 따로 둘 때 저장은 7장의 bf16 어댑터 크기(80.7MB)를 쓴다. 어댑터 3개(문의 답변, 상품 설명, 리뷰 답글)는 가정이다.
- 합치기 장난감의 변화 크기 rho, 행렬 크기(8×64), 랭크 2, 시드 10은 이 장에서 정한 값이다. 실제 rho는 노트북 4절에서 잰다.
- GGUF 메타데이터 몫(5,953,888바이트)은 Qwen의 F16 GGUF에서 텐서 몫을 빼서 잰 값을 다른 형식에도 그대로 쓴다. 남는 128~448바이트는 조각 파일 머리말과 정렬 차이로 본다.
- 토큰/초는 "토큰 하나마다 가중치를 한 번 다 읽는다"는 상한이다. 계산 시간, KV 캐시 읽기, 프롬프트 처리 시간은 넣지 않았다.
- 한 달 비용은 답변 평균 200토큰(가정), 한 달 730시간(가정), AWS us-east-1 Linux 온디맨드 인스턴스 가격(vCPU·메모리 포함)이다.
- vLLM 예산은 가중치와 어댑터 파일 크기만 빼고 남은 자리를 모두 KV 캐시로 본 상한이다. 활성값, CUDA 그래프, 어댑터 칸을 미리 잡는 몫은 빼지 않았다.

## 한계
- 실제 합치기, 변환, 서빙은 이 폴더에서 돌리지 않았다. 노트북을 돌려 `checks/ch10.csv`에 적어야 한다.
- 합치기 장난감은 합성 가중치다. 실제 어댑터의 변화가 어디에 몰려 있는지, 답의 품질이 얼마나 바뀌는지는 9장 방식으로 검증 세트에서 재야 한다.
- 양자화 형식별 품질(혼란도, KL 발산)은 재지 않았다.
- Unsloth `save_pretrained_merged`가 우리 설정에서 4비트를 되돌린 가중치에 합치는지, 16비트 원본을 내려받아 합치는지는 확인하지 못했다(save.py에 두 길이 모두 있다).

확인일 2026-10-11.
