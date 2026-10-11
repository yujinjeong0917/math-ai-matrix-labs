# 8장. Unsloth로 무료 Colab에서 학습 돌리기 (`ch08_unsloth_colab`)

웹 챕터: `math-ai-matrix/tracks/finetune/08-unsloth-colab.html`

관통 프로젝트(작은 팀이 고객 문의 답변 초안을 우리 말투로 쓰는 모델 만들기)에서, 16GB GPU 한 장(예: NVIDIA T4)으로 Qwen2.5-7B-Instruct를 QLoRA로 학습할 때 메모리·스텝 수·에폭 시간을 GPU 없이 어림하고, 오너가 직접 돌릴 Colab 노트북을 함께 둔다.

1. `config.json`으로 파라미터 수를 세고(7B·3B·1.5B), Hub의 safetensors 합계와 대조한다.
2. Unsloth가 미리 4bit로 줄여 올린 파일(`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`)의 크기를 "양자화하는 선형층은 NF4 + 두 번 양자화, 나머지는 bf16"으로 계산해 실제 파일 크기와 대조한다(차이 0.007%).
3. LoRA 학습 파라미터 수를 랭크별로 닫힌 식과 텐서 392개 합으로 센다(랭크 16에서 40,370,176개).
4. 학습 메모리를 4bit 가중치 + LoRA 몫 + 활성값(5장 식, 체크포인팅) + 출력 칸(logits)으로 어림하고, 묶음 × 길이 × 랭크 격자와 모델·방식별 표를 만든다.
5. 스텝 수(Trainer 방식), 채우기(padding)와 묶어 담기(packing)의 토큰 수, 가정한 활용률(MFU)별 에폭 시간을 계산한다.

## 실행
```bash
uv run python finetune/ch08_unsloth_colab/experiments.py   # results/ch08.json 갱신
uv run pytest -q finetune/ch08_unsloth_colab
```
표준 라이브러리만 쓴다(`json`, `math`, `fractions`). 1GB = 10^9바이트. 무작위 값을 쓰지 않는다.

## Colab 노트북
`colab/ch08_unsloth_qlora_t4.ipynb`를 Colab에서 열고 런타임을 GPU로 바꿔 위에서부터 차례로 돌린다. **작성자는 이 노트북을 돌려 보지 않았다**(출력 칸이 비어 있다). 설치 칸과 API 모양은 Unsloth 공식 노트북(2026-10-11)을 따랐고, 버전이 바뀌면 공식 노트북의 설치 칸으로 바꿔 넣는다. 14번 칸이 찍는 한 줄을 `checks/ch08.csv`에 붙여 넣으면 계산값과 잰 값을 나란히 볼 수 있다.

## 파일
| 파일 | 역할 |
|---|---|
| `ft_ch08_unsloth.py` | 파라미터 세기, 4bit 파일 크기, LoRA 수, 메모리 어림, 스텝 수, 채우기·묶어 담기, 시간 어림 |
| `experiments.py` | E0~E6 |
| `data/specs_2026-10-11.json` | 사양표. Unsloth 버전·지원 범위·회사 주장 인용, TRL 기본값, Colab FAQ, T4 사양, 모델 파일 크기. 항목마다 URL과 확인일 |
| `data/scenario.json` | 관통 프로젝트 설정 가정값(학습 540쌍, 길이 1,024, 묶음 2 × 누적 8, 랭크 16, 3에폭, 토큰 길이 분포, 가정 활용률) |
| `data/qwen2.5-*_config.json` | Hugging Face Hub에서 내려받은 config.json 원본(2026-10-11) |
| `colab/ch08_unsloth_qlora_t4.ipynb` | 오너가 돌릴 노트북(출력 칸 비움) |
| `checks/ch08.csv` | 독자 기록 양식. 계산값과 잰 값을 나란히 적는다(예시 행은 모형 값) |

## 한계
- 활성값은 5장의 GPT 모양 논문 식에 체크포인팅을 적용한 추정이다. Unsloth의 체크포인팅은 층 입력을 CPU 메모리로 옮긴다고 회사가 밝혀서, 실제 GPU 몫은 더 작을 수 있다. 회사 주장(30% 절감)은 활성값 몫에만 곱한 별도 줄로 두었다.
- 출력 칸(logits)은 fp32로 한 번에 다 만든다고 본 상한이다. 나눠 계산하는 구현이면 더 작다.
- LoRA 가중치를 fp32로 둔다고 가정했다(가중치 4 + 기울기 4 + 8비트 Adam 2 = 10바이트). 랭크 16에서 0.40GB라 가정이 바뀌어도 결론(T4에 들어간다)은 그대로다.
- CUDA 컨텍스트, 캐시 할당기 여유분, 양자화를 풀 때 잠깐 생기는 텐서는 합계에 넣지 않았다.
- 에폭 시간은 FLOPs 어림(토큰당 6N)을 T4 최고 성능 65 TFLOPS × 가정한 활용률(0.1, 0.2, 0.3)로 나눈 값이다. 활용률은 측정값이 아니고, 노트북의 `train_runtime`으로 거꾸로 구해 바꿔 넣는다.
- 토큰 길이 분포와 학습·검증 나누기(540/60)는 가정이다. 3장 데이터로 바꿔 넣는다.
- 무료 Colab GPU 종류는 Google FAQ에 적혀 있지 않다([확인 필요]). T4는 Unsloth 문서의 표현과 5장의 예를 따랐다.

확인일 2026-10-11.
