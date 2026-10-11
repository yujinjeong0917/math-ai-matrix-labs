---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_relation: adapter
library_name: peft
language:
- ko
tags:
- lora
- customer-support
---

# 문의 답변 초안 어댑터 (예시)

- 기본 모델: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0). 학습은 4비트판 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` 위에서 했어요.
- 학습 방법: QLoRA, 모든 선형층, r = 16, lora_alpha = 16, 3에폭 (8장 설정).
- 데이터 출처: 우리 팀이 쓴 지난 문의·답변 600쌍(가상 데이터, 비공개). 고객 이름·연락처·주문번호는 학습 전에 지웠어요. Hub에 올린 데이터셋이 아니라서 `datasets` 칸은 비워 뒀어요.
- 평가: 검증 세트 60쌍, 9장 채점표 기준. 숫자는 직접 돌린 뒤 채워요.
- 쓰면 안 되는 곳: 환불 기한·배송비 같은 바뀌는 정책은 학습한 값을 믿지 말고 검색으로 붙여요(1장).
