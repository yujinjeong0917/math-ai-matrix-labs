---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B
library_name: peft
pipeline_tag: text-generation
language:
- ko
datasets:
- our-shop/inquiry-reply-pairs
tags:
- lora
- sft
- trl
---

# inquiry-draft-lora

우리 쇼핑몰 고객 문의에 보낼 답변 초안을 우리 말투로 쓰도록 학습한 LoRA 어댑터예요.
기본 모델 가중치는 들어 있지 않아서, 쓰려면 base_model을 함께 불러야 해요.

## 쓰면 안 되는 곳

- 환불 기한, 배송비 같은 정책 숫자의 근거로 쓰지 마세요. 정책은 검색으로 붙인 문서를 따르게 해요.
- 사람이 읽고 고치기 전에 고객에게 바로 보내지 마세요.

## 학습

- 데이터: 지난 1년 동안 사람이 쓴 문의·답변 600쌍(개인정보 지움)
- 기본 모델 revision: d149729398750b98c0af14eb82c78cfe92750796
- 대화 템플릿: 기본 모델 tokenizer_config.json의 chat_template, eos_token은 <|im_end|>

<!-- 직접 만든 예시예요. 실제로 올린 저장소가 아니고, our-shop/inquiry-reply-pairs도 가상 이름이에요. -->
