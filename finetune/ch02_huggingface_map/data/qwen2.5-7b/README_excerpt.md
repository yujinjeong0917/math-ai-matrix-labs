---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# Qwen2.5-7B

[...]

**This repo contains the base 7B Qwen2.5 model**, which has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 7.61B
- Number of Paramaters (Non-Embedding): 6.53B
- Number of Layers: 28
- Number of Attention Heads (GQA): 28 for Q and 4 for KV
- Context Length: 131,072 tokens

[...]

**We do not recommend using base language models for conversations.** Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., on this model.

<!-- 발췌(excerpt): https://huggingface.co/Qwen/Qwen2.5-7B/blob/d149729398750b98c0af14eb82c78cfe92750796/README.md 의 앞부분과 일부 줄. [...]는 생략한 곳. 확인일 2026-10-11 -->
