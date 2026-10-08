# math-ai-matrix-labs

[수학 × AI 매트릭스](https://math-ai-matrix-pi.vercel.app) 학습 트랙의 실습 코드예요. 장마다 원리가 드러나는 NumPy 최소 구현, 같은 입력에서 수치를 대조하는 PyTorch 구현, 비교 실험, 테스트가 한 폴더에 있어요. 웹 챕터의 실험 숫자는 모두 각 폴더의 `results/*.json`에서 그대로 옮겼어요.

## 실행

```bash
uv sync                      # Python 3.12, numpy 2.3.3, torch 2.8.0 (uv.lock으로 고정)
uv run pytest -q             # 모든 장의 테스트
uv run python llm/ch07_self_attention/experiments.py   # 예: 7장 비교 실험, results/ch07.json 갱신
```

## 장 목록

| 폴더 | 장 | 웹 |
|---|---|---|
| `llm/ch01_bigram/` | 밑바닥부터 만드는 LLM 1장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/01-bigram.html) |
| `llm/ch02_ngram/` | 밑바닥부터 만드는 LLM 2장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/02-ngram.html) |
| `llm/ch03_tokenizer/` | 밑바닥부터 만드는 LLM 3장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/03-tokenizer.html) |
| `llm/ch04_embedding_mlp/` | 밑바닥부터 만드는 LLM 4장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/04-embedding-mlp.html) |
| `llm/ch05_rnn_lstm/` | 밑바닥부터 만드는 LLM 5장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/05-rnn-lstm.html) |
| `llm/ch06_attention_seq2seq/` | 밑바닥부터 만드는 LLM 6장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/06-attention-seq2seq.html) |
| `llm/ch07_self_attention/` | 밑바닥부터 만드는 LLM 7장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/07-self-attention.html) |
| `llm/ch08_positional/` | 밑바닥부터 만드는 LLM 8장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/08-positional.html) |
| `llm/ch09_residual_norm/` | 밑바닥부터 만드는 LLM 9장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/09-residual-norm.html) |
| `llm/ch10_train_gpt/` | 밑바닥부터 만드는 LLM 10장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/10-train-gpt.html) |
| `llm/ch11_decoding/` | 밑바닥부터 만드는 LLM 11장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/11-decoding.html) |
| `llm/ch12_kv_cache/` | 밑바닥부터 만드는 LLM 12장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/12-kv-cache.html) |
| `llm/ch13_scaling/` | 밑바닥부터 만드는 LLM 13장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/llm/13-scaling.html) |
| `rl/ch00_map/` | 밑바닥부터 만드는 강화학습 0장 | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/rl/00-map.html) |

모델 평가, 알고리즘, ML 파이프라인, 생산성 도구 트랙은 작성 중이에요. 전체 목록은 [학습 트랙](https://math-ai-matrix-pi.vercel.app/tracks/)에서 볼 수 있어요.

## 폴더마다 같은 구성

- `*_np.py`: NumPy 최소 구현. 원리가 드러나는 가장 작은 코드예요.
- `*_torch.py`: PyTorch 대응 구현과 학습용 모델(필요한 장만).
- `experiments.py`: 비교 실험. 결과는 `results/*.json`에 저장해요.
- `test_*.py`: NumPy와 PyTorch의 수치 일치, 수식 성질 검증.

데이터는 모두 코드 안에서 만드는 합성 데이터예요. 실측 시간은 기기마다 다르지만, 계산량과 테스트 결과는 같아야 해요.
