# 3장 토크나이저

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [3장 토크나이저](https://math-ai-matrix-pi.vercel.app/tracks/llm/03-tokenizer.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch03_tokenizer
```

| 파일 | 하는 일 |
|---|---|
| `bpe_np.py` | 3장 최소 구현: 단어 단위 어휘와 BPE(byte pair encoding) 토크나이저. |
| `ch03_data.py` | 3장 합성 말뭉치. 외부 텍스트 없이 고정 시드로 만든다. |
| `ch03_ngram.py` | 토큰 단위가 다른 모델을 같은 잣대로 재는 도구: 보간 trigram + 글자당 비트(bpc). |
| `experiments.py` | 3장 비교 실험. `uv run python llm/ch03_tokenizer/experiments.py` 로 실행한다. |
| `sennrich_alg1.py` | 대조용 기준 구현: Sennrich, Haddow, Birch(2016) Algorithm 1. |
<!-- sync:end -->
