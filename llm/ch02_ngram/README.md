# 2장 n-gram과 평활

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [2장 n-gram과 평활](https://math-ai-matrix-pi.vercel.app/tracks/llm/02-ngram.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch02_ngram
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 2장 비교 실험. `uv run python llm/ch02_ngram/experiments.py` 로 실행한다. |
| `ngram_np.py` | 2장 NumPy·표준 라이브러리 최소 구현: n-gram 카운트, MLE, 덧셈 평활, 보간, 보간형 Kneser-Ney. |
| `ngram_torch.py` | 2장 PyTorch 대응 구현. |
<!-- sync:end -->
