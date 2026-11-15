# 7장 self-attention

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [7장 self-attention](https://math-ai-matrix-pi.vercel.app/tracks/llm/07-self-attention.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch07_self_attention
```

| 파일 | 하는 일 |
|---|---|
| `attention_np.py` | 7장 NumPy 최소 구현: 순환 신경망(RNN)의 한 걸음과 self-attention 한 층. |
| `attention_torch.py` | 7장 PyTorch 대응 구현. |
| `experiments.py` | 7장 비교 실험. `uv run python llm/ch07_self_attention/experiments.py` 로 실행한다. |
<!-- sync:end -->
