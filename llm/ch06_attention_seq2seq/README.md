# 6장 seq2seq 어텐션

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [6장 seq2seq 어텐션](https://math-ai-matrix-pi.vercel.app/tracks/llm/06-attention-seq2seq.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch06_attention_seq2seq
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 6장 비교 실험. `uv run python llm/ch06_attention_seq2seq/experiments.py` 로 실행한다. |
| `seq2seq_np.py` | 6장 NumPy 최소 구현: 인코더-디코더에 붙이는 어텐션. |
| `seq2seq_torch.py` | 6장 PyTorch 대응 구현. |
<!-- sync:end -->
