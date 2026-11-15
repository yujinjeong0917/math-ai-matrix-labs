# 10장 작은 GPT 학습

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [10장 작은 GPT 학습](https://math-ai-matrix-pi.vercel.app/tracks/llm/10-train-gpt.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch10_train_gpt
```

| 파일 | 하는 일 |
|---|---|
| `ch10_data.py` | 10장 데이터: 작은 한국어 문법 말뭉치. |
| `ch10_np.py` | 10장 NumPy 최소 구현: 옵티마이저와 학습 루프. |
| `ch10_torch.py` | 10장 PyTorch 대응 구현. |
| `experiments.py` | 10장 비교 실험. `uv run python llm/ch10_train_gpt/experiments.py` 로 실행한다. |
<!-- sync:end -->
