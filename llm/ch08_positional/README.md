# 8장 위치 정보

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [8장 위치 정보](https://math-ai-matrix-pi.vercel.app/tracks/llm/08-positional.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch08_positional
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 8장 비교 실험. `uv run python llm/ch08_positional/experiments.py` 로 실행한다. |
| `positional_np.py` | 8장 NumPy 최소 구현: self-attention에 위치 정보를 넣는 네 가지 방법. |
| `positional_torch.py` | 8장 PyTorch 대응 구현. |
<!-- sync:end -->
