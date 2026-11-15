# 9장 잔차와 정규화

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [9장 잔차와 정규화](https://math-ai-matrix-pi.vercel.app/tracks/llm/09-residual-norm.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch09_residual_norm
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 9장 비교 실험. `uv run python llm/ch09_residual_norm/experiments.py` 로 실행한다. |
| `resnorm_np.py` | 9장 NumPy 최소 구현: 정규화(LayerNorm, BatchNorm), 멀티헤드 어텐션, MLP, 그리고 블록 하나. |
| `resnorm_torch.py` | 9장 PyTorch 대응 구현. |
<!-- sync:end -->
