# 11장 디코딩

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [11장 디코딩](https://math-ai-matrix-pi.vercel.app/tracks/llm/11-decoding.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch11_decoding
```

| 파일 | 하는 일 |
|---|---|
| `decoding_np.py` | 11장 NumPy 최소 구현: 디코딩(다음 토큰 고르기) 방법과 반복 지표. |
| `decoding_torch.py` | 11장 PyTorch 대응 구현. |
| `experiments.py` | 11장 비교 실험. `uv run python llm/ch11_decoding/experiments.py` 로 실행한다. |
<!-- sync:end -->
