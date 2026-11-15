# 12장 KV cache

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [12장 KV cache](https://math-ai-matrix-pi.vercel.app/tracks/llm/12-kv-cache.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch12_kv_cache
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 12장 비교 실험. `uv run python llm/ch12_kv_cache/experiments.py` 로 실행한다. |
| `kvcache_np.py` | 12장 NumPy 최소 구현: 작은 GPT의 생성 루프를 캐시 없이 / KV cache로 돌리기. |
| `kvcache_torch.py` | 12장 PyTorch 대응 구현. |
<!-- sync:end -->
