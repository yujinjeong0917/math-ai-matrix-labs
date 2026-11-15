# 4장 임베딩과 MLP

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [4장 임베딩과 MLP](https://math-ai-matrix-pi.vercel.app/tracks/llm/04-embedding-mlp.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch04_embedding_mlp
```

| 파일 | 하는 일 |
|---|---|
| `ch04_grammar.py` | 4장 합성 문법. 일반화 효과가 크게 드러나도록 일부러 설계한 작은 말뭉치다. |
| `ch04_ngram.py` | 보간 트라이그램 기준선. 2장 설계도(docs/tracks/llm.md)의 보간 n-gram 식을 이 장에 필요한 만큼만 직접 구현했다(2장 코드와 공유하지 않는다). |
| `experiments.py` | 4장 비교 실험. `uv run python llm/ch04_embedding_mlp/experiments.py` 로 실행한다. |
| `mlp_lm_np.py` | 4장 NumPy 최소 구현: Bengio 등(2003)의 신경 확률 언어모델(식 1)과 손으로 쓴 역전파. |
| `mlp_lm_torch.py` | 4장 PyTorch 대응 구현. mlp_lm_np.py와 같은 식을 nn.Module로 옮긴다. |
<!-- sync:end -->
