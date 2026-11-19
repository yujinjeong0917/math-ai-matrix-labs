# 14장 전문가 혼합과 비슷한 갈래들

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [14장 전문가 혼합과 비슷한 갈래들](https://math-ai-matrix-pi.vercel.app/tracks/llm/14-moe-routing.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch14_moe
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 14장 실험. `python experiments.py` 로 실행하면 results/results.json 을 쓰고 표를 출력한다. |
| `moe_np.py` | 14장 NumPy 최소 구현: 전문가 혼합(MoE) 라우팅, 부하분산 손실, 파라미터·KV 세기. |
| `moe_torch.py` | 14장 PyTorch 대응 구현: 작은 MoE 층, 밀집 기준선, 합성 데이터, 학습 루프. |
<!-- sync:end -->
