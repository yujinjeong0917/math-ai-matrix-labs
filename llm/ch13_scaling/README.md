# 13장 스케일링 법칙

<!-- sync:start -->
> [밑바닥부터 만드는 LLM](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/llm) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [13장 스케일링 법칙](https://math-ai-matrix-pi.vercel.app/tracks/llm/13-scaling.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q llm/ch13_scaling
```

| 파일 | 하는 일 |
|---|---|
| `ch13_data.py` | 13장 합성 말뭉치: 드물게 나오는 사실일수록 늦게 배우는 '열쇠-값' 언어. |
| `experiments.py` | 13장 비교 실험. `uv run python llm/ch13_scaling/experiments.py` 로 실행한다. |
| `scaling_np.py` | 13장 NumPy 최소 구현: 파라미터·FLOP 세기, 거듭제곱 법칙 피팅, 계산 예산 배분. |
| `scaling_torch.py` | 13장 PyTorch 대응 구현. |
<!-- sync:end -->
