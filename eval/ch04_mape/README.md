# 4장 MAPE

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [4장 MAPE](https://math-ai-matrix-pi.vercel.app/tracks/eval/04-mape.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch04_mape
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch04_cases.py` | 모델 평가 4장의 데이터. 모두 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다. |
| `eval_ch04_metrics.py` | 모델 평가 4장. 백분율 오차(APE)와 MAPE의 NumPy 최소 구현. |
| `eval_ch04_torch.py` | 모델 평가 4장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 4장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch04_mape/experiments.py` 로 실행한다. |
<!-- sync:end -->
