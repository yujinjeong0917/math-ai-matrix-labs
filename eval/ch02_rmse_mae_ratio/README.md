# 2장 RMSE/MAE 비율

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [2장 RMSE/MAE 비율](https://math-ai-matrix-pi.vercel.app/tracks/eval/02-rmse-mae-ratio.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch02_rmse_mae_ratio
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch02_cases.py` | 모델 평가 2장의 합성 데이터. 1장처럼 정답 y는 0으로 두고 예측값 = 오차로 쓴다. |
| `eval_ch02_metrics.py` | 모델 평가 2장. RMSE/MAE 비율과 그 기준값·범위의 NumPy 최소 구현. |
| `eval_ch02_torch.py` | 모델 평가 2장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 2장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch02_rmse_mae_ratio/experiments.py` 로 실행한다. |
<!-- sync:end -->
