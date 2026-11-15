# 1장 MAE와 RMSE

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [1장 MAE와 RMSE](https://math-ai-matrix-pi.vercel.app/tracks/eval/01-mae-rmse.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch01_mae_rmse
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch01_cases.py` | 모델 평가 1장의 합성 데이터. 정답 y는 0으로 두고 예측값 = 오차로 쓴다(오차만 중요하므로). |
| `eval_ch01_metrics.py` | 모델 평가 1장. MAE·MSE·RMSE와 '상수 하나로 예측할 때 가장 좋은 값'의 NumPy 최소 구현. |
| `eval_ch01_torch.py` | 모델 평가 1장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 1장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch01_mae_rmse/experiments.py` 로 실행한다. |
<!-- sync:end -->
