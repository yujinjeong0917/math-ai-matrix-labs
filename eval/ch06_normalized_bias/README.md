# 6장 NMBE와 CV(RMSE)

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [6장 NMBE와 CV(RMSE)](https://math-ai-matrix-pi.vercel.app/tracks/eval/06-normalized-bias.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch06_normalized_bias
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch06_cases.py` | 모델 평가 6장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다. |
| `eval_ch06_metrics.py` | 모델 평가 6장. 크기를 정규화한 지표(NMAE, CV(RMSE))와 방향을 남기는 지표(MBE, NMBE)의 NumPy 최소 구현. |
| `eval_ch06_torch.py` | 모델 평가 6장의 PyTorch 대응 구현. scikit-learn에는 NMBE·CV(RMSE)·NMAE가 없고 이 저장소 의존성에도 없어서, |
| `experiments.py` | 모델 평가 6장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch06_normalized_bias/experiments.py` 로 실행한다. |
<!-- sync:end -->
