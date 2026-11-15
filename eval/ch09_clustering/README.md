# 9장 클러스터링 평가

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [9장 클러스터링 평가](https://math-ai-matrix-pi.vercel.app/tracks/eval/09-clustering.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch09_clustering
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch09_cases.py` | 모델 평가 9장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다. |
| `eval_ch09_metrics.py` | 모델 평가 9장. 클러스터링 평가 지표의 NumPy 최소 구현. |
| `eval_ch09_torch.py` | 모델 평가 9장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 9장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch09_clustering/experiments.py` 로 실행한다. |
<!-- sync:end -->
