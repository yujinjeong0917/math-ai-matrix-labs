# 5장 MASE

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [5장 MASE](https://math-ai-matrix-pi.vercel.app/tracks/eval/05-mase.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch05_mase
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch05_cases.py` | 모델 평가 5장의 데이터. 손으로 고른 작은 표이거나 고정 시드의 합성 데이터다. |
| `eval_ch05_metrics.py` | 모델 평가 5장. 척도 오차(scaled error)와 MASE의 NumPy 최소 구현. |
| `eval_ch05_torch.py` | 모델 평가 5장의 PyTorch 대응 구현. scikit-learn에는 MASE가 없어서(설계도) 손 계산 fixture와 함께 |
| `experiments.py` | 모델 평가 5장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch05_mase/experiments.py` 로 실행한다. |
<!-- sync:end -->
