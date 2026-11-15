# 8장 분류 평가

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [8장 분류 평가](https://math-ai-matrix-pi.vercel.app/tracks/eval/08-classification.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch08_classification
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch08_cases.py` | 모델 평가 8장의 합성 데이터. 모두 고정 시드다. |
| `eval_ch08_metrics.py` | 모델 평가 8장. 이진 분류 지표의 NumPy 최소 구현. |
| `eval_ch08_torch.py` | 모델 평가 8장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 8장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch08_classification/experiments.py` 로 실행한다. |
<!-- sync:end -->
