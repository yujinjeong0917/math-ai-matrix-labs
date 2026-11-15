# 3장 결정계수 R²

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [3장 결정계수 R²](https://math-ai-matrix-pi.vercel.app/tracks/eval/03-r-squared.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch03_r_squared
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch03_cases.py` | 모델 평가 3장의 데이터. Anscombe 네 쌍은 원 논문 표 그대로, 나머지는 합성 데이터. |
| `eval_ch03_metrics.py` | 모델 평가 3장. 결정계수 R², 제곱상관 r², 잔차 진단의 NumPy 최소 구현. |
| `eval_ch03_torch.py` | 모델 평가 3장의 PyTorch 대응 구현. |
| `experiments.py` | 모델 평가 3장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch03_r_squared/experiments.py` 로 실행한다. |
<!-- sync:end -->
