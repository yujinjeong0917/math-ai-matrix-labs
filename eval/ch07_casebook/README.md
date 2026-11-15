# 7장 판독 사례집

<!-- sync:start -->
> [모델 평가](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/eval) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [7장 판독 사례집](https://math-ai-matrix-pi.vercel.app/tracks/eval/07-casebook.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q eval/ch07_casebook
```

| 파일 | 하는 일 |
|---|---|
| `eval_ch07_casebook.py` | 모델 평가 7장의 판독 사례집. 지표 표 만들기, 순위 뒤집힘 재기, 짝지은 부트스트랩, 판독 절차, 리포트. |
| `eval_ch07_cases.py` | 모델 평가 7장의 판독 사례. 모두 고정 시드의 합성 데이터다. |
| `eval_ch07_metrics.py` | 모델 평가 7장. 판독 사례집이 쓰는 회귀 지표 여섯 개와 진단값의 NumPy 최소 구현. |
| `eval_ch07_torch.py` | 모델 평가 7장의 PyTorch 대응 구현. scikit-learn은 이 저장소 의존성에 없어서(6장과 같은 사정), |
| `experiments.py` | 모델 평가 7장 실험. `OMP_NUM_THREADS=2 uv run python eval/ch07_casebook/experiments.py` 로 실행한다. |
<!-- sync:end -->
