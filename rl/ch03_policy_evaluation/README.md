# 3장 정책 평가

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [3장 정책 평가](https://math-ai-matrix-pi.vercel.app/tracks/rl/03-policy-evaluation.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch03_policy_evaluation
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 3장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch03_policy_evaluation/experiments.py` |
| `rl_ch03_eval_np.py` | 강화학습 3장 NumPy 최소 구현: 고정 정책의 가치를 구하는 세 가지 방법. |
| `rl_ch03_eval_torch.py` | 강화학습 3장 PyTorch 대응 구현. 같은 P^pi, r^pi에서 NumPy와 float64로 atol=1e-10 안에서 같아야 한다. |
| `rl_ch03_grid.py` | 강화학습 3장용 격자. 고정 정책 하나를 넣어 P^pi (S x S)와 r^pi (S)를 바로 만든다. |
| `viz.py` | results/ch03.json을 터미널에 그린다(외부 패키지 없음). `uv run python rl/ch03_policy_evaluation/viz.py` |
<!-- sync:end -->
