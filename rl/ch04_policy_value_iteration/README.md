# 4장 정책 반복·가치 반복

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [4장 정책 반복·가치 반복](https://math-ai-matrix-pi.vercel.app/tracks/rl/04-policy-value-iteration.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch04_policy_value_iteration
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 4장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch04_policy_value_iteration/experiments.py` |
| `rl_ch04_grid.py` | 강화학습 4장용 격자. P[S,A,S], R[S,A,S]를 통째로 만든다. |
| `rl_ch04_pivi_np.py` | 강화학습 4장 NumPy 최소 구현: 정책 반복, 가치 반복, 수정 정책 반복. |
| `rl_ch04_pivi_torch.py` | 강화학습 4장 PyTorch 대응 구현. 같은 P, R에서 NumPy와 최종 정책이 같고 가치가 float64로 atol=1e-10 안이어야 한다. |
| `viz.py` | results/ch04.json을 터미널에 그린다(외부 패키지 없음). `uv run python rl/ch04_policy_value_iteration/viz.py` |
<!-- sync:end -->
