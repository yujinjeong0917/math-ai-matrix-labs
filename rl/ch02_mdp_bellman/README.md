# 2장 MDP와 벨만 방정식

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [2장 MDP와 벨만 방정식](https://math-ai-matrix-pi.vercel.app/tracks/rl/02-mdp-bellman.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch02_mdp_bellman
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 2장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch02_mdp_bellman/experiments.py` |
| `rl_ch02_gridworld.py` | 강화학습 2장용 Gridworld. 전이확률 P[S,A,S']와 보상 R[S,A,S']를 표로 만든다. |
| `rl_ch02_mdp_np.py` | 강화학습 2장 NumPy 최소 구현: 벨만 백업 두 개와 롤아웃. |
| `rl_ch02_mdp_torch.py` | 강화학습 2장 PyTorch 대응 구현. 같은 P, R, V에서 NumPy 백업과 float64로 atol=1e-12 안에서 같아야 한다. |
| `viz.py` | results/ch02.json과 logs/ch02_rollouts.jsonl을 터미널에 격자로 그린다(외부 패키지 없음). |
<!-- sync:end -->
