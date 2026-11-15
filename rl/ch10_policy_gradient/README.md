# 10장 정책경사·actor-critic

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [10장 정책경사·actor-critic](https://math-ai-matrix-pi.vercel.app/tracks/rl/10-policy-gradient.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch10_policy_gradient
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 10장 실험. `cd math-ai-matrix-labs && uv run python rl/ch10_policy_gradient/experiments.py` |
| `rl_ch10_cartpole.py` | NumPy로 쓴 cart-pole(수레 위 막대 세우기) 동역학. 외부 패키지 없음. |
| `rl_ch10_corridor.py` | 강화학습 10장의 짧은 복도. 행동 효과가 가운데 칸에서 뒤집히는 3칸 복도. |
| `rl_ch10_pg_np.py` | 강화학습 10장 NumPy 최소 구현: 정책경사(REINFORCE), 베이스라인, one-step actor-critic. |
| `rl_ch10_pg_torch.py` | 강화학습 10장 PyTorch 대응 구현. float64. |
| `viz.py` | results/ch10.json과 logs/ch10_episodes.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
