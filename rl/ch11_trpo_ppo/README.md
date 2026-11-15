# 11장 TRPO·PPO

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [11장 TRPO·PPO](https://math-ai-matrix-pi.vercel.app/tracks/rl/11-trpo-ppo.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch11_trpo_ppo
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 11장 실험. `cd math-ai-matrix-labs && uv run python rl/ch11_trpo_ppo/experiments.py` |
| `rl_ch11_cartpole.py` | 강화학습 11장 cart-pole 환경(NumPy, 여러 판을 한꺼번에). |
| `rl_ch11_ppo_np.py` | 강화학습 11장 NumPy 최소 구현: 바닐라 정책경사, TRPO, PPO(클리핑)와 시드 통계. |
| `rl_ch11_ppo_torch.py` | 강화학습 11장 PyTorch 대응 구현. float64. |
| `viz.py` | results/ch11.json과 logs/ch11_updates.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
