# 6장 TD·Sarsa·Q-러닝

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [6장 TD·Sarsa·Q-러닝](https://math-ai-matrix-pi.vercel.app/tracks/rl/06-td-sarsa-qlearning.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch06_td_sarsa_qlearning
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 6장 실험. `cd math-ai-matrix-labs && uv run python rl/ch06_td_sarsa_qlearning/experiments.py` |
| `rl_ch06_envs.py` | 강화학습 6장용 환경. 에이전트는 step()으로만 경험하고 전이확률 표를 보지 않는다. |
| `rl_ch06_td_np.py` | 강화학습 6장 NumPy 최소 구현: 끝까지 기다리지 않고 "보상 + 다음 칸의 지금 추정값"으로 고친다. |
| `rl_ch06_td_torch.py` | 강화학습 6장 PyTorch 대응 구현. |
| `viz.py` | results/ch06.json과 logs/ch06_transitions.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
