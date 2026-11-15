# 5장 몬테카를로

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [5장 몬테카를로](https://math-ai-matrix-pi.vercel.app/tracks/rl/05-monte-carlo.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch05_monte_carlo
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 5장 실험. `cd math-ai-matrix-labs && uv run python rl/ch05_monte_carlo/experiments.py` |
| `rl_ch05_grid.py` | 강화학습 5장용 환경. 이 장의 에이전트는 P와 R을 보지 않고 step()으로만 경험한다. |
| `rl_ch05_mc_np.py` | 강화학습 5장 NumPy 최소 구현: 전이확률 없이 에피소드의 리턴을 평균 내서 가치를 구한다. |
| `rl_ch05_mc_torch.py` | 강화학습 5장 PyTorch 대응 구현. 같은 에피소드 묶음에서 리턴, 첫 방문 MC 평균, 중요도 비율과 |
| `viz.py` | results/ch05.json과 logs/ch05_episodes.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
