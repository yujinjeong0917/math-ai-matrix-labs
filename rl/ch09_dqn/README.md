# 9장 DQN

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [9장 DQN](https://math-ai-matrix-pi.vercel.app/tracks/rl/09-dqn.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch09_dqn
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 9장 실험. `cd math-ai-matrix-labs && uv run python rl/ch09_dqn/experiments.py` |
| `rl_ch09_cartpole.py` | NumPy로 쓴 cart-pole(수레 위 막대 세우기) 동역학. 외부 패키지 없음. |
| `rl_ch09_dqn_np.py` | 강화학습 9장 NumPy 최소 구현: 신경망 Q-러닝(DQN)과 두 안정화 장치, 그리고 최댓값의 치우침. |
| `rl_ch09_dqn_torch.py` | 강화학습 9장 PyTorch 대응 구현. float64. |
| `viz.py` | results/ch09.json과 logs/ch09_transitions.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
