# 1장 밴딧과 탐험

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [1장 밴딧과 탐험](https://math-ai-matrix-pi.vercel.app/tracks/rl/01-bandit.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch01_bandit
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 1장 비교 실험. `cd math-ai-matrix-labs && uv run python rl/ch01_bandit/experiments.py` |
| `rl_ch01_bandit_np.py` | 강화학습 1장 NumPy 최소 구현: 베르누이 밴딧과 네 가지 고르기 규칙. |
| `rl_ch01_bandit_torch.py` | 강화학습 1장 PyTorch 대응 구현: UCB1과 Thompson 선택을 텐서(float64)로. |
| `viz.py` | results/ch01.json과 logs/ch01_two_arm.jsonl을 터미널에 막대로 그린다(외부 패키지 없음). |
<!-- sync:end -->
