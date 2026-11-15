# 7장 n스텝·TD(λ)

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [7장 n스텝·TD(λ)](https://math-ai-matrix-pi.vercel.app/tracks/rl/07-nstep-td-lambda.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch07_nstep_td_lambda
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 7장 실험. `cd math-ai-matrix-labs && uv run python rl/ch07_nstep_td_lambda/experiments.py` |
| `rl_ch07_np.py` | 강화학습 7장 NumPy 최소 구현: n스텝 TD, λ-리턴, 적격 흔적 TD(λ). |
| `rl_ch07_torch.py` | 강화학습 7장 PyTorch 대응 구현. 같은 판에서 NumPy 결과와 수치를 대조한다(float64). |
| `rl_ch07_walk.py` | 강화학습 7장용 환경. 에이전트는 전이 기록만 보고 전이확률 표를 보지 않는다. |
| `viz.py` | results/ch07.json과 logs/ch07_traces.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
