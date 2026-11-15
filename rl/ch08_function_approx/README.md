# 8장 함수 근사·죽음의 삼각형

<!-- sync:start -->
> [밑바닥부터 만드는 강화학습](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/rl) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [8장 함수 근사·죽음의 삼각형](https://math-ai-matrix-pi.vercel.app/tracks/rl/08-function-approx.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q rl/ch08_function_approx
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 강화학습 8장 실험. `cd math-ai-matrix-labs && uv run python rl/ch08_function_approx/experiments.py` |
| `rl_ch08_np.py` | 강화학습 8장 NumPy 최소 구현: 선형 함수 근사와 반경사 TD(0), 그리고 그것이 발산하는 조건. |
| `rl_ch08_torch.py` | 강화학습 8장 PyTorch 대응 구현. 같은 스윕을 autograd로 계산해 NumPy와 대조한다(float64). |
| `viz.py` | results/ch08.json과 logs/ch08_offpolicy.jsonl을 터미널에 그린다(외부 패키지 없음). |
<!-- sync:end -->
