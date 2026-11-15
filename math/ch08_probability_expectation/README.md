# 8장 · 확률과 기댓값

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [8장 가능한 결과가 여러 개면 무엇을 예측할까](https://math-ai-matrix-pi.vercel.app/tracks/math/08-probability-expectation.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch08_probability_expectation
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 8장 · 확률과 기댓값: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 8장 · 확률과 기댓값: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch08_probability_expectation/experiment.py
uv run python math/ch08_probability_expectation/experiment_torch.py
uv run pytest -q math/ch08_probability_expectation
```
