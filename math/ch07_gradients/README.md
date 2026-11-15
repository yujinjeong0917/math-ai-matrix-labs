# 7장 · 그래디언트

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [7장 가중치가 여러 개면 어느 쪽으로 고칠까](https://math-ai-matrix-pi.vercel.app/tracks/math/07-gradients.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch07_gradients
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 7장 · 그래디언트: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 7장 · 그래디언트: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch07_gradients/experiment.py
uv run python math/ch07_gradients/experiment_torch.py
uv run pytest -q math/ch07_gradients
```
