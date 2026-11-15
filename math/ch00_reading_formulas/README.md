# 0장 · 식을 읽는 약속

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [0장 식에서 어디부터 읽어야 할까](https://math-ai-matrix-pi.vercel.app/tracks/math/00-reading-formulas.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch00_reading_formulas
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 0장 · 식을 읽는 약속: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 0장 · 식을 읽는 약속: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch00_reading_formulas/experiment.py
uv run python math/ch00_reading_formulas/experiment_torch.py
uv run pytest -q math/ch00_reading_formulas
```
