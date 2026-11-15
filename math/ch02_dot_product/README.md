# 2장 · 가중치를 곱한다는 게 무슨 말이에요?

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [2장 가중치를 곱한다는 게 무슨 말이에요?](https://math-ai-matrix-pi.vercel.app/tracks/math/02-dot-product.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch02_dot_product
```

| 파일 | 하는 일 |
|---|---|
| `dot_product_np.py` | 2장: 내적을 가중합으로 계산하는 NumPy 최소 구현. |
| `dot_product_torch.py` | 2장: 같은 내적을 PyTorch로 대조한다. |
| `experiments.py` | 웹 2장에 넣는 숫자를 재현해 results/ch02.json으로 저장한다. |
<!-- sync:end -->

동아리 행사 게시물의 특징 두 개에 가중치를 주고 점수 하나를 만들어요.

```bash
uv run python math/ch02_dot_product/experiments.py
uv run pytest -q math/ch02_dot_product
```

기대 결과는 손계산·NumPy·PyTorch가 모두 `10`이고, 두 게시물의 점수는 `[10, 11]`이에요.
