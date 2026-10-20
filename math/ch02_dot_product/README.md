# 2장 · 가중치를 곱한다는 게 무슨 말이에요?

동아리 행사 게시물의 특징 두 개에 가중치를 주고 점수 하나를 만들어요.

```bash
uv run python math/ch02_dot_product/experiments.py
uv run pytest -q math/ch02_dot_product
```

기대 결과는 손계산·NumPy·PyTorch가 모두 `10`이고, 두 게시물의 점수는 `[10, 11]`이에요.
