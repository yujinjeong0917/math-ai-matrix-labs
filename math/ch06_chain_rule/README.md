# 6장 · 연쇄법칙

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [6장 계산이 여러 단계를 지나면 변화를 어떻게 이을까](https://math-ai-matrix-pi.vercel.app/tracks/math/06-chain-rule.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch06_chain_rule
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 6장 · 연쇄법칙: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 6장 · 연쇄법칙: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch06_chain_rule/experiment.py
uv run python math/ch06_chain_rule/experiment_torch.py
uv run pytest -q math/ch06_chain_rule
```
