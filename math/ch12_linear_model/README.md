# 12장 · 선형 모델

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [12장 예측에서 가중치 갱신까지 한 번에 이어 보기](https://math-ai-matrix-pi.vercel.app/tracks/math/12-linear-model.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch12_linear_model
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 12장 · 선형 모델: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 12장 · 선형 모델: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch12_linear_model/experiment.py
uv run python math/ch12_linear_model/experiment_torch.py
uv run pytest -q math/ch12_linear_model
```
