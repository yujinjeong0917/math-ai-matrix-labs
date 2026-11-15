# 10장 · PCA와 고유벡터

<!-- sync:start -->
> [AI를 읽기 위한 수학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/math) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [10장 점구름을 한 줄로 줄이면 무엇을 남길까](https://math-ai-matrix-pi.vercel.app/tracks/math/10-pca-eigenvectors.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q math/ch10_pca_eigenvectors
```

| 파일 | 하는 일 |
|---|---|
| `experiment.py` | 10장 · PCA와 고유벡터: 손계산과 같은 숫자를 만드는 결정적 실험. |
| `experiment_torch.py` | 10장 · PCA와 고유벡터: 같은 입력을 PyTorch로 대조하는 실험. |
<!-- sync:end -->

웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.

```bash
uv run python math/ch10_pca_eigenvectors/experiment.py
uv run python math/ch10_pca_eigenvectors/experiment_torch.py
uv run pytest -q math/ch10_pca_eigenvectors
```
