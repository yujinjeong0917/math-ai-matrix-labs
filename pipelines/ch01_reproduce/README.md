# 1장 재현

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [1장 재현](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/01-reproduce.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch01_reproduce
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 1장 비교 실험. `uv run python pipelines/ch01_reproduce/experiments.py` 로 실행한다. |
| `pipelines_ch01_data.py` | 1장 데이터: 고객 이탈을 흉내 낸 합성 표 데이터. |
| `pipelines_ch01_model_np.py` | 1장 모델: 무작위가 세 군데 들어가는 작은 학습 스크립트(NumPy). |
| `pipelines_ch01_model_torch.py` | 1장 PyTorch 대응 구현: 같은 (데이터, 초기값, 배치 순서)를 받아 같은 가중치를 내는지 본다. |
| `pipelines_ch01_repro.py` | 1장 재현 도구: 시드 고정, 환경 기록, 부동소수 합의 순서 실험. |
<!-- sync:end -->
