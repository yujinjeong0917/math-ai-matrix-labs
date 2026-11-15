# 3장 실험 기록

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [3장 실험 기록](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/03-tracking.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch03_tracking
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 3장 비교 실험. `uv run python pipelines/ch03_tracking/experiments.py` 로 실행한다. |
| `pipelines_ch03_data.py` | 3장 데이터: 2장(pipelines/ch02_hidden_debt/pipelines_ch02_data.py)의 고객 이탈 생성기를 복사해 줄였다. |
| `pipelines_ch03_hashing.py` | 3장 데이터 해시: 표의 '내용'이 같은지를 짧은 문자열 하나로 판정한다. |
| `pipelines_ch03_model_np.py` | 3장 NumPy 모델: 2장 pipelines_ch02_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다. |
| `pipelines_ch03_model_torch.py` | 3장 PyTorch 대응 구현: 2장 pipelines_ch02_model_torch.py를 복사했다. |
| `pipelines_ch03_tracking.py` | 3장 실험 기록: 파일 하나(JSON Lines)와 결과 파일 폴더로 만든 가장 작은 실험 추적기. |
<!-- sync:end -->
