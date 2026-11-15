# 2장 숨은 기술부채

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [2장 숨은 기술부채](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/02-hidden-debt.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch02_hidden_debt
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 2장 비교 실험. `uv run python pipelines/ch02_hidden_debt/experiments.py` 로 실행한다. |
| `pipelines_ch02_checks.py` | 2장 검사: 데이터 계약(스키마), 학습·서빙 분포 비교, ML Test Score 계산. |
| `pipelines_ch02_data.py` | 2장 데이터: 고객 이탈을 흉내 낸 합성 표 데이터와, 데이터를 보내 주는 쪽(업스트림)의 버전별 출력. |
| `pipelines_ch02_model_np.py` | 2장 NumPy 최소 구현: 전처리(표준화 + 원-핫)와 로지스틱 회귀. |
| `pipelines_ch02_model_torch.py` | 2장 PyTorch 대응 구현: NumPy fit_logistic과 같은 입력에서 같은 가중치가 나와야 한다. |
<!-- sync:end -->
