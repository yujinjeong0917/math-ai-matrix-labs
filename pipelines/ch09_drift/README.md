# 9장 드리프트

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [9장 드리프트](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/09-drift.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch09_drift
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 9장 비교 실험. `uv run python pipelines/ch09_drift/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스·kfp 불필요). |
| `pipelines_ch09_data.py` | 9장 데이터: 8장(pipelines/ch08_lineage/pipelines_ch08_data.py)의 고객 이탈 생성기(source_version v1)와 |
| `pipelines_ch09_drift_np.py` | 9장 NumPy 최소 구현: 기준 분포와 지금 분포를 비교하는 두 검사. |
| `pipelines_ch09_drift_torch.py` | 9장 PyTorch 대응 구현: 같은 입력에서 MMD^2_u가 NumPy와 같은 값인지 본다(float64, torch.cdist). |
| `pipelines_ch09_jenkinsfile.py` | 9장 Jenkinsfile 검사. stages()는 7장 pipelines_ch07_jenkinsfile.py(6장에서 복사한 것)에서 복사했다. |
| `pipelines_ch09_loop.py` | 9장 감시·재학습·되돌리기 루프 시뮬레이션. Jenkins 서버와 KFP 클러스터는 띄우지 않는다. |
| `pipelines_ch09_model_np.py` | 9장 NumPy 모델: 6장 pipelines_ch06_model_np.py(4장에서 복사한 것)의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다. |
| `pipelines_ch09_release_gate.py` | 9장 배포 판정: 6장 pipelines_ch06_release_gate.py의 z_worse()와 decide()를 그대로 복사했다(구간 가드 부분은 뺐다). |
<!-- sync:end -->
