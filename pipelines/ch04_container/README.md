# 4장 환경 옮기기

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [4장 환경 옮기기](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/04-container.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch04_container
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 4장 비교 실험. `uv run python pipelines/ch04_container/experiments.py` 로 실행한다(네트워크와 uv 필요). |
| `pipelines_ch04_data.py` | 4장 데이터: 3장(pipelines/ch03_tracking/pipelines_ch03_data.py)의 고객 이탈 생성기를 그대로 복사했다. |
| `pipelines_ch04_envs.py` | 4장 환경 도구: 환경을 '설치 방법'이 아니라 '설치된 결과의 목록'으로 다루고, 그 목록의 지문을 만든다. |
| `pipelines_ch04_image.py` | 4장 최소 구현: 내용으로 이름을 정하는(content-addressed) 이미지 빌더를 파이썬 표준 라이브러리로 흉내 낸다. |
| `pipelines_ch04_job.py` | 4장 작업 스크립트: 어느 환경에서 돌려도 같은 일을 하고, 결과를 JSON 한 줄로 찍는다. |
| `pipelines_ch04_model_np.py` | 4장 NumPy 모델: 3장 pipelines_ch03_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다. |
| `pipelines_ch04_model_torch.py` | 4장 PyTorch 대응 구현: 3장 pipelines_ch03_model_torch.py를 복사했다. |
<!-- sync:end -->
