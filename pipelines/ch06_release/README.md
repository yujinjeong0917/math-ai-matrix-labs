# 6장 단계적 배포

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [6장 단계적 배포](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/06-release.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch06_release
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 6장 비교 실험. `uv run python pipelines/ch06_release/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스 불필요). |
| `pipelines_ch06_card.py` | 6장 모델 카드: 배포 승인 전에 사람이 읽는 한 장짜리 설명서를 만든다. |
| `pipelines_ch06_data.py` | 6장 데이터: 4장(pipelines/ch04_container/pipelines_ch04_data.py)의 고객 이탈 생성기를 복사하고 열 하나를 더했다. |
| `pipelines_ch06_jenkinsfile.py` | 6장 Jenkinsfile 검사: 서버 없이 단계 순서와 승인 규칙만 읽는다. |
| `pipelines_ch06_model_np.py` | 6장 NumPy 모델: 4장 pipelines_ch04_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다. |
| `pipelines_ch06_model_torch.py` | 6장 PyTorch 대응 구현: 5장 pipelines_ch05_model_torch.py(4장에서 복사한 것)를 다시 복사했다. |
| `pipelines_ch06_release_gate.py` | 6장 배포 판정: 지금까지 모인 숫자로 '올린다 / 더 본다 / 되돌린다'를 정한다. |
| `pipelines_ch06_router.py` | 6장 라우터: 요청을 v1과 v2 중 어디로 보낼지 정한다. |
| `pipelines_ch06_schema.py` | 6장 되돌리기와 공유 데이터: 앱은 되돌렸는데 데이터는 이미 바뀌었으면? |
| `pipelines_ch06_simulate.py` | 6장 배포 시뮬레이터: 같은 요청 흐름에 배포 방식만 바꿔 대 본다. |
<!-- sync:end -->
