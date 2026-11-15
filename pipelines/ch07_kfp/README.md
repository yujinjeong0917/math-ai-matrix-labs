# 7장 Kubeflow Pipelines

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [7장 Kubeflow Pipelines](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/07-kfp.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch07_kfp
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 7장 비교 실험. `uv run python pipelines/ch07_kfp/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스·kfp 불필요). |
| `kfp_local_check.py` | 7장 실제 KFP 확인(선택). 이 저장소의 uv 환경에는 kfp가 없으므로 따로 만든 환경에서 돌린다. |
| `kfp_pipeline.py` | 7장 실제 KFP v2 파이프라인(kfp 2.17.0 SDK 문법). 이 실습 저장소의 테스트는 이 파일을 import하지 않는다. |
| `pipelines_ch07_dag.py` | 7장 최소 구현: 입력·출력이 선언된 컴포넌트, 그래프(DAG) 컴파일, 위상 정렬 실행기. 표준 라이브러리만 쓴다. |
| `pipelines_ch07_data.py` | 7장 데이터: 6장(pipelines/ch06_release/pipelines_ch06_data.py)의 고객 이탈 생성기를 복사했다. |
| `pipelines_ch07_glue.py` | 7장 실패 예제를 돌리는 도우미: run_all.sh를 진짜 bash로 돌리고, 두 실행이 한 폴더를 같이 쓰는 경우를 순서대로 흉내 낸다. |
| `pipelines_ch07_jenkinsfile.py` | 7장 Jenkinsfile 검사: Jenkins와 KFP의 경계를 규칙으로 읽는다. stages()는 6장 pipelines_ch06_jenkinsfile.py에서 복사했다. |
| `pipelines_ch07_model_np.py` | 7장 NumPy 모델: 6장 pipelines_ch06_model_np.py(4장에서 복사한 것)를 다시 복사했다. |
| `pipelines_ch07_model_torch.py` | 7장 PyTorch 대응 구현: 6장 pipelines_ch06_model_torch.py를 복사했다. |
| `pipelines_ch07_pipeline.py` | 7장 파이프라인 정의(최소 구현판). kfp_pipeline.py와 같은 파이프라인을 pipelines_ch07_dag로 적었다. |
| `pipelines_ch07_registry.py` | 7장 이미지 태그 시뮬레이션: Jenkins가 시험한 이미지와 KFP task가 실제로 돈 이미지가 같은가. |
| `pipelines_ch07_steps.py` | 7장 단계 세 개: 데이터 준비(prepare) -> 학습(train) -> 평가(evaluate). |
| `pipelines_ch07_submit.py` | 7장 Jenkins가 부르는 KFP run 제출 스크립트(실제 클러스터용). 이 실습 저장소에서는 실행하지 않는다. |
<!-- sync:end -->
