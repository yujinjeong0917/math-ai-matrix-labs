# 5장 늦은 통합

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [5장 늦은 통합](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/05-jenkins-ci.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch05_jenkins_ci
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 5장 비교 실험. `uv run python pipelines/ch05_jenkins_ci/experiments.py` 로 실행한다(git 필요, 네트워크·Jenkins·Docker 불필요). |
| `pipelines_ch05_history.py` | 2주 동안 4명이 같은 저장소를 고치는 상황을 진짜 git으로 만들고, 합치는 간격만 바꿔 비교한다. |
| `pipelines_ch05_jenkins.py` | Jenkinsfile(Declarative Pipeline)의 작은 부분집합을 읽고 실제로 실행하는 흉내 러너. |
| `pipelines_ch05_model_torch.py` | 5장 PyTorch 대응 구현: 4장 pipelines_ch04_model_torch.py를 복사했다. |
| `pipelines_ch05_project.py` | 5장 실습용 작은 프로젝트를 진짜 git 저장소로 만든다. |
| `pipelines_ch05_scenarios.py` | 5장 실험과 테스트가 함께 쓰는 도우미: 브랜치를 새 작업 공간에 받아 파이프라인 돌리기, Jenkinsfile 변형 만들기. |
<!-- sync:end -->
