# 8장 데이터 리니지

<!-- sync:start -->
> [ML 파이프라인 구축](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/pipelines) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [8장 데이터 리니지](https://math-ai-matrix-pi.vercel.app/tracks/pipelines/08-lineage.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q pipelines/ch08_lineage
```

| 파일 | 하는 일 |
|---|---|
| `experiments.py` | 8장 비교 실험. `uv run python pipelines/ch08_lineage/experiments.py` 로 실행한다(네트워크·쿠버네티스·kfp 불필요). |
| `pipelines_ch08_data.py` | 8장 데이터: 7장(pipelines/ch07_kfp/pipelines_ch07_data.py)의 고객 이탈 생성기와 |
| `pipelines_ch08_model_np.py` | 8장 NumPy 모델: 7장 pipelines_ch07_model_np.py(6장·4장에서 이어 복사한 것)를 다시 복사했다. |
| `pipelines_ch08_model_torch.py` | 8장 PyTorch 대응 구현: 7장 pipelines_ch07_model_torch.py(6장에서 복사한 것)를 다시 복사했다. |
| `pipelines_ch08_runner.py` | 8장 실행기: 7장 그래프(prepare -> train -> evaluate)를 돌리면서 메타데이터를 남기고, 캐시를 쓴다. |
| `pipelines_ch08_steps.py` | 8장 단계 세 개: prepare -> train -> evaluate. 7장 pipelines_ch07_steps.py를 고쳐 복사했다. |
| `pipelines_ch08_store.py` | 8장 최소 메타데이터 저장소: 표준 라이브러리 sqlite3만 쓴다. |
<!-- sync:end -->
