# 11장. 스토어 제출 묶음 점검

<!-- sync:start -->
> [1인 개발 앱 출시 실전](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/webapp) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [11장 앱스토어에 올리려면 파일 말고 무엇이 필요할까요?](https://math-ai-matrix-pi.vercel.app/tracks/webapp/11-store-submission.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q webapp/ch11_store_submission
```

| 파일 | 하는 일 |
|---|---|
| `submission.py` | 스토어 제출 점검표 (1인 개발 앱 출시 실전 11장). |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/11-store-submission.html

항목을 체크했다는 말 대신 근거 URL·파일 위치가 있는지를 확인합니다. 정책 값은 저장하지 않고 제출 당일 공식 콘솔에서 사람이 다시 확인할 항목으로 남깁니다.
