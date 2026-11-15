# 13장. 출시 뒤 다음 실험 고르기

<!-- sync:start -->
> [1인 개발 앱 출시 실전](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/webapp) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [13장 다운로드는 늘었는데 왜 아무도 끝까지 쓰지 않을까요?](https://math-ai-matrix-pi.vercel.app/tracks/webapp/13-release-learning.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q webapp/ch13_release_learning
```

| 파일 | 하는 일 |
|---|---|
| `funnel.py` | 단계 전환율로 먼저 조사할 단계 찾기 (1인 개발 앱 출시 실전 13장). |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/13-release-learning.html

단계별 고유 사용자 수로 퍼널을 계산하고 가장 큰 직전 단계 이탈을 찾습니다. 이 결과는 원인 판정이 아니라 다음 조사 후보입니다.
