# 12장. 제품 분석 이벤트 계약

<!-- sync:start -->
> [1인 개발 앱 출시 실전](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/webapp) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [12장 사람이 들어왔는지만 보면 앱이 나아질까요?](https://math-ai-matrix-pi.vercel.app/tracks/webapp/12-product-analytics.html)

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q webapp/ch12_product_analytics
```

| 파일 | 하는 일 |
|---|---|
| `events.py` | 제품 분석 이벤트 계약과 완료 퍼널 (1인 개발 앱 출시 실전 12장). |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/webapp/12-product-analytics.html

이벤트 이름·발생 조건·허용 속성을 계약으로 만들고, 중복 완료와 금지 개인정보를 찾습니다.
