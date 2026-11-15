# 15장. 내 데이터 구조에는 무엇을 써야 하나

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [내 데이터 구조에는 어떤 방법을 써야 할까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/15-identification-map.html)

```bash
cd causal/ch15_identification_map
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `id_map.py` | 15장 실습: 식별 전략 선택 지도. 같은 데이터에 방법을 바꿔 대 보고, 데이터 구조에서 후보를 고른다. |
<!-- sync:end -->

웹 챕터: `tracks/causal/15-identification-map.html`

두 가지를 합니다.
1. 점수 60점 이상에게만 쿠폰을 준 가상 데이터(고객 20,000명, 시드 3) 하나에 방법 이름부터 골라 대 봅니다. 처치군 전후 비교, 사후 단순 차이, 이중차분, 회귀불연속(대역폭 5, 10, 20)을 같은 데이터에 돌립니다.
2. `recommend()`가 데이터 구조(배정 방식, 전후 기간, 비교 집단, 번짐 여부 등)를 받아 후보 장 번호와 먼저 막는 문제를 돌려줍니다.

```bash
python3 id_map.py
python3 -m unittest -q
```

기대 출력(요약):

```
진짜 효과 1500원
처치군 전후 비교 2611원
사후 단순 차이 7888원
이중차분(60점 미만을 비교군으로) 1863원
회귀불연속 대역폭 5: 1499원 (표준오차 253, 표본 1986명)
회귀불연속 대역폭 10: 1748원 (표준오차 177, 표본 3916명)
회귀불연속 대역폭 20: 1551원 (표준오차 126, 표본 7942명)
```

확인하지 못하는 것: `recommend()`는 후보를 줄이는 규칙표이지 가정이 맞는지 검정하지 않습니다. 평행추세, 기준선 조작 여부, 도구의 배제제약은 해당 장의 점검으로 따로 확인해야 합니다. 데이터는 한 가지 구조에서만 만들었으므로 다른 구조에서 어떤 방법이 틀리는지는 재지 않습니다.
