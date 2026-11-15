# 6장. 배정·수송: 주문을 가장 싼 창고에 보내면 무엇이 깨질까

<!-- sync:start -->
> [경영과학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/or) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [주문을 가장 싼 창고에 보내면 무엇이 깨질까요?](https://math-ai-matrix-pi.vercel.app/tracks/or/06-assignment-transport.html)

```bash
cd or/ch06_assignment_transport
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `transport.py` | 6장. 주문을 어느 창고에 보낼까: 수송 문제를 최소비용 흐름으로 풀기. |
<!-- sync:end -->

무엇을 재나: 창고 3곳, 지역 3곳, 금지 조합 1개가 있는 가상 수송 문제를 최소비용 흐름(연속 최단경로)으로 풀고, 지역별 최저가 선택·순서대로 탐욕·전수 탐색과 대조한다. 단위는 10개 묶음, 비용은 가상 배송비다.

실행(이 폴더 안에서):

```bash
python3 -m unittest -v
python3 transport.py
```

기대 출력(요약):

- 각 지역이 최저가 창고: 비용 25, 그러나 W2에 9묶음(용량 3 초과)
- 순서대로 탐욕: 비용 40
- 최소비용 흐름: 비용 34, 전수 탐색도 34(가능한 계획 63개, 최적 해 1개)
- W2 용량 3, 4, 5, 6 일 때 비용 34, 32, 30, 28
- 금지 조합과 낮은 공급이 겹친 경우: 9묶음 중 8묶음만 가능(불가능)

확인하지 못하는 것: 실제 배송비 구조(구간 요금, 고정비), 시간에 따른 재고 변화, 외부 솔버와의 결과 비교.
