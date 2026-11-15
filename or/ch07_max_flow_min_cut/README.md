# 7장. 최대유량·최소절단: 병목 연결을 늘리면 흐름이 그만큼 늘어날까

<!-- sync:start -->
> [경영과학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/or) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [병목 연결을 늘리면 흐름이 그만큼 늘어날까요?](https://math-ai-matrix-pi.vercel.app/tracks/or/07-max-flow-min-cut.html)

```bash
cd or/ch07_max_flow_min_cut
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `max_flow.py` | 7장. 병목은 어느 연결인가: 최대유량·최소절단. |
<!-- sync:end -->

무엇을 재나: 6개 점, 8개 연결의 가상 주문 처리망에서 에드먼즈-카프 최대유량, 잔여 그래프의 최소절단, 모든 절단 열거, 연결 증설 효과를 확인한다. 용량 단위는 건/시간(가상)이다.

실행(이 폴더 안에서):

```bash
python3 -m unittest -v
python3 max_flow.py
```

기대 출력(요약):

- 최대유량 11 (증가경로 4, 3, 1, 3건). 최소절단은 C→T, D→T (5+6=11)
- 점 4개를 나누는 절단 16개 중 최소 11. 최소절단은 하나
- 연결 하나를 +2: C→T, D→T는 +1, 나머지 6개는 +0. 두 병목을 함께 +2: 13
- 변형(D→T=7): 최대유량 12, 최소절단 둘. D→T, A→D는 +0, C→T는 +1

확인하지 못하는 것: 실제 처리 구조와의 일치, 용량의 시간 변화, 여러 종류 흐름. 외부 솔버와의 결과 비교.
