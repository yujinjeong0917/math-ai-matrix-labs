# 4장. 솔버가 멈췄으면 최선일까

<!-- sync:start -->
> [경영과학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/or) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [솔버가 멈췄다면 그 답이 최선일까요?](https://math-ai-matrix-pi.vercel.app/tracks/or/04-local-vs-global.html)

```bash
cd or/ch04_local_vs_global
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `local_global.py` | 4장: 국소 최적과 전역 최적. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용. |
<!-- sync:end -->

운동화 가격 하나를 고르는 가상 이익 곡선에 봉우리가 둘 있습니다. 경사 상승이 시작점에 따라 다른 봉우리에서 멈추는 것을 보이고, 촘촘한 격자 탐색과 다중 시작점으로 전역 최대를 대조합니다. 표준 라이브러리만 쓰며 난수가 없어 결정적입니다.

## 실행

```bash
cd or/ch04_local_vs_global
python3 local_global.py
python3 -m unittest -v
```

## 기대 출력 (핵심)

```
격자(0.01)에서 찾은 국소 최대: [(41.58, 1707.22), (86.16, 1805.19)]
시작 25 -> p=41.58 이익=1707.22 / 시작 80 -> p=86.16 이익=1805.19
골짜기: 56.26 1230.51
```

## 확인하지 못하는 것

- 변수가 하나라서 격자로 전역 최대를 확인할 수 있습니다. 변수가 많으면 격자는 불가능하고 전역 최적을 보장하기 어렵습니다.
- 이익 곡선(두 고객 묶음의 시그모이드 합)은 학습용으로 만든 가상 모형입니다.
