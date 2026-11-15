# 1장. 두 상품을 몇 개 만들까

<!-- sync:start -->
> [경영과학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/or) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [두 상품을 몇 개씩 만들어야 이익이 가장 클까요?](https://math-ai-matrix-pi.vercel.app/tracks/or/01-product-mix-lp.html)

```bash
cd or/ch01_product_mix_lp
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `product_mix.py` | 1장: 두 상품 생산량 선형계획. 표준 라이브러리만 쓴다. 모든 숫자는 가상 예시용. |
<!-- sync:end -->

운동화 생산량 $x_1$과 양말 묶음 생산량 $x_2$를 정하는 가상의 선형계획입니다. 꼭짓점을 분수(`Fraction`)로 정확히 구한 답과, 가능한 정수 생산량을 전부 세어 본 답을 대조하고, 숨은 최소 생산량이 해를 바꾸거나 없애는 장면을 확인합니다.

## 실행

```bash
cd or/ch01_product_mix_lp
python3 product_mix.py
python3 -m unittest -v
```

## 기대 출력 (핵심)

```
기본 LP 최적: (20, 60), 마진 180만원 / 전수: ((20, 60), 180)
양말 최소 70: (10, 70), 170 / 전수: ((10, 70), 170)
양말 최소 90: 가능한 생산량 없음 (둘 다 None)
```

## 확인하지 못하는 것

- 변수가 둘인 문제의 꼭짓점 열거만 다룹니다. 변수가 많을 때의 단체법·내점법은 다루지 않고, 실무에서는 HiGHS나 OR-Tools 같은 솔버를 씁니다.
- 마진·시간은 가상 값입니다.
