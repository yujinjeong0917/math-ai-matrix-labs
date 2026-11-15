# 13장. 수요 예측이 틀려도 재고 계획이 버틸까요?

<!-- sync:start -->
> [경영과학](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/or) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [수요 예측이 틀려도 재고 계획이 버틸까요?](https://math-ai-matrix-pi.vercel.app/tracks/or/13-robust-stock.html)

```bash
cd or/ch13_robust_stock
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `robust_stock.py` | 13장: 재고 100개를 두 지역에 나눠 두기. 수요 시나리오에 과적합한 계획과 강건한 계획을 비교한다. |
<!-- sync:end -->

운동화 100켤레를 지역 A·B에 나눠 둘 때, 예측 수요 한 점에 맞춘 계획과 흔들림을 넣은 강건 계획·후회 최소 계획의 이익을 비교하고, 시나리오 개수별 과적합 폭을 잰다. 모든 값은 학습용 가상값이다.

## 실행

```bash
cd or/ch13_robust_stock
python3 robust_stock.py
python3 -m unittest -v
```

(pytest도 쓸 수 있으면 `python3 -m pytest -q`)

## 기대 출력 (일부)

```
x_A = 60 예측 수요에서 이익 = 2760
강건(최악 이익 최대) x_A = 53  후회 최소 x_A = 56
n=   3 학습 안 이익=2489.5 새 시나리오 이익=2394.2 차이=95.3
충격=60 x_A=50 예측수요이익=2700 집합 안 최악=1200 집합 밖(15,50)=1650
n=200 차이(반올림 전) = -5.2518
손계산: 3450-24x = 30x+600 -> x = 52.77777777777778
```

## 규칙

- 충격으로 낮아진 수요는 0 아래로 내려가지 않는다(0에서 자른다).
- 강건 계획이 동점이면 예측 수요 이익이 큰 쪽, 그래도 같으면 작은 x를 고른다.

## 확인하지 못하는 것

- 실제 마켓의 수요 분포. 표준편차 20, 두 지역 독립은 가정이다.
- 재고 이동, 추가 주문, 남은 재고 가치. 모두 없다고 두었다.
- 난수는 `random.Random(seed).gauss`를 쓴다. 다른 파이썬 버전에서 같은 값이 나오는지는 3.15.2에서만 확인했다.
