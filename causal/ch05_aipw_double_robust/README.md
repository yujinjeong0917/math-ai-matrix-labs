# 5장 실습. AIPW와 이중강건

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [결과모형과 가중치 중 하나가 틀리면 효과 추정은 어떻게 될까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/05-aipw-double-robust.html)

```bash
cd causal/ch05_aipw_double_robust
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `aipw.py` | 5장 실습: AIPW(증강 역확률가중)와 이중강건, 교차적합. 표준 라이브러리만 사용합니다. |
<!-- sync:end -->

웹 챕터: https://math-ai-matrix-pi.vercel.app/tracks/causal/05-aipw-double-robust.html

가상 온라인 마켓 쿠폰 효과(진짜 값 +8.0, 천 원 단위 가상)를 네 가지 조합으로 추정합니다. 결과모형(쿠폰이 없을 때/있을 때 주문액을 x로 예측)과 성향점수모형(쿠폰 받을 확률)을 각각 맞게(2차식) 또는 틀리게(1차식) 두고, 결과모형만 쓰는 방법·IPW·AIPW를 비교합니다. 마지막에 과적합하는 1-최근접 결과모형으로 교차적합이 무엇을 바꾸는지 봅니다.

## 실행

```bash
cd causal/ch05_aipw_double_robust
python3 aipw.py              # 본문 숫자를 모두 출력 (약 4초)
python3 -m unittest -v       # 출력값을 고정한 검사 (pytest도 가능)
```

표준 라이브러리만 씁니다. 난수는 `random.Random(seed).random()`만 쓰고 정규분포는 직접 만듭니다.

## 기대 출력 (일부)

```
둘 다 맞음:           결과모형만 8.53, IPW 9.77, AIPW 8.47
결과모형만 틀림:       AIPW 9.83 (100번 반복 평균 오차 +0.05, 표준편차 4.37)
성향점수모형만 틀림:   AIPW 8.53 (100번 반복 평균 오차 +0.02)
둘 다 틀림:           AIPW 18.38 (100번 반복 평균 오차 +10.59)
```

## 이 실습이 확인하지 못하는 것

- 두 모형이 모두 틀려도 편향이 반드시 큰 것은 아니고, 이 장난감에서는 두 틀림이 같은 방향(제곱항 누락)이라 크게 나왔습니다.
- 이중강건은 "둘 중 하나가 맞으면 평균적으로 맞는다"는 뜻이지 분산이 작다는 뜻이 아닙니다. 결과모형만 틀린 경우 평균은 맞아도 표준편차가 4.37로 커요.
- 교차적합 비교는 한 가지 학습기(1-최근접)와 한 가지 표본 크기(400)만 봤습니다. 교차적합이 늘 낫다는 증거가 아니에요. 이 설정에서는 평균 오차는 줄고 흔들림은 커져 제곱평균제곱근 오차는 오히려 컸습니다.
- 표준오차·신뢰구간은 계산하지 않았습니다.
