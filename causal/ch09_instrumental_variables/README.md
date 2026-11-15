# 9장. 도구변수와 LATE

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [쿠폰 알림을 도구로 쓰면 쿠폰의 진짜 효과를 가려낼 수 있을까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/09-instrumental-variables.html)

```bash
cd causal/ch09_instrumental_variables
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `iv.py` | 9장 도구변수. 표준 라이브러리만 쓴다. 가상 온라인 마켓의 장난감 데이터. |
<!-- sync:end -->

쿠폰 알림(무작위 권유)을 도구로 써서 쿠폰 사용의 효과를 추정합니다. 쿠폰을 쓴 사람과 안 쓴 사람의 단순 비교, Wald 추정량(알림이 지출을 바꾼 정도 ÷ 알림이 쿠폰 사용을 바꾼 정도), 첫 단계 F, 약한 도구에서 추정값이 흩어지는 정도, 배제제약을 어겼을 때의 편향을 같은 장난감 데이터에서 잽니다. 진짜 효과를 아는 가상 데이터라 맞는 장면과 틀리는 장면을 함께 봅니다.

## 실행

```bash
cd causal/ch09_instrumental_variables
python3 iv.py          # 숫자를 출력해요
python3 -m unittest -q  # 테스트 (pytest도 됩니다)
```

파이썬 표준 라이브러리만 쓰고 시드를 고정해서 같은 출력이 나옵니다.

## 기대 출력 (`python3 iv.py`)

```text
n 20000
naive 26.926
itt_y 4.857
first_stage 0.503
wald 9.662
F 6835.155
true_late 10.0
true_ate 5.849
strong_reps {'median': 9.961405441196254, 'p10': 8.253951332539975, 'p90': 11.307599611837446, 'sd': 1.1970980144369507}
weak_reps {'median': 13.415946514297273, 'p10': -67.63411077733664, 'p90': 32.08649450026539, 'sd': 447.0125957775689}
weak_first_stage 0.033
weak_F_n20000 31.014
weak_F_n2000 2.145
weak_F_mean_n2000 3.497
weak_F_mean_n20000 25.504
weak_wald_n20000_no_direct 7.15
n_coupon_users 9105
n_notified 10042
share_always 0.202
share_complier 0.504
share_never 0.294
strong_F_n2000 790.942
violate_strong 11.651
violate_weak 37.863
```

## 이 실습이 확인하지 못하는 것

도구가 무작위여도 '순응자'에게만 성립하는 값(LATE)이며 전체 평균효과(ATE)가 아닙니다. 단조성(도망치는 사람 없음)과 배제제약은 코드가 참이라고 정해 둔 가정이지 데이터로 증명하지 않습니다. 실제 데이터의 표준오차, 공변량, 도구가 여러 개인 경우는 다루지 않습니다.
