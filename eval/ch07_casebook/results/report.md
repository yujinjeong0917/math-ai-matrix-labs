# 모델 평가 7장 판독 리포트

숫자는 results/ch07.json(시드 0)에서 옮겼다. 깃발은 read_table의 판독 결과다.

## control

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 2.211 | 2.813 | 0.868 | 2.23% | 0.628 | -0.34% | 2.211 |
| B | 2.410 | 2.974 | 0.853 | 2.43% | 0.684 | -0.11% | 2.410 |

- 목적 손실(abs_units)이 고른 모델: A
- 지표별 선택: mae=A, rmse=A, r2=A, mape=A, mase=A, nmbe=B
- 켜진 깃발: indistinct

## spiky

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 1.814 | 4.592 | 0.658 | 1.80% | 0.498 | 0.40% | 4.814 |
| B | 2.651 | 3.364 | 0.817 | 2.66% | 0.728 | -0.11% | 2.798 |

- 목적 손실(big_miss)이 고른 모델: B
- 지표별 선택: mae=A, rmse=B, r2=B, mape=A, mase=A, nmbe=B
- 켜진 깃발: spiky

## small_values

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 0.745 | 0.965 | 0.987 | 24.36% | 0.078 | -0.40% | 0.745 |
| B | 1.159 | 1.958 | 0.948 | 15.23% | 0.121 | 2.29% | 1.159 |

- 목적 손실(abs_units)이 고른 모델: A
- 지표별 선택: mae=A, rmse=A, r2=A, mape=B, mase=A, nmbe=A
- 켜진 깃발: near_zero, spiky

## flat_level

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 7.996 | 9.374 | 0.621 | 0.90% | 2.075 | -0.14% | 7.996 |
| B | 3.590 | 4.643 | 0.907 | 0.40% | 0.931 | -0.01% | 3.590 |

- 목적 손실(abs_units)이 고른 모델: B
- 지표별 선택: mae=B, rmse=B, r2=B, mape=B, mase=B, nmbe=B
- 켜진 깃발: below_naive

## monthly_bias

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 12.087 | 15.126 | 0.518 | 12.71% | 1.608 | -0.10% | 294.195 |
| B | 8.352 | 9.646 | 0.804 | 8.43% | 1.111 | 8.07% | 5810.782 |

- 목적 손실(monthly_total)이 고른 모델: A
- 지표별 선택: mae=B, rmse=B, r2=B, mape=B, mase=B, nmbe=A
- 켜진 깃발: below_naive, bias

## fixable_bias

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 9.116 | 9.631 | 0.799 | 9.59% | 1.647 | 9.10% | 2.513 |
| B | 6.550 | 8.211 | 0.854 | 6.84% | 1.183 | 0.33% | 6.550 |

- 목적 손실(abs_after_offset)이 고른 모델: A
- 지표별 선택: mae=B, rmse=B, r2=B, mape=B, mase=B, nmbe=B
- 켜진 깃발: below_naive, bias

## extrapolation

| | MAE | RMSE | R2 | MAPE | MASE | NMBE | 목적 손실 |
|---|---|---|---|---|---|---|---|
| A | 23.295 | 27.371 | -0.594 | 37.12% | 19.350 | 41.28% | 23.295 |
| B | 0.914 | 1.170 | 0.997 | 1.77% | 0.759 | 0.87% | 0.914 |

- 목적 손실(abs_units)이 고른 모델: B
- 지표별 선택: mae=B, rmse=B, r2=B, mape=B, mase=B, nmbe=B
- 켜진 깃발: below_mean, below_naive, bias
