# 10장. 합성통제

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [비슷한 비교 지역이 없다면 여러 지역을 섞어 하나 만들 수 있을까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/10-synthetic-control.html)

```bash
cd causal/ch10_synthetic_control
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `synth.py` | 10장 합성통제. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 A에만 무료배송을 켠 뒤 주간 주문지수. |
<!-- sync:end -->

처치 지역 1곳과 대조 후보 8곳, 처치 전 20주·후 8주의 가상 주문지수에서 합이 1이고 음수가 아닌 가중치를 좌표 탐색으로 찾아 합성 대조 지역을 만듭니다. 처치 지역이 대조 후보들의 볼록결합 안에 있는 경우와 밖에 있는 경우(사전 적합 실패)에서 효과 추정값, 처치 전 RMSPE, 단순 평균 대조, 가짜 처치 검정(후/전 RMSPE 비율의 순위)을 비교합니다.

## 실행

```bash
cd causal/ch10_synthetic_control
python3 synth.py          # 숫자를 출력해요
python3 -m unittest -q  # 테스트 (pytest도 됩니다)
```

파이썬 표준 라이브러리만 쓰고 시드를 고정해서 같은 출력이 나옵니다.

## 기대 출력 (`python3 synth.py`)

```text
in_hull
  weights [0.069, 0.491, 0.0, 0.0, 0.18, 0.118, 0.141, 0.0]
   pre_rmspe 0.339
   post_rmspe 9.748
   effect 9.713
   equal_weight_did 7.996
   max_donor_pre_mean 130.752
   treated_pre_mean 103.274
   placebo_rank 1
   placebo_ratio_treated 28.74
   placebo_ratio_max_donor 1.952
   pre_mean_gap -0.01
   placebo_ratios_donors [1.55, 1.95, 1.4, 0.75, 1.35, 0.95, 0.93, 1.23]
   placebo_n_donors_above_treated 0
   design_weights_pre_rmspe 0.573
   design_vs_fitted_path_rmse 0.435
   effect_minus_pre_gap_minus_true -0.278
   equal_weight_did_minus_true -2.004
out_hull
  weights [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
   pre_rmspe 51.755
   post_rmspe 81.245
   effect 80.928
   equal_weight_did 38.244
   max_donor_pre_mean 130.752
   treated_pre_mean 181.502
   placebo_rank 2
   placebo_ratio_treated 1.57
   placebo_ratio_max_donor 1.952
   pre_mean_gap 50.753
   placebo_ratios_donors [1.55, 1.95, 1.4, 0.75, 1.35, 0.95, 0.93, 1.23]
   placebo_n_donors_above_treated 1
   effect_minus_pre_gap_minus_true 20.176
   equal_weight_did_minus_true 28.244
```

## 이 실습이 확인하지 못하는 것

가중치 탐색은 단순 좌표 탐색이라 공변량 매칭이나 통계 라이브러리의 최적화와 같은 해라고 보장하지 않습니다. 가짜 처치 검정은 순위만 보며 정식 p값이나 신뢰구간은 다루지 않습니다. 실제 데이터의 구조 변화·파급효과는 넣지 않았습니다.
