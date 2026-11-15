# 인과추론

[저장소 전체 목차](https://github.com/yujinjeong0917/math-ai-matrix-labs) · [웹에서 이 트랙 보기](https://math-ai-matrix-pi.vercel.app/tracks/causal/00-prediction-intervention-counterfactual.html) · 장 16개

## 실행

```bash
uv sync   # 저장소 루트에서 한 번
uv run pytest -q causal   # 1/16장만 `uv` 환경 필요
```

## 장 목록

| 폴더 | 웹 챕터 제목 | 웹 |
|---|---|---|
| [`ch00_prediction_intervention_counterfactual/`](ch00_prediction_intervention_counterfactual/) | 매출이 올랐으면 새 기능 덕분일까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/00-prediction-intervention-counterfactual.html) |
| [`ch01_potential_outcomes/`](ch01_potential_outcomes/) | 같은 사람의 쿠폰 있을 때와 없을 때를 왜 함께 볼 수 없을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/01-potential-outcomes.html) |
| [`ch02_randomized_experiments/`](ch02_randomized_experiments/) | 무작위 실험이면 다 끝나나요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/02-randomized-experiments.html) |
| [`ch03_dag_adjustment/`](ch03_dag_adjustment/) | 회귀식에 무엇을 넣고 무엇을 빼야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/03-dag-adjustment.html) |
| [`ch04_propensity_matching_ipw/`](ch04_propensity_matching_ipw/) | 비슷한 고객끼리 비교하면 쿠폰 효과를 알 수 있을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/04-propensity-matching-ipw.html) |
| [`ch05_aipw_double_robust/`](ch05_aipw_double_robust/) | 결과모형과 가중치 중 하나가 틀리면 효과 추정은 어떻게 될까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/05-aipw-double-robust.html) |
| [`ch06_difference_in_differences/`](ch06_difference_in_differences/) | 전후 비교에 대조군 하나를 더하면 무엇이 달라질까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/06-difference-in-differences.html) |
| [`ch07_staggered_did_event_study/`](ch07_staggered_did_event_study/) | 지역마다 쿠폰 도입일이 다르면 어떻게 비교해야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/07-staggered-did-event-study.html) |
| [`ch08_regression_discontinuity/`](ch08_regression_discontinuity/) | 기준선 바로 위아래 고객을 비교해도 될까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/08-regression-discontinuity.html) |
| [`ch09_instrumental_variables/`](ch09_instrumental_variables/) | 쿠폰 알림을 도구로 쓰면 쿠폰의 진짜 효과를 가려낼 수 있을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/09-instrumental-variables.html) |
| [`ch10_synthetic_control/`](ch10_synthetic_control/) | 비슷한 비교 지역이 없다면 여러 지역을 섞어 하나 만들 수 있을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/10-synthetic-control.html) |
| [`ch11_interference_sutva/`](ch11_interference_sutva/) | 내 쿠폰이 이웃의 지출까지 바꾸면 A/B 테스트는 무엇을 재고 있을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/11-interference-sutva.html) |
| [`ch12_switchback_washout/`](ch12_switchback_washout/) | 지역을 나눌 수 없다면 시간대마다 켰다 꺼서 효과를 잴 수 있을까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/12-switchback-washout.html) |
| [`ch13_cate_uplift/`](ch13_cate_uplift/) | 누구에게 효과가 큰지 찾아 쿠폰을 그 고객에게만 줘도 될까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/13-cate-uplift.html) |
| [`ch14_sensitivity_placebo/`](ch14_sensitivity_placebo/) | 이 결론은 보이지 않는 변수와 다른 지역에서도 버틸까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/14-sensitivity-placebo-external-validity.html) |
| [`ch15_identification_map/`](ch15_identification_map/) | 내 데이터 구조에는 어떤 방법을 써야 할까요? | [웹 챕터](https://math-ai-matrix-pi.vercel.app/tracks/causal/15-identification-map.html) |
