# 13장. CATE와 업리프트

<!-- sync:start -->
> [인과추론](https://github.com/yujinjeong0917/math-ai-matrix-labs/tree/main/causal) 트랙 · 전체 목차는 [저장소 README](https://github.com/yujinjeong0917/math-ai-matrix-labs)
> 웹 챕터: [누구에게 효과가 큰지 찾아 쿠폰을 그 고객에게만 줘도 될까요?](https://math-ai-matrix-pi.vercel.app/tracks/causal/13-cate-uplift.html)

```bash
cd causal/ch13_cate_uplift
python3 -m unittest -q   # 표준 라이브러리만 사용
```

| 파일 | 하는 일 |
|---|---|
| `cate_uplift.py` | 13장 실습: CATE, 업리프트, 사후 하위집단 탐색의 함정. 표준 라이브러리만 쓴다. |
<!-- sync:end -->

웹 챕터: `tracks/causal/13-cate-uplift.html`

무작위로 쿠폰을 준 가상 온라인 마켓(고객 20,000명, 시드 2)에서 고객 유형별 효과(CATE)를 재고, 그 추정으로 쿠폰 대상을 고르는 업리프트 정책을 보류 데이터로 평가합니다. 이어서 효과가 모든 고객에게 0인 세계에서 하위집단을 뒤지면 얼마나 자주 가짜 발견이 나오는지 200번 반복해 셉니다.

```bash
python3 cate_uplift.py
python3 -m unittest -q
```

기대 출력(요약):

```
전체 평균효과 추정 1480원 (표준오차 99), 진짜 평균 1554원
사전에 정한 하위집단 신규: 2870원 (표준오차 154)
사전에 정한 하위집단 기존: 562원 (표준오차 127)
정책 순이익(보류 절반, 원/명): 업리프트 기준 604, 모두에게 59, 아무에게도 0
share_best_sig: 0.5600        # 효과 0인데 최고 하위집단이 |t|>1.96인 비율
share_best_sig_bonf: 0.0150   # 다중비교 보정 기준(|t|>3.23)을 넘는 비율
holdout_share_sig: 0.0400     # 같은 하위집단을 보류 데이터에서 다시 재면
```

확인하지 못하는 것: 진짜 효과를 아는 가상 데이터라서 실제 데이터의 모형 오지정, 연속 변수 분할, 숲(causal forest) 같은 기계학습 추정기의 성능은 재지 않습니다. 쿠폰 비용 1,500원과 효과 크기는 예시용 가상 값입니다.
