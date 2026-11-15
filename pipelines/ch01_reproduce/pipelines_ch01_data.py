"""1장 데이터: 고객 이탈을 흉내 낸 합성 표 데이터.

make_customers는 2장(pipelines/ch02_hidden_debt/pipelines_ch02_data.py)의 생성기를 그대로 복사했다.
장끼리 import하지 않는 저장소 규칙 때문이다. 같은 seed면 2장과 같은 고객이 나온다.

- 수치 6열: tenure_months, monthly_fee, support_calls, usage_hours, age, discount_rate
- 범주 2열: plan(요금제 3종), region(지역 4곳)
- 라벨 churned: 로지스틱 모형에서 표본추출한다.

1장은 데이터 자체를 고정해 두고(데이터 시드 DATA_SEED), 학습 쪽의 무작위만 바꿔 가며 본다.
"""

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
PLAN_LABELS = ["basic", "standard", "premium"]
REGION_LABELS = ["seoul", "gyeonggi", "busan", "etc"]
FEATURES = NUMERIC + [f"plan={p}" for p in PLAN_LABELS] + [f"region={r}" for r in REGION_LABELS]

# 데이터 시드는 고정한다. 장 전체에서 바뀌는 것은 분할·초기값·순서의 무작위뿐이다
DATA_SEED = 2026
N_ROWS = 2000


def make_customers(seed, n_rows, plan_probs=(0.5, 0.3, 0.2)):
    """(2장에서 복사) 세상 쪽 고객 표. 금액은 원 단위, 범주는 글자 라벨."""
    # default_rng(seed): 전역 상태가 없는 난수 생성기. 같은 seed면 같은 표가 나온다
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    # 월 요금(원). 정규분포에서 뽑아 15,000~120,000으로 자르고 100원 단위로 반올림
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    # 상담 횟수: 가입 기간이 길수록 평균(최소 0.2)이 줄어든다(모형의 선택)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(len(PLAN_LABELS), n_rows, p=np.asarray(plan_probs))
    region = rng.choice(len(REGION_LABELS), n_rows, p=[0.35, 0.30, 0.15, 0.20])

    # 범주 효과는 로짓에 더하는 상수. 계수는 합성 데이터를 만들기 위한 값이고 현실 추정치가 아니다
    plan_effect = np.array([0.9, 0.0, -1.1])[plan]
    region_effect = np.array([0.0, 0.1, -0.1, 0.2])[region]
    # 이탈 로짓 = 절편 + 각 변수의 표준화 효과. 각 항은 (값 - 기준)/척도 꼴이다
    logit = (
        -0.3
        + 1.6 * (fee - 55_000) / 15_000
        - 1.3 * (tenure - 36) / 20
        + 0.5 * (calls - 1.8)
        - 0.8 * (usage - 40) / 12
        - 0.6 * (discount - 0.15) / 0.087
        + plan_effect
        + region_effect
    )
    # 로지스틱 확률 1/(1+e^-logit)보다 uniform 난수가 작으면 이탈(1.0)로 표본추출
    churned = (rng.uniform(size=n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return {
        "tenure_months": tenure,
        "monthly_fee": fee,
        "support_calls": calls,
        "usage_hours": usage,
        "age": age,
        "discount_rate": discount,
        "plan": np.array([PLAN_LABELS[i] for i in plan]),
        "region": np.array([REGION_LABELS[i] for i in region]),
        "churned": churned,
    }


def make_dataset(seed=DATA_SEED, n_rows=N_ROWS, schema_version=1):
    """모델이 받는 행렬 X (n, 13)과 라벨 y.

    schema_version=1: 요금은 원 단위. schema_version=2: 요금을 천 원 단위로 보낸다(2장에서 다룰 변경).
    수치 열은 아직 표준화하지 않는다. 표준화 기준(평균·표준편차)은 학습 분할에서 구한다.
    """
    c = make_customers(seed, n_rows)
    # 수치 6열을 NUMERIC 순서대로 쌓는다. 열 순서는 FEATURES와 같아야 한다
    cols = [c[k].astype(float) for k in NUMERIC]
    if schema_version == 2:
        # schema_version 2: 같은 요금을 천 원 단위로 바꿔 열의 스케일만 달라진다(2장 사고 재현용)
        cols[NUMERIC.index("monthly_fee")] = cols[NUMERIC.index("monthly_fee")] / 1000.0
    elif schema_version != 1:
        raise ValueError(f"모르는 schema_version: {schema_version}")
    # 범주 열은 원-핫(해당 라벨이면 1.0). 전체 열 수는 6 + 3 + 4 = 13
    for labels, key in ((PLAN_LABELS, "plan"), (REGION_LABELS, "region")):
        for lab in labels:
            cols.append((c[key] == lab).astype(float))
    return np.column_stack(cols), c["churned"]
