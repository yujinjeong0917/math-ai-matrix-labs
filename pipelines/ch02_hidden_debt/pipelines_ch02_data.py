"""2장 데이터: 고객 이탈을 흉내 낸 합성 표 데이터와, 데이터를 보내 주는 쪽(업스트림)의 버전별 출력.

생성기(make_customers)는 '세상'이고, 업스트림 버전(emit_batch)은 그 세상을 표로 적어 보내는 방식이다.
같은 고객이라도 업스트림이 단위나 코드표를 바꾸면 모델이 받는 숫자가 달라진다.

- 수치 6열: tenure_months, monthly_fee, support_calls, usage_hours, age, discount_rate
- 범주 2열: plan(요금제), region(지역). 업스트림은 범주를 정수 코드로 보낸다.
- 라벨 churned: 로지스틱 모형에서 표본추출한다.
"""

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
CATEGORICAL = ["plan", "region"]
PLAN_LABELS = ["basic", "standard", "premium"]
REGION_LABELS = ["seoul", "gyeonggi", "busan", "etc"]

# 업스트림 버전마다 다른 '보내는 방식'. 고객(세상)은 그대로다.
UPSTREAM = {
    1: {"fee_divisor": 1.0, "plan_codebook": ["basic", "standard", "premium"], "usage_scale": 1.0},
    # v2: 금액을 천 원 단위로 보내고, 요금제 코드 순서를 바꿨다(열 이름은 그대로).
    2: {"fee_divisor": 1000.0, "plan_codebook": ["premium", "basic", "standard"], "usage_scale": 1.0},
}


def make_customers(seed, n_rows, plan_probs=(0.5, 0.3, 0.2)):
    """세상 쪽 고객 표. 금액은 원 단위, 범주는 글자 라벨, 사용 시간은 실제 값."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)  # 오래 쓴 고객일수록 문의가 적다
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(len(PLAN_LABELS), n_rows, p=np.asarray(plan_probs))
    region = rng.choice(len(REGION_LABELS), n_rows, p=[0.35, 0.30, 0.15, 0.20])

    plan_effect = np.array([0.9, 0.0, -1.1])[plan]
    region_effect = np.array([0.0, 0.1, -0.1, 0.2])[region]
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


def emit_batch(customers, version=1, fee_divisor=None, plan_codebook=None, usage_scale=None):
    """업스트림이 보내는 표. 수치는 float, 범주는 그 버전의 코드표로 바꾼 정수 코드.

    인자를 직접 주면 그 버전 설정을 덮어쓴다(단위 배율 곡선 실험용).
    """
    cfg = dict(UPSTREAM[version])
    if fee_divisor is not None:
        cfg["fee_divisor"] = fee_divisor
    if plan_codebook is not None:
        cfg["plan_codebook"] = plan_codebook
    if usage_scale is not None:
        cfg["usage_scale"] = usage_scale
    out = {k: customers[k].astype(float).copy() for k in NUMERIC}
    out["monthly_fee"] = out["monthly_fee"] / cfg["fee_divisor"]
    out["usage_hours"] = out["usage_hours"] * cfg["usage_scale"]
    plan_index = {lab: i for i, lab in enumerate(cfg["plan_codebook"])}
    out["plan"] = np.array([plan_index[v] for v in customers["plan"]], dtype=np.int64)
    out["region"] = np.array([REGION_LABELS.index(v) for v in customers["region"]], dtype=np.int64)
    return out


def emit_labeled_batch(customers, fee_divisor=1.0, usage_scale=1.0):
    """'뜻이 담긴 계약' 설계: 범주는 글자 라벨 그대로, 금액 단위는 열 이름에 적어 보낸다."""
    out = {k: customers[k].astype(float).copy() for k in NUMERIC if k != "monthly_fee"}
    unit = "krw" if fee_divisor == 1.0 else ("krw_thousand" if fee_divisor == 1000.0 else f"krw_div{fee_divisor:g}")
    out[f"monthly_fee_{unit}"] = customers["monthly_fee"].astype(float) / fee_divisor
    out["usage_hours"] = out["usage_hours"] * usage_scale
    out["plan"] = customers["plan"].copy()
    out["region"] = customers["region"].copy()
    return out
