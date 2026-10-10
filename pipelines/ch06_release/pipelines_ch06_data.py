"""6장 데이터: 4장(pipelines/ch04_container/pipelines_ch04_data.py)의 고객 이탈 생성기를 복사하고 열 하나를 더했다.

장끼리 동시에 작성되므로 import하지 않고 필요한 만큼 복사했다.

이번 장에서 더한 것
- usage_drop: 최근 4주 사용 시간이 그 전 4주보다 얼마나 줄었는지의 비율(0~1). 이탈과 강하게 엮여 있다.
  v1 모델은 이 열을 모르고, v2 모델은 이 열을 새로 쓴다. 그래서 오프라인 검증에서는 v2가 더 잘 맞힌다.
- serve_view: 운영 요청이 모델에 들어올 때의 모습. 한 지역(region 코드 2, 부산)의 앱만
  usage_drop을 비율이 아니라 퍼센트(0~100)로 보낸다. 학습·검증 데이터(오프라인 로그)는
  정리된 테이블에서 왔으므로 모두 비율이다. 학습과 서빙에서 같은 열의 뜻이 다른 상황(training-serving skew)이다.
"""

import numpy as np

NUMERIC_V1 = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
NUMERIC_V2 = NUMERIC_V1 + ["usage_drop"]
CATEGORICAL = {"plan": 3, "region": 4}
REGIONS = ["seoul", "gyeonggi", "busan", "etc"]
SKEW_REGION = 2  # busan
LABEL = "churned"


def make_table(data_seed, n_rows, shift=0.0):
    """shift: 이탈 로짓에 더하는 값. 배포와 상관없이 세상이 바뀐 날(예: 경쟁사 할인 행사)을 흉내 낸다.
    전후 비교(배포 전 v1 vs 배포 뒤 v2)가 배포 효과와 시간 효과를 섞는 문제를 재현할 때 쓴다."""
    rng = np.random.default_rng(data_seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(3, n_rows, p=[0.5, 0.3, 0.2]).astype(np.int64)
    region = rng.choice(4, n_rows, p=[0.35, 0.30, 0.15, 0.20]).astype(np.int64)
    usage_drop = rng.beta(2.0, 5.0, n_rows)  # 평균 약 0.29
    logit = (
        -0.3
        + shift
        + 1.6 * (fee - 55_000) / 15_000
        - 1.3 * (tenure - 36) / 20
        + 0.5 * (calls - 1.8)
        - 0.8 * (usage - 40) / 12
        - 0.6 * (discount - 0.15) / 0.087
        + np.array([0.9, 0.0, -1.1])[plan]
        + np.array([0.0, 0.1, -0.1, 0.2])[region]
        + 0.8 * (usage_drop - 0.29) / 0.16
    )
    churned = (rng.uniform(size=n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return {
        "tenure_months": tenure, "monthly_fee": fee, "support_calls": calls, "usage_hours": usage,
        "age": age, "discount_rate": discount, "usage_drop": usage_drop,
        "plan": plan, "region": region, LABEL: churned,
    }


def split(table, n_train, split_seed):
    """행을 섞어 앞 n_train행은 학습, 나머지는 검증."""
    n = len(table[LABEL])
    order = np.random.default_rng(split_seed).permutation(n)
    tr, va = order[:n_train], order[n_train:]
    return {k: v[tr] for k, v in table.items()}, {k: v[va] for k, v in table.items()}


def serve_view(table, skew=True):
    """운영 요청이 모델에 들어오는 모습. skew=True면 부산 앱이 usage_drop을 퍼센트로 보낸다."""
    out = {k: v.copy() for k, v in table.items()}
    if skew:
        m = out["region"] == SKEW_REGION
        out["usage_drop"][m] = out["usage_drop"][m] * 100.0
    return out
