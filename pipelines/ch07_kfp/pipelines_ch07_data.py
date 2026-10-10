"""7장 데이터: 6장(pipelines/ch06_release/pipelines_ch06_data.py)의 고객 이탈 생성기를 복사했다.

장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 복사했다. 이 장은 v1 모델(수치 6열 + 요금제·지역)만 쓴다.
shift는 이탈 로짓에 더하는 값이다. "어제 데이터"(shift=0)와 "오늘 데이터"(shift=1, 경쟁사 할인 행사로 이탈이 늘어난 날)를
같은 코드로 만들어, 단계 사이 연결이 틀렸을 때 어느 날의 데이터를 썼는지 숫자로 드러나게 한다.
"""

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
CATEGORICAL = {"plan": 3, "region": 4}
LABEL = "churned"


def make_table(data_seed, n_rows, shift=0.0):
    rng = np.random.default_rng(data_seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(3, n_rows, p=[0.5, 0.3, 0.2]).astype(np.int64)
    region = rng.choice(4, n_rows, p=[0.35, 0.30, 0.15, 0.20]).astype(np.int64)
    usage_drop = rng.beta(2.0, 5.0, n_rows)
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
        "age": age, "discount_rate": discount, "plan": plan, "region": region, LABEL: churned,
    }


def split(table, n_train, split_seed):
    n = len(table[LABEL])
    order = np.random.default_rng(split_seed).permutation(n)
    tr, va = order[:n_train], order[n_train:]
    return {k: v[tr] for k, v in table.items()}, {k: v[va] for k, v in table.items()}
