"""3장 데이터: 2장(pipelines/ch02_hidden_debt/pipelines_ch02_data.py)의 고객 이탈 생성기를 복사해 줄였다.

장끼리 동시에 작성되므로 import하지 않고 필요한 만큼 복사했다.
make_table(data_seed, n_rows)는 표 하나를 열 이름 -> 배열 딕셔너리로 돌려준다.
- 수치 6열(float64), 범주 2열(int64 코드, 코드표 고정), 라벨 churned(float64)
"""

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
# 범주 열 이름 -> 수준 수. 원-핫 열 수(3, 4)가 여기서 정해진다
CATEGORICAL = {"plan": 3, "region": 4}
LABEL = "churned"


def make_table(data_seed, n_rows):
    # data_seed가 같으면 같은 표. 3장의 "데이터를 바꿨는데 안 적었다"는 사고는 이 시드를 바꿔서 흉내 낸다
    rng = np.random.default_rng(data_seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(3, n_rows, p=[0.5, 0.3, 0.2]).astype(np.int64)
    region = rng.choice(4, n_rows, p=[0.35, 0.30, 0.15, 0.20]).astype(np.int64)
    # 이탈 로짓(합성용 가정). 로지스틱 확률로 라벨을 표본추출한다
    logit = (
        -0.3
        + 1.6 * (fee - 55_000) / 15_000
        - 1.3 * (tenure - 36) / 20
        + 0.5 * (calls - 1.8)
        - 0.8 * (usage - 40) / 12
        - 0.6 * (discount - 0.15) / 0.087
        + np.array([0.9, 0.0, -1.1])[plan]
        + np.array([0.0, 0.1, -0.1, 0.2])[region]
    )
    churned = (rng.uniform(size=n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return {
        "tenure_months": tenure, "monthly_fee": fee, "support_calls": calls, "usage_hours": usage,
        "age": age, "discount_rate": discount, "plan": plan, "region": region, LABEL: churned,
    }


def split(table, n_train, split_seed):
    """행을 섞어 앞 n_train행은 학습, 나머지는 검증."""
    n = len(table[LABEL])
    # split_seed로 행을 섞어 앞 n_train행을 학습, 나머지를 검증에 쓴다. 반환 순서: (학습, 검증)
    order = np.random.default_rng(split_seed).permutation(n)
    tr, va = order[:n_train], order[n_train:]
    return {k: v[tr] for k, v in table.items()}, {k: v[va] for k, v in table.items()}
