"""9장 데이터: 8장(pipelines/ch08_lineage/pipelines_ch08_data.py)의 고객 이탈 생성기(source_version v1)와
3장의 dataset_hash를 복사했다. 장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 복사했다.

세상이 바뀌는 손잡이를 세 가지로 나눴다. 기본값이면 8장 v1 표와 바이트까지 같다(테스트로 확인).
새 난수는 하나도 뽑지 않는다. 이미 뽑은 값을 결정적으로 옮기기만 해서, 같은 seed면 같은 고객이 조금 달라진 모습이 된다.

- fee_shift  (입력 분포 P(X)만 바뀜) 월 요금 평균을 표준편차(15,000원) 단위로 옮긴다. 이탈 로짓은 옮긴 요금으로
             계산하므로 '요금과 이탈의 관계'는 그대로다. 예: 요금 인상.
- fee_coef   (관계 P(Y|X)만 바뀜) 로짓에서 요금 항의 계수. 기본 1.6. 줄어들면 비싼 요금이 예전만큼 이탈로 이어지지 않는다.
             입력 표의 숫자는 하나도 바뀌지 않는다.
- shift      (8장까지와 같은 뜻) 이탈 로짓에 더하는 값. 6장에서 '경쟁사 할인 행사'로 쓴 손잡이다. 이것도 입력은 그대로다.
- fee_usage_corr (P(X)의 짝 관계만 바뀜) 요금과 사용 시간의 상관. 두 열 각각의 분포(정규분포의 평균·표준편차)는
             그대로 두고 둘이 함께 움직이는 정도만 바꾼다. 열 하나씩 보는 검사가 원리상 못 보는 변화다.
"""

import hashlib

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
CATEGORICAL = {"plan": 3, "region": 4}
LABEL = "churned"
FEE_MEAN, FEE_STD = 55_000, 15_000


def make_table(data_seed, n_rows, shift=0.0, fee_shift=0.0, fee_coef=1.6, fee_usage_corr=0.0):
    rng = np.random.default_rng(data_seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee_raw = rng.normal(FEE_MEAN, FEE_STD, n_rows)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage_raw = rng.normal(40, 12, n_rows)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(3, n_rows, p=[0.5, 0.3, 0.2]).astype(np.int64)
    region = rng.choice(4, n_rows, p=[0.35, 0.30, 0.15, 0.20]).astype(np.int64)
    usage_drop = rng.beta(2.0, 5.0, n_rows)
    if fee_usage_corr:
        r = float(fee_usage_corr)
        z_fee = (fee_raw - FEE_MEAN) / FEE_STD
        z_use = (usage_raw - 40) / 12
        usage_raw = 40 + 12 * (r * z_fee + np.sqrt(1.0 - r * r) * z_use)  # 여전히 N(40, 12^2)
    fee = np.round(np.clip(fee_raw + fee_shift * FEE_STD, 15_000, 120_000), -2)
    usage = np.clip(usage_raw, 0, 200)
    logit = (
        -0.3
        + shift
        + fee_coef * (fee - 55_000) / 15_000
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


def numeric_matrix(table):
    return np.column_stack([table[k] for k in NUMERIC])


def dataset_hash(table):
    """3장과 같은 규칙: 열 이름 정렬 -> 열마다 이름, dtype, 모양, 원시 바이트(little-endian)."""
    h = hashlib.sha256()
    for name in sorted(table):
        a = np.ascontiguousarray(table[name])
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        h.update(name.encode())
        h.update(b"\x00" + a.dtype.str.encode() + b"\x00" + repr(a.shape).encode() + b"\x00")
        h.update(a.tobytes())
    return h.hexdigest()
