"""8장 데이터: 7장(pipelines/ch07_kfp/pipelines_ch07_data.py)의 고객 이탈 생성기와
3장(pipelines/ch03_tracking/pipelines_ch03_hashing.py)의 dataset_hash를 복사했다.

장끼리 동시에 작성되므로 import하지 않고 필요한 만큼만 복사했다.

이 장에서 새로 넣은 것은 source_version 하나다. 원천 테이블을 뽑는 규칙이 바뀐 상황을 흉내 낸다.
- "v1": 7장과 같은 표 그대로.
- "v2": 데이터 팀이 추출 규칙을 고쳐 가입 3개월 미만 고객을 뺀다. 같은 seed라도 행 수와 내용이 달라진다.
누가 이 값을 정하느냐(이미지 안의 기본값, 환경 변수, 파이프라인 파라미터)가 이 장의 캐시 함정이다.
"""

import hashlib

import numpy as np

NUMERIC = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
CATEGORICAL = {"plan": 3, "region": 4}
LABEL = "churned"
SOURCE_VERSIONS = ("v1", "v2")


def make_table(data_seed, n_rows, shift=0.0, source_version="v1"):
    if source_version not in SOURCE_VERSIONS:
        raise ValueError(f"모르는 source_version: {source_version!r}")
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
    table = {
        "tenure_months": tenure, "monthly_fee": fee, "support_calls": calls, "usage_hours": usage,
        "age": age, "discount_rate": discount, "plan": plan, "region": region, LABEL: churned,
    }
    if source_version == "v2":
        keep = tenure >= 3
        table = {k: v[keep] for k, v in table.items()}
    return table


def split(table, n_train, split_seed):
    n = len(table[LABEL])
    order = np.random.default_rng(split_seed).permutation(n)
    tr, va = order[:n_train], order[n_train:]
    return {k: v[tr] for k, v in table.items()}, {k: v[va] for k, v in table.items()}


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
