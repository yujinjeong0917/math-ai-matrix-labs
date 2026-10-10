"""6장 NumPy 모델: 4장 pipelines_ch04_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다.

바뀐 점은 쓰는 수치 열 목록(numeric)을 인자로 받는 것 하나다. v1은 6열, v2는 usage_drop을 더한 7열을 쓴다.
"""

import numpy as np

from pipelines_ch06_data import CATEGORICAL, LABEL


def fit_preprocessor(table, numeric):
    return {
        "numeric": list(numeric),
        "mean": {k: float(table[k].mean()) for k in numeric},
        "std": {k: float(table[k].std()) for k in numeric},
    }


def transform(table, prep):
    cols = [(table[k] - prep["mean"][k]) / prep["std"][k] for k in prep["numeric"]]
    n = len(table[LABEL])
    for k, levels in CATEGORICAL.items():
        onehot = np.zeros((n, levels))
        onehot[np.arange(n), table[k]] = 1.0
        cols.extend(onehot.T)
    return np.column_stack(cols)


def fit_logistic(X, y, lr=0.5, steps=400, l2=1e-3):
    """전체 배치 경사하강법. 초기값 0, 고정된 단계 수라 같은 입력이면 비트까지 같은 결과가 나온다."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -500, 500)))


def train(table, numeric, fit=fit_logistic):
    prep = fit_preprocessor(table, numeric)
    w, b = fit(transform(table, prep), table[LABEL])
    return {"prep": prep, "w": w, "b": b}


def predict(model, table):
    """0/1 예측(이탈할 것 같다 = 1)."""
    return (predict_proba(transform(table, model["prep"]), model["w"], model["b"]) >= 0.5).astype(float)


def correct(model, table):
    """요청마다 맞혔으면 1, 틀렸으면 0."""
    return (predict(model, table) == table[LABEL]).astype(np.int64)
