"""9장 NumPy 모델: 6장 pipelines_ch06_model_np.py(4장에서 복사한 것)의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다.
쓰는 수치 열은 6개(NUMERIC)로 고정했다."""

import numpy as np

from pipelines_ch09_data import CATEGORICAL, LABEL, NUMERIC


def fit_preprocessor(table, numeric=NUMERIC):
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
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -500, 500)))


def train(table, fit=fit_logistic):
    prep = fit_preprocessor(table)
    w, b = fit(transform(table, prep), table[LABEL])
    return {"prep": prep, "w": w, "b": b}


def predict(model, table):
    return (predict_proba(transform(table, model["prep"]), model["w"], model["b"]) >= 0.5).astype(float)


def accuracy(model, table):
    return float(np.mean(predict(model, table) == table[LABEL]))


def model_bytes(model):
    """모델을 바이트로 굳힌다. 이 바이트의 sha256을 '이미지 다이제스트' 대신 쓴다."""
    parts = [np.asarray(model["w"], dtype="<f8").tobytes(), np.asarray([model["b"]], dtype="<f8").tobytes()]
    for k in model["prep"]["numeric"]:
        parts.append(np.asarray([model["prep"]["mean"][k], model["prep"]["std"][k]], dtype="<f8").tobytes())
    return b"".join(parts)
