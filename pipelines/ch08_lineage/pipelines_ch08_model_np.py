"""8장 NumPy 모델: 7장 pipelines_ch07_model_np.py(6장·4장에서 이어 복사한 것)를 다시 복사했다.

표준화 + 원-핫 + 로지스틱 회귀. 초기값 0, 고정된 단계 수라 같은 입력이면 비트까지 같은 결과가 나온다.
"""

import numpy as np

from pipelines_ch08_data import CATEGORICAL, LABEL, NUMERIC


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


def train(table, fit=fit_logistic, steps=400):
    prep = fit_preprocessor(table)
    w, b = fit(transform(table, prep), table[LABEL], steps=steps)
    return {"prep": prep, "w": [float(v) for v in w], "b": float(b)}


def evaluate(model, table, threshold=0.5):
    p = predict_proba(transform(table, model["prep"]), np.array(model["w"]), model["b"])
    pred = (p >= threshold).astype(float)
    y = table[LABEL]
    return {
        "accuracy": float(np.mean(pred == y)),
        "churn_rate": float(np.mean(y)),
        "predicted_churn_rate": float(np.mean(pred)),
        "n": int(len(y)),
    }
