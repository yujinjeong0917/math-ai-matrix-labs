"""3장 NumPy 모델: 2장 pipelines_ch02_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다.

이번 장에서는 lr, steps, l2를 하이퍼파라미터로 바꿔 가며 12번 돌린다.
"""

import numpy as np

from pipelines_ch03_data import CATEGORICAL, LABEL, NUMERIC


def fit_preprocessor(table):
    return {"mean": {k: float(table[k].mean()) for k in NUMERIC}, "std": {k: float(table[k].std()) for k in NUMERIC}}


def transform(table, prep):
    cols = [(table[k] - prep["mean"][k]) / prep["std"][k] for k in NUMERIC]
    n = len(table[NUMERIC[0]])
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
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def train_and_eval(train, valid, lr, steps, l2, fit=fit_logistic):
    """학습 표로 전처리와 가중치를 정하고, 검증 표 정확도를 낸다."""
    prep = fit_preprocessor(train)
    w, b = fit(transform(train, prep), train[LABEL], lr=lr, steps=steps, l2=l2)
    acc = float(np.mean((predict_proba(transform(valid, prep), w, b) >= 0.5) == valid[LABEL]))
    return {"prep": prep, "w": w, "b": b}, {"val_acc": acc}
