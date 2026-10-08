"""4장 NumPy 모델: 3장 pipelines_ch03_model_np.py의 표준화 + 원-핫 + 로지스틱 회귀를 복사했다.

이번 장에서 덧붙인 것은 서빙 경로 하나다(transform_serving).
- 학습 통계(평균·표준편차)를 float()로 감싸지 않고 NumPy 스칼라(np.float64) 그대로 둔다.
- 서빙 입력은 메모리를 아끼려고 float32로 들어온다.
이 두 가지가 만나면 NumPy 1.x와 2.x가 결과 타입을 다르게 정한다(NEP 50).
코드는 NumPy 1.26에서도 돌아가야 하므로 2.x 전용 문법을 쓰지 않는다.
"""

import numpy as np

from pipelines_ch04_data import CATEGORICAL, LABEL, NUMERIC


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
    prep = fit_preprocessor(train)
    w, b = fit(transform(train, prep), train[LABEL], lr=lr, steps=steps, l2=l2)
    acc = float(np.mean((predict_proba(transform(valid, prep), w, b) >= 0.5) == valid[LABEL]))
    return {"prep": prep, "w": w, "b": b}, {"val_acc": acc}


def serving_stats(table):
    """float()로 감싸지 않은 통계. table[k].mean()은 np.float64 스칼라를 돌려준다."""
    return {"mean": {k: table[k].mean() for k in NUMERIC}, "std": {k: table[k].std() for k in NUMERIC}}


def transform_serving(table32, stats):
    """float32 입력 열에서 np.float64 스칼라를 빼고 나눈다. 결과 타입: 1.x는 float32, 2.x는 float64."""
    cols = [(table32[k] - stats["mean"][k]) / stats["std"][k] for k in NUMERIC]
    return cols[0].dtype.name, np.column_stack(cols)
