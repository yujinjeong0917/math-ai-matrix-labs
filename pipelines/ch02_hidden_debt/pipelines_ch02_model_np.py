"""2장 NumPy 최소 구현: 전처리(표준화 + 원-핫)와 로지스틱 회귀.

이 파일에서 '모델 코드'라고 부를 수 있는 건 fit_logistic, predict_proba 두 함수뿐이다.
나머지(전처리, 검사, 데이터 생성, 테스트)가 시스템의 대부분을 차지한다(experiments.py E6에서 줄 수를 센다).
"""

import numpy as np

from pipelines_ch02_data import CATEGORICAL, NUMERIC

N_LEVELS = {"plan": 3, "region": 4}


def fit_preprocessor(batch):
    """학습 표에서 수치 열의 평균·표준편차를 기억한다. 서빙 때도 이 값을 그대로 쓴다."""
    return {
        "mean": {k: float(batch[k].mean()) for k in NUMERIC},
        "std": {k: float(batch[k].std()) for k in NUMERIC},
    }


def transform(batch, prep):
    """z = (x - 학습 평균) / 학습 표준편차, 범주 코드는 원-핫. 반환: (n, 6 + 3 + 4)."""
    cols = [(batch[k] - prep["mean"][k]) / prep["std"][k] for k in NUMERIC]
    n = len(batch[NUMERIC[0]])
    for k in CATEGORICAL:
        onehot = np.zeros((n, N_LEVELS[k]))
        onehot[np.arange(n), batch[k]] = 1.0  # 코드 2는 '세 번째 칸'일 뿐, 그 코드가 무슨 뜻인지는 모른다
        cols.extend(onehot.T)
    return np.column_stack(cols)


# ---- 모델 코드 시작 ----
def fit_logistic(X, y, lr=0.5, steps=400, l2=1e-3):
    """전체 배치 경사하강법. 초기값 0, 고정된 단계 수라 같은 입력이면 같은 결과가 나온다."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))
# ---- 모델 코드 끝 ----


def accuracy(batch, y, model):
    prep, w, b = model
    return float(np.mean((predict_proba(transform(batch, prep), w, b) >= 0.5) == y))


def train(batch, y):
    prep = fit_preprocessor(batch)
    w, b = fit_logistic(transform(batch, prep), y)
    return prep, w, b


class ContractError(ValueError):
    pass


EXPECTED_LABELED = {"monthly_fee_krw"}  # 단위를 열 이름에 적은 계약


def labeled_to_codes(batch, plan_labels, region_labels):
    """'뜻이 담긴 계약'을 받는 쪽: 글자 라벨을 우리 코드표로 직접 바꾸고, 열 이름의 단위를 확인한다."""
    missing = EXPECTED_LABELED - set(batch)
    if missing:
        raise ContractError(f"계약에 없는 열 구성: {sorted(missing)} 없음, 받은 열 {sorted(batch)}")
    out = {k: batch[k] for k in NUMERIC if k != "monthly_fee"}
    out["monthly_fee"] = batch["monthly_fee_krw"]
    for k, labels in (("plan", plan_labels), ("region", region_labels)):
        unknown = set(batch[k]) - set(labels)
        if unknown:
            raise ContractError(f"{k}에 모르는 라벨 {sorted(unknown)}")
        index = {lab: i for i, lab in enumerate(labels)}
        out[k] = np.array([index[v] for v in batch[k]], dtype=np.int64)
    return out
