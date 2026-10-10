"""모델 평가 8장의 PyTorch 대응 구현.

이 실습 저장소는 numpy·torch·pytest만 쓰므로(scikit-learn 없음), 설계도의 sklearn 대조 대신
같은 식을 텐서 연산으로 다시 써서 NumPy 구현과 같은 입력에서 값을 대조한다.
혼동행렬은 one-hot 행렬곱으로, ROC-AUC는 모든 (양성, 음성) 쌍을 한 번에 비교하는 방식으로(NumPy는 사다리꼴),
AP는 정렬 + 누적합 + 동점 덩어리 끝 고르기로 계산한다.
"""

import torch

torch.set_num_threads(2)
DT = torch.float64


def confusion(y, yhat):
    """[[TP, FN], [FP, TN]]. 행 = 실제(양성, 음성), 열 = 예측(양성, 음성)."""
    y = torch.as_tensor(y).long(); yhat = torch.as_tensor(yhat).long()
    oy = torch.nn.functional.one_hot(1 - y, 2).to(DT)      # 0열 = 양성
    op = torch.nn.functional.one_hot(1 - yhat, 2).to(DT)
    return oy.T @ op


def f_beta(cm, beta=1.0):
    cm = torch.as_tensor(cm, dtype=DT)
    tp, fn, fp = cm[0, 0], cm[0, 1], cm[1, 0]
    b2 = beta * beta
    return ((1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)).item()


def mcc(cm):
    cm = torch.as_tensor(cm, dtype=DT)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den).item()  # 분모 0이면 nan (NumPy 기본 정책과 같다)


def roc_auc_pairs(y, s):
    """모든 (양성, 음성) 쌍을 비교. 동점은 1/2. 메모리가 P*N에 비례하니 작은 입력용."""
    y = torch.as_tensor(y).bool(); s = torch.as_tensor(s, dtype=DT)
    d = s[y][:, None] - s[~y][None, :]
    return ((d > 0).to(DT).mean() + 0.5 * (d == 0).to(DT).mean()).item()


def average_precision(y, s):
    y = torch.as_tensor(y).to(DT); s = torch.as_tensor(s, dtype=DT)
    s_sorted, order = torch.sort(s, descending=True, stable=True)
    y_sorted = y[order]
    tp_all = torch.cumsum(y_sorted, 0)
    fp_all = torch.cumsum(1 - y_sorted, 0)
    change = torch.ones_like(s_sorted, dtype=torch.bool)
    change[:-1] = s_sorted[1:] != s_sorted[:-1]                 # 동점 덩어리의 마지막 위치
    tp, fp = tp_all[change], fp_all[change]
    r = tp / tp[-1]
    p = tp / (tp + fp)
    dr = torch.diff(torch.cat([torch.zeros(1, dtype=DT), r]))
    return (dr * p).sum().item()


def brier(y, prob):
    y = torch.as_tensor(y, dtype=DT); prob = torch.as_tensor(prob, dtype=DT)
    return ((prob - y) ** 2).mean().item()
