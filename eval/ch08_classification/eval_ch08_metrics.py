"""모델 평가 8장. 이진 분류 지표의 NumPy 최소 구현.

혼동행렬 배치는 Chicco·Jurman(2020) 표 1과 같다.
    M = [[TP, FN],
         [FP, TN]]     (행 = 실제 양성/음성, 열 = 예측 양성/음성)

정의되지 않는 값(0/0)은 기본으로 nan을 돌려준다. 숨기지 않고 보이게 하려는 선택이다.
MCC만 policy="chicco"로 Chicco·Jurman(2020)이 극한으로 채운 값을 고를 수 있다.
  - 0이 아닌 칸이 하나뿐: 모두 맞혔으면 +1, 모두 틀렸으면 -1
  - 한 행이나 한 열이 통째로 0(나머지 두 칸은 0이 아님): 0
"""

import numpy as np


def _arr(a):
    return np.asarray(a)


def confusion(y, yhat):
    """y, yhat은 0/1. 돌려주는 값은 [[TP, FN], [FP, TN]] (int64)."""
    y, yhat = _arr(y).astype(bool), _arr(yhat).astype(bool)
    tp = int(np.sum(y & yhat)); fn = int(np.sum(y & ~yhat))
    fp = int(np.sum(~y & yhat)); tn = int(np.sum(~y & ~yhat))
    return np.array([[tp, fn], [fp, tn]], dtype=np.int64)


def cells(cm):
    cm = _arr(cm)
    return int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])  # TP, FN, FP, TN


def _div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")


def accuracy(cm):
    tp, fn, fp, tn = cells(cm)
    return _div(tp + tn, tp + fn + fp + tn)


def precision(cm):
    """TP / (TP + FP). 양성이라고 한 번도 말하지 않으면 0/0이라 nan."""
    tp, fn, fp, tn = cells(cm)
    return _div(tp, tp + fp)


def recall(cm):
    """TP / (TP + FN). 실제 양성이 없으면 nan."""
    tp, fn, fp, tn = cells(cm)
    return _div(tp, tp + fn)


def f_beta(cm, beta=1.0):
    """(1 + b^2) TP / ((1 + b^2) TP + b^2 FN + FP).
    P, R로 쓴 꼴 (1+b^2)PR/(b^2 P + R)과 같은 값이지만, 이 꼴은 TP = 0이어도 FP나 FN이 있으면 0으로 정의된다.
    TN은 식에 들어가지 않는다."""
    tp, fn, fp, tn = cells(cm)
    b2 = beta * beta
    return _div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp)


def f1(cm):
    return f_beta(cm, 1.0)


def van_rijsbergen_e(p, r, alpha):
    """van Rijsbergen(1979) 7장의 효과 척도 E = 1 - 1 / (alpha/P + (1 - alpha)/R).
    alpha = 1/(beta^2 + 1)로 바꾸면 1 - E = F_beta."""
    return 1.0 - 1.0 / (alpha / p + (1.0 - alpha) / r)


def mcc(cm, policy="nan"):
    """(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
    policy="nan"   : 분모가 0이면 nan
    policy="chicco": Chicco·Jurman(2020)의 극한값으로 채운다(모듈 설명 참고)"""
    tp, fn, fp, tn = cells(cm)
    den = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    if den > 0:
        return (tp * tn - fp * fn) / np.sqrt(den)
    if policy == "nan":
        return float("nan")
    if policy != "chicco":
        raise ValueError(policy)
    nonzero = [v for v in (tp, fn, fp, tn) if v != 0]
    if len(nonzero) == 1:
        return 1.0 if (tp or tn) else -1.0
    return 0.0


def all_scores(cm, policy="nan"):
    return {"accuracy": accuracy(cm), "precision": precision(cm), "recall": recall(cm), "f1": f1(cm),
            "mcc": mcc(cm, policy), "mcc_chicco": mcc(cm, "chicco")}


# ---------- 임계값을 훑는 곡선 ----------

def _sorted_counts(y, s):
    """점수를 큰 것부터 정렬하고, 같은 점수는 한 덩어리로 묶어 덩어리 끝마다 누적 TP, FP를 돌려준다."""
    y, s = _arr(y).astype(bool), _arr(s).astype(float)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]  # 같은 점수 덩어리의 마지막 위치
    tp = np.cumsum(y_sorted)[last]
    fp = np.cumsum(~y_sorted)[last]
    return tp.astype(float), fp.astype(float), s_sorted[last]


def roc_curve_np(y, s):
    """(FPR, TPR, 임계값). 맨 앞에 (0, 0)을 붙인다."""
    tp, fp, thr = _sorted_counts(y, s)
    P, N = tp[-1], fp[-1]
    return np.r_[0.0, fp / N], np.r_[0.0, tp / P], np.r_[np.inf, thr]


def roc_auc_np(y, s):
    """ROC 곡선 아래 넓이(사다리꼴). 같은 점수 덩어리는 대각선으로 이어서, 동점을 1/2로 세는 것과 같다."""
    fpr, tpr, _ = roc_curve_np(y, s)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_pairwise(y, s):
    """양성 하나와 음성 하나를 짝지은 모든 쌍에서 '양성 점수가 더 크다'의 비율(동점은 1/2). 맨-휘트니 U / (P N)."""
    y, s = _arr(y).astype(bool), _arr(s).astype(float)
    sp, sn = s[y], s[~y]
    sn_sorted = np.sort(sn)
    less = np.searchsorted(sn_sorted, sp, side="left")
    leq = np.searchsorted(sn_sorted, sp, side="right")
    return float(np.sum(less + 0.5 * (leq - less)) / (len(sp) * len(sn)))


def pr_curve_np(y, s):
    """(재현율, 정밀도, 임계값). 같은 점수 덩어리마다 한 점."""
    tp, fp, thr = _sorted_counts(y, s)
    return tp / tp[-1], tp / (tp + fp), thr


def average_precision_np(y, s):
    """AP = sum_n (R_n - R_{n-1}) P_n, R_0 = 0. 계단식 정의(scikit-learn average_precision_score 문서와 같은 식)."""
    r, p, _ = pr_curve_np(y, s)
    return float(np.sum(np.diff(np.r_[0.0, r]) * p))


def pr_auc_trapezoid(y, s):
    """비교용: PR 점들을 직선으로 이은 넓이. 첫 점 앞은 (0, 첫 정밀도)에서 시작한다. AP와 다를 수 있다."""
    r, p, _ = pr_curve_np(y, s)
    r, p = np.r_[0.0, r], np.r_[p[0], p]
    return float(np.sum(np.diff(r) * (p[1:] + p[:-1]) / 2.0))


def brier(y, prob):
    """이진 브라이어 점수 (1/n) sum (p - y)^2. Brier(1950)의 원래 정의(두 범주를 모두 더해 0~2)의 절반."""
    y, prob = _arr(y).astype(float), _arr(prob).astype(float)
    return float(np.mean((prob - y) ** 2))
