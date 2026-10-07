"""2장 PyTorch 대응 구현.

n-gram에는 경사하강으로 학습할 파라미터가 없어서, 이 장의 PyTorch 대응은 얇다.
NumPy 모델이 낸 분포 행을 받아 perplexity만 `F.cross_entropy`로 다시 계산하고,
NumPy가 정답 토큰 확률만 모아 낸 값과 수치가 같은지 대조한다.

`cross_entropy(logits, y)`는 안에서 log_softmax를 한 번 더 한다. 각 행이 정규화된 분포 p라면
log_softmax(log p) = log p - log Σ p = log p 이므로 값이 그대로다. 그래서 이 대조는
"모든 행의 합이 1인가"도 함께 검사한다. 확률 0이 있는 MLE는 log 0 = -inf라 대조에서 뺀다.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def perplexity_torch(rows, targets):
    """rows: (T, V) 확률 분포, targets: (T,) 정답 id. float64로 계산한다."""
    logp = torch.log(torch.as_tensor(np.asarray(rows), dtype=torch.float64))
    y = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    return math.exp(float(F.cross_entropy(logp, y, reduction="mean")))


def rows_and_targets(model, pos_iter, limit=None):
    """NumPy 모델에서 (문맥별 전체 분포, 정답) 배열을 만든다."""
    rows, ys = [], []
    for i, (h, w) in enumerate(pos_iter):
        if limit is not None and i >= limit:
            break
        rows.append(model.dist(h))
        ys.append(w)
    return np.stack(rows), np.array(ys)
