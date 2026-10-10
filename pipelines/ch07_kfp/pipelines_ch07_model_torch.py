"""7장 PyTorch 대응 구현: 6장 pipelines_ch06_model_torch.py를 복사했다.

이 장에서는 train 컴포넌트의 몸통만 이것으로 바꿔 끼워도, 파이프라인 그래프와 평가 숫자가 그대로인지 본다.
"""

import torch


def fit_logistic(X, y, lr=0.5, steps=400, l2=1e-3):
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * l2 * (w @ w)
        gw, gb = torch.autograd.grad(loss, (w, b))
        with torch.no_grad():
            w -= lr * gw
            b -= lr * gb
    return w.detach().numpy(), float(b.detach())
