"""2장 PyTorch 대응 구현: NumPy fit_logistic과 같은 입력에서 같은 가중치가 나와야 한다.

경사는 손으로 쓰지 않고 autograd로 구한다. 손실은 평균 로그 손실 + (l2/2)·||w||² 이라서,
그 기울기가 NumPy 쪽 X.T @ (p - y) / n + l2 * w 와 같다.
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


def predict_proba(X, w, b):
    return torch.sigmoid(torch.as_tensor(X, dtype=torch.float64) @ torch.as_tensor(w) + b).numpy()
