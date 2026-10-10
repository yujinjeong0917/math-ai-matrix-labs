"""6장 PyTorch 대응 구현: 5장 pipelines_ch05_model_torch.py(4장에서 복사한 것)를 다시 복사했다.

배포 판정이 모델 구현에 기대지 않는지 본다. 같은 특성 행렬에서 NumPy 구현과 같은 예측을 내면,
요청별 맞힘/틀림이 같아지고 배포 시뮬레이션의 숫자도 그대로다.
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
