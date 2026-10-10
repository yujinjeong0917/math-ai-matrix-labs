"""8장 PyTorch 대응 구현: 7장 pipelines_ch07_model_torch.py(6장에서 복사한 것)를 다시 복사했다.

이 장에서는 몸통을 바꾼 학습이 정확도는 같아도 다른 모델 파일을 만든다는 것, 그래서 리니지에 impl을 남겨야 한다는 것을 본다.
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
