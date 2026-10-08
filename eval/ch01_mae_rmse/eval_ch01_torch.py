"""모델 평가 1장의 PyTorch 대응 구현.

1) torch 내장 손실(l1_loss, mse_loss)로 같은 지표를 계산해 NumPy와 대조한다.
2) 상수 c 하나를 경사하강으로 학습시켜, L1 손실은 중앙값으로, L2 손실은 평균으로 가는지 본다.
"""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)


def mae(y, yhat):
    return F.l1_loss(yhat, y).item()


def rmse(y, yhat):
    return torch.sqrt(F.mse_loss(yhat, y)).item()


def fit_constant(y, loss, steps=4000, lr=0.05):
    """모든 경우를 상수 c로 예측하는 '가장 단순한 모델'을 학습한다.

    loss="l1"이면 기울기가 (c보다 작은 y의 개수 - c보다 큰 y의 개수)/n 이라 개수가 반반인 곳(중앙값)에서 멈추고,
    loss="l2"면 기울기가 2(c - 평균)이라 평균에서 멈춘다.
    """
    y = torch.as_tensor(y, dtype=torch.float64)
    c = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([c], lr=lr)
    fn = F.l1_loss if loss == "l1" else F.mse_loss
    for t in range(steps):
        if loss == "l1":  # L1 기울기는 크기가 일정해서, 학습률을 줄여야 중앙값 근처에서 진동이 잦아든다
            for g in opt.param_groups:
                g["lr"] = lr * (1 - t / steps) + 1e-4
        opt.zero_grad()
        fn(c.expand_as(y), y).backward()
        opt.step()
    return float(c.detach())
