"""강화학습 10장 PyTorch 대응 구현. float64.

NumPy 쪽은 log pi의 기울기 (a - p) phi 를 손으로 적었다. 여기서는 손실
    loss = -sum_t (G_t - b_t) * log pi(a_t | s_t)
를 만들고 autograd로 미분한다. 같은 판 묶음에서 두 기울기가 rtol=1e-9 안에서 같아야 한다.
(G_t - b_t)는 상수로 취급한다(.detach 대신 텐서를 requires_grad 없이 만든다).
"""

import numpy as np
import torch


def log_pi(theta, Phi, A):
    """행동 두 개짜리 소프트맥스 정책의 log pi(a|s). 선호도 [0, theta.phi]에 log_softmax."""
    h = Phi @ theta
    logits = torch.stack([torch.zeros_like(h), h], dim=-1)
    return torch.log_softmax(logits, dim=-1).gather(-1, A.unsqueeze(-1)).squeeze(-1)


def pg_grad(theta_np, Phi_np, A_np, weights_np):
    """sum_t weights_t * d/dtheta log pi(a_t|s_t) 를 autograd로. weights_t = G_t - b_t."""
    theta = torch.tensor(np.asarray(theta_np, dtype=float), dtype=torch.float64, requires_grad=True)
    Phi = torch.tensor(np.asarray(Phi_np, dtype=float), dtype=torch.float64)
    A = torch.tensor(np.asarray(A_np), dtype=torch.long)
    W = torch.tensor(np.asarray(weights_np, dtype=float), dtype=torch.float64)
    loss = -(log_pi(theta, Phi, A) * W).sum()
    loss.backward()
    return -theta.grad.numpy()  # 손실의 기울기에 - 를 붙이면 J를 올리는 방향


def returns(R, gamma):
    """G_t를 텐서로. 뒤에서부터 누적."""
    R = torch.as_tensor(R, dtype=torch.float64)
    G = torch.zeros_like(R)
    g = torch.tensor(0.0, dtype=torch.float64)
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + gamma * g
        G[t] = g
    return G
