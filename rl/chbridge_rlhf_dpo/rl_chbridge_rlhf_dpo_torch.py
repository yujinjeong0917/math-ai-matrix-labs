"""강화학습 연결 장 PyTorch 대응 구현. 손으로 쓴 NumPy 기울기를 autograd와 대조하고,
같은 선호 쌍으로 DPO 학습을 PyTorch(torch.optim.Adam)로 다시 돌린다. 모두 float64."""

import math

import numpy as np
import torch

import rl_chbridge_rlhf_dpo_np as m

torch.set_default_dtype(torch.float64)


def _flat(Y):
    return torch.as_tensor(m._flat_index(Y), dtype=torch.long)


def logprob(theta, Y):
    """theta: (8, 7, 6) 텐서. log pi(y) = sum_t log_softmax(theta[t, prev])[y_t]."""
    ls = torch.log_softmax(theta, dim=-1).reshape(-1)
    return ls[_flat(Y)].sum(dim=1)


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta):
    z = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    return -torch.nn.functional.logsigmoid(z).mean()


def bt_loss(u, Yw, Yl, l2=1e-3):
    d = torch.as_tensor(m.counts(Yw) - m.counts(Yl))
    z = d @ u
    return -torch.nn.functional.logsigmoid(z).mean() + 0.5 * l2 * (u @ u)


def dpo_loss_theta(theta, Yw, Yl, beta):
    ref_w = torch.as_tensor(m.logprob_ref(Yw))
    ref_l = torch.as_tensor(m.logprob_ref(Yl))
    return dpo_loss(logprob(theta, Yw), logprob(theta, Yl), ref_w, ref_l, beta)


def train_dpo(Yw, Yl, beta, steps=300, lr=0.05):
    """NumPy `train_dpo`와 같은 순서(전체 묶음, Adam)로 학습한 theta를 돌려준다."""
    theta = torch.zeros((m.L, m.V + 1, m.V), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(steps):
        opt.zero_grad()
        loss = dpo_loss_theta(theta, Yw, Yl, beta)
        loss.backward()
        opt.step()
    return theta.detach().numpy()


def kl_regularized_objective(theta, u, beta):
    """전체 문장 공간에서 정확한 E_pi[r_hat] - beta KL(pi || pi_ref). 열거로 계산(정답 확인용)."""
    tab = m._enum_tables()
    lp = torch.log_softmax(theta, dim=-1).reshape(-1)[torch.as_tensor(tab["flat"], dtype=torch.long)].sum(dim=1)
    p = torch.exp(lp)
    rhat = torch.as_tensor(tab["counts"]) @ torch.as_tensor(u)
    return p @ rhat - beta * (p @ (lp + m.L * math.log(m.V)))


def to_np(x):
    return x.detach().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
