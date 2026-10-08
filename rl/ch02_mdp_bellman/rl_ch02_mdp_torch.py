"""강화학습 2장 PyTorch 대응 구현. 같은 P, R, V에서 NumPy 백업과 float64로 atol=1e-12 안에서 같아야 한다."""

import torch


def q_from_v(V, P, R, gamma):
    return torch.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_expectation_backup(V, pi, P, R, gamma):
    return (pi * q_from_v(V, P, R, gamma)).sum(dim=1)


def bellman_optimality_backup(V, P, R, gamma):
    return q_from_v(V, P, R, gamma).max(dim=1).values
