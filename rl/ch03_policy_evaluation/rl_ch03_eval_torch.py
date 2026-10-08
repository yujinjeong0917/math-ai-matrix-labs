"""강화학습 3장 PyTorch 대응 구현. 같은 P^pi, r^pi에서 NumPy와 float64로 atol=1e-10 안에서 같아야 한다."""

import torch


def evaluate_direct(P_pi, r_pi, gamma):
    S = P_pi.shape[0]
    return torch.linalg.solve(torch.eye(S, dtype=P_pi.dtype) - gamma * P_pi, r_pi)


def evaluate_iterative(P_pi, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000):
    V = torch.zeros_like(r_pi)
    for k in range(1, max_sweeps + 1):
        V_new = r_pi + gamma * (P_pi @ V)
        ref = V_true if V_true is not None else V
        e = float((V_new - ref).abs().max())
        V = V_new
        if e < tol:
            return V, k
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")
