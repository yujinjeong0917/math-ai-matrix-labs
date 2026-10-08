"""강화학습 4장 PyTorch 대응 구현. 같은 P, R에서 NumPy와 최종 정책이 같고 가치가 float64로 atol=1e-10 안이어야 한다.

동률 처리는 NumPy의 tie="first"와 같게 맞췄다(최댓값과 1e-9 안인 행동 중 번호가 가장 작은 것).
"""

import torch

TIE_TOL = 1e-9


def q_from_v(V, P, R, gamma):
    return torch.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def greedy_first(V, P, R, gamma, tol=TIE_TOL):
    Q = q_from_v(V, P, R, gamma)
    best = Q >= Q.max(dim=1, keepdim=True).values - tol
    return best.to(torch.int8).argmax(dim=1)  # 첫 번째 True


def evaluate_exact(pi, P, R, gamma):
    idx = torch.arange(P.shape[0])
    Ppi = P[idx, pi]
    rpi = (Ppi * R[idx, pi]).sum(dim=1)
    return torch.linalg.solve(torch.eye(P.shape[0], dtype=P.dtype) - gamma * Ppi, rpi)


def policy_iteration(P, R, gamma, max_iter=1000):
    pi = torch.zeros(P.shape[0], dtype=torch.long)
    for it in range(1, max_iter + 1):
        V = evaluate_exact(pi, P, R, gamma)
        new = greedy_first(V, P, R, gamma)
        if torch.equal(new, pi):
            return pi, V, it
        pi = new
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def value_iteration(P, R, gamma, theta=1e-10, max_iter=1_000_000):
    V = torch.zeros(P.shape[0], dtype=P.dtype)
    for k in range(1, max_iter + 1):
        V_new = q_from_v(V, P, R, gamma).max(dim=1).values
        d = float((V_new - V).abs().max())
        V = V_new
        if d < theta:
            return V, k
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def modified_policy_iteration(P, R, gamma, k, theta=1e-10, max_iter=1_000_000):
    idx = torch.arange(P.shape[0])
    V = torch.zeros(P.shape[0], dtype=P.dtype)
    for it in range(1, max_iter + 1):
        Q = q_from_v(V, P, R, gamma)
        TV, pi = Q.max(dim=1)
        d = float((TV - V).abs().max())
        Ppi = P[idx, pi]
        rpi = (Ppi * R[idx, pi]).sum(dim=1)
        V = TV
        for _ in range(k):
            V = rpi + gamma * Ppi @ V
        if d < theta:
            return pi, V, it
    raise RuntimeError("max_iter 안에 멈추지 않았어요")
