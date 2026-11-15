"""강화학습 2장 PyTorch 대응 구현. 같은 P, R, V에서 NumPy 백업과 float64로 atol=1e-12 안에서 같아야 한다.

식과 모양은 NumPy 판(rl_ch02_mdp_np.py)과 같다: P, R은 (S, A, S'), V는 (S,), pi는 (S, A) 텐서.
할인율 gamma만 인자로 받고, 탐험 파라미터와 학습률은 없다.
"""

import torch


def q_from_v(V, P, R, gamma):
    # 식: Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]
    return torch.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_expectation_backup(V, pi, P, R, gamma):
    # 식: V(s) <- sum_a pi(a|s) Q(s,a)  (벨만 기대방정식 백업)
    return (pi * q_from_v(V, P, R, gamma)).sum(dim=1)


def bellman_optimality_backup(V, P, R, gamma):
    # 식: V(s) <- max_a Q(s,a)  (벨만 최적방정식 백업). torch의 max(dim)은 (values, indices)를 돌려줘 .values만 쓴다
    return q_from_v(V, P, R, gamma).max(dim=1).values
