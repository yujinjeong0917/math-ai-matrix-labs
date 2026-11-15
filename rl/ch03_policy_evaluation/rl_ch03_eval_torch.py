"""강화학습 3장 PyTorch 대응 구현. 같은 P^pi, r^pi에서 NumPy와 float64로 atol=1e-10 안에서 같아야 한다."""

import torch


def evaluate_direct(P_pi, r_pi, gamma):
    """NumPy evaluate_direct와 같은 연립방정식 풀이. P_pi: (S, S), r_pi: (S,), 반환 V: (S,)."""
    S = P_pi.shape[0]
    # 식: (I - gamma P^pi) V = r^pi. dtype을 P_pi에 맞춰 float64 비교(atol=1e-10)가 가능하게 한다
    return torch.linalg.solve(torch.eye(S, dtype=P_pi.dtype) - gamma * P_pi, r_pi)


def evaluate_iterative(P_pi, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000):
    """NumPy evaluate_iterative의 대응 구현(두 배열 야코비식). 반환: (V, 스윕 수). 오차 기록 errs는 돌려주지 않는다."""
    V = torch.zeros_like(r_pi)
    for k in range(1, max_sweeps + 1):
        # 식: V_{k+1} = r^pi + gamma P^pi V_k  (벨만 기대 갱신)
        V_new = r_pi + gamma * (P_pi @ V)
        # V_true가 있으면 정답과의 오차, 없으면 앞뒤 스윕 차이(무한 노름)
        ref = V_true if V_true is not None else V
        e = float((V_new - ref).abs().max())
        V = V_new
        # 수렴 허용오차: e < tol (NumPy판과 같은 스윕 수에서 멈춰야 테스트의 kn == kt가 성립)
        if e < tol:
            return V, k
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")
