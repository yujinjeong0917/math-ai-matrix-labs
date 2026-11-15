"""강화학습 4장 PyTorch 대응 구현. 같은 P, R에서 NumPy와 최종 정책이 같고 가치가 float64로 atol=1e-10 안이어야 한다.

동률 처리는 NumPy의 tie="first"와 같게 맞췄다(최댓값과 1e-9 안인 행동 중 번호가 가장 작은 것).
"""

import torch

TIE_TOL = 1e-9


def q_from_v(V, P, R, gamma):
    """NumPy q_from_v의 대응. P, R: (S, A, S), V: (S,), 반환 Q: (S, A)."""
    # 식: Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]
    return torch.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def greedy_first(V, P, R, gamma, tol=TIE_TOL):
    """식 pi'(s) = argmax_a Q(s,a). 최댓값과 tol 이내인 행동 중 번호가 가장 작은 것(NumPy tie="first")."""
    Q = q_from_v(V, P, R, gamma)
    best = Q >= Q.max(dim=1, keepdim=True).values - tol  # 동률 후보 (S, A) 불리언
    # 불리언에 argmax를 직접 쓰지 않고 int8로 바꾼다. 같은 값(1)이 여럿일 때 가장 앞 인덱스를 돌려주는지가
    # 백엔드 구현에 달려 있어 NumPy와 같은 선택을 보장하려는 목적으로 추정(확인 필요)
    return best.to(torch.int8).argmax(dim=1)  # 첫 번째 True


def evaluate_exact(pi, P, R, gamma):
    """NumPy evaluate_exact의 대응. 정책 pi: (S,) 행동 번호, 반환 V^pi: (S,)."""
    idx = torch.arange(P.shape[0])
    Ppi = P[idx, pi]  # P^pi[s,s'] = P(s'|s, pi(s)), (S, S)
    rpi = (Ppi * R[idx, pi]).sum(dim=1)  # 식: r^pi(s) = sum_s' P(s'|s,pi(s)) R(s,pi(s),s')
    # 식: (I - gamma P^pi) V^pi = r^pi
    return torch.linalg.solve(torch.eye(P.shape[0], dtype=P.dtype) - gamma * Ppi, rpi)


def policy_iteration(P, R, gamma, max_iter=1000):
    """정책 반복. 반환: (pi, V, 반복 수). 시작 정책은 모든 칸 행동 0(위)."""
    pi = torch.zeros(P.shape[0], dtype=torch.long)
    for it in range(1, max_iter + 1):
        V = evaluate_exact(pi, P, R, gamma)  # 평가: 식 (I - gamma P^pi) V = r^pi
        new = greedy_first(V, P, R, gamma)  # 개선: 식 pi'(s) = argmax_a Q^pi(s,a)
        # 종료 조건: 정책이 그대로면 멈춤(NumPy판의 changed == 0 과 같은 조건)
        if torch.equal(new, pi):
            return pi, V, it
        pi = new
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def value_iteration(P, R, gamma, theta=1e-10, max_iter=1_000_000):
    """가치 반복. 반환: (V, 스윕 수). 앞뒤 차이 < theta 에서 멈추고, 못 멈추면 RuntimeError."""
    V = torch.zeros(P.shape[0], dtype=P.dtype)
    for k in range(1, max_iter + 1):
        # 식: V_{k+1}(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V_k(s')]  (벨만 최적 갱신)
        V_new = q_from_v(V, P, R, gamma).max(dim=1).values
        d = float((V_new - V).abs().max())  # 앞뒤 스윕 차이(무한 노름)
        V = V_new
        # 수렴 허용오차: d < theta (NumPy value_iteration과 같은 기준)
        if d < theta:
            return V, k
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def modified_policy_iteration(P, R, gamma, k, theta=1e-10, max_iter=1_000_000):
    """수정 정책 반복(NumPy modified_policy_iteration의 대응). 반환: (pi, V, 반복 수). 백업 수는 세지 않는다."""
    idx = torch.arange(P.shape[0])
    V = torch.zeros(P.shape[0], dtype=P.dtype)
    for it in range(1, max_iter + 1):
        Q = q_from_v(V, P, R, gamma)
        TV, pi = Q.max(dim=1)  # 식: TV(s) = max_a Q(s,a), pi(s) = argmax_a Q(s,a)
        d = float((TV - V).abs().max())  # 식: ||T V - V||_inf
        Ppi = P[idx, pi]
        rpi = (Ppi * R[idx, pi]).sum(dim=1)
        V = TV
        for _ in range(k):
            # 식: V <- r^pi + gamma P^pi V  (평가 스윕 k번, k=0이면 가치 반복)
            V = rpi + gamma * Ppi @ V
        # 수렴 허용오차: d < theta (평가 스윕 뒤에 검사)
        if d < theta:
            return pi, V, it
    raise RuntimeError("max_iter 안에 멈추지 않았어요")
