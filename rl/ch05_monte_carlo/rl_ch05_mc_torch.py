"""강화학습 5장 PyTorch 대응 구현. 같은 에피소드 묶음에서 리턴, 첫 방문 MC 평균, 중요도 비율과
일반·가중 IS 추정값을 텐서로 계산한다. float64.

허용오차: 리턴과 MC 평균은 atol=1e-12. 중요도 비율은 곱하는 순서가 NumPy(앞에서부터 하나씩)와
torch.prod(내부 순서)에서 다를 수 있고, 값이 1e4를 넘기도 해서 상대오차 rtol=1e-12로 비교한다.
"""

import torch


def pad(episodes, gamma_dtype=torch.float64):
    """끝난 판만 골라 [B, T] 텐서로 채운다. 짧은 판의 뒤쪽은 mask=False."""
    eps = [e for e in episodes if e.done]
    B = len(eps)
    T = max((len(e.states) for e in eps), default=0)
    S = torch.zeros((B, T), dtype=torch.long)
    A = torch.zeros((B, T), dtype=torch.long)
    R = torch.zeros((B, T), dtype=gamma_dtype)
    M = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(eps):
        n = len(e.states)
        if n:
            S[i, :n] = torch.tensor(e.states)
            A[i, :n] = torch.tensor(e.actions)
            R[i, :n] = torch.tensor(e.rewards, dtype=gamma_dtype)
            M[i, :n] = True
    return S, A, R, M


def returns_batch(R, gamma):
    """G[:, t] = R[:, t] + gamma G[:, t+1]. 판 B개를 한꺼번에, 시간 방향으로만 반복."""
    G = torch.zeros_like(R)
    g = torch.zeros(R.shape[0], dtype=R.dtype)
    for t in range(R.shape[1] - 1, -1, -1):
        g = R[:, t] + gamma * g
        G[:, t] = g
    return G


def first_visit_mask(S, M, n_states):
    """판마다 각 상태를 처음 방문한 칸만 True."""
    B, T = S.shape
    first = torch.zeros_like(M)
    seen = torch.zeros((B, n_states), dtype=torch.bool)
    rows = torch.arange(B)
    for t in range(T):
        s = S[:, t]
        new = M[:, t] & ~seen[rows, s]
        first[:, t] = new
        seen[rows, s] = seen[rows, s] | M[:, t]
    return first


def mc_first_visit(episodes, gamma, n_states):
    S, A, R, M = pad(episodes)
    G = returns_batch(R, gamma)
    fv = first_visit_mask(S, M, n_states)
    total = torch.zeros(n_states, dtype=G.dtype).index_add_(0, S[fv], G[fv])
    N = torch.zeros(n_states, dtype=torch.long).index_add_(0, S[fv], torch.ones_like(S[fv]))
    V = torch.where(N > 0, total / N.clamp(min=1), torch.zeros_like(total))
    return V, N


def importance_ratios(episodes, pi, b):
    S, A, R, M = pad(episodes)
    ratio = pi[S, A] / b[S, A]
    ratio = torch.where(M, ratio, torch.ones_like(ratio))
    return ratio.prod(dim=1)


def off_policy_is(episodes, pi, b, gamma):
    """시작 칸(t=0) 기준 일반·가중 IS 추정값."""
    S, A, R, M = pad(episodes)
    G0 = returns_batch(R, gamma)[:, 0] if R.shape[1] else torch.zeros(R.shape[0], dtype=R.dtype)
    rho = importance_ratios(episodes, pi, b)
    num = (rho * G0).sum()
    return num / len(rho), num / rho.sum()


def is_estimates_prefix(k, p_more, Ns, b_more=0.5):
    """NumPy rl_ch05_mc_np.is_estimates_prefix와 같은 계산 순서(pow, 곱, cumsum)."""
    k = torch.as_tensor(k, dtype=torch.float64)
    rho = torch.pow(torch.tensor(p_more / b_more, dtype=torch.float64), k) * ((1.0 - p_more) / (1.0 - b_more))
    cs_num = torch.cumsum(rho * k, dim=0)
    cs_den = torch.cumsum(rho, dim=0)
    idx = torch.as_tensor(Ns) - 1
    Nt = torch.as_tensor(Ns, dtype=torch.float64)
    return cs_num[idx] / Nt, cs_num[idx] / cs_den[idx]
