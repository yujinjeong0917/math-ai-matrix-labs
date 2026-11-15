"""강화학습 5장 PyTorch 대응 구현. 같은 에피소드 묶음에서 리턴, 첫 방문 MC 평균, 중요도 비율과
일반·가중 IS 추정값을 텐서로 계산한다. float64.

허용오차: 리턴과 MC 평균은 atol=1e-12. 중요도 비율은 곱하는 순서가 NumPy(앞에서부터 하나씩)와
torch.prod(내부 순서)에서 다를 수 있고, 값이 1e4를 넘기도 해서 상대오차 rtol=1e-12로 비교한다.
"""

import torch


def pad(episodes, gamma_dtype=torch.float64):
    """끝난 판만 골라 [B, T] 텐서로 채운다. 짧은 판의 뒤쪽은 mask=False."""
    # 잘린 판(done=False)은 리턴을 모르므로 NumPy판과 같이 뺀다
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
        # 식: G_t = r_t + gamma G_{t+1}을 판 B개에 동시에. 시간 방향으로만 반복
        # 패딩 칸은 R = 0, g = 0이라 판 끝 뒤의 값이 0으로 이어져 마스크 없이도 맞다
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
        # 패딩 칸(M=False)은 s = 0으로 채워져 있다. M으로 걸러야 상태 0을 방문한 것으로 잘못 세지 않는다
        new = M[:, t] & ~seen[rows, s]
        first[:, t] = new
        seen[rows, s] = seen[rows, s] | M[:, t]
    return first


def mc_first_visit(episodes, gamma, n_states):
    """첫 방문 MC 가치 추정. 반환 (V[n_states], 방문 횟수 N[n_states]). NumPy mc_first_visit의 앞 두 값과 대응."""
    S, A, R, M = pad(episodes)
    G = returns_batch(R, gamma)
    fv = first_visit_mask(S, M, n_states)
    # 상태별 리턴 합과 방문 횟수를 index_add_로 모은다. 식: V(s) = (sum G) / N(s)
    # NumPy판의 증분형 평균과 수학적으로 같고 부동소수 계산 순서만 달라 atol 1e-12로 비교한다
    total = torch.zeros(n_states, dtype=G.dtype).index_add_(0, S[fv], G[fv])
    N = torch.zeros(n_states, dtype=torch.long).index_add_(0, S[fv], torch.ones_like(S[fv]))
    # 한 번도 못 간 상태는 0(NumPy판과 같은 규칙). clamp는 0으로 나누는 것을 막는다
    V = torch.where(N > 0, total / N.clamp(min=1), torch.zeros_like(total))
    return V, N


def importance_ratios(episodes, pi, b):
    """끝난 판마다 중요도 비율 rho = prod_t pi(a_t|s_t) / b(a_t|s_t). 반환 모양 [B]."""
    S, A, R, M = pad(episodes)
    # 식: pi(a_t|s_t) / b(a_t|s_t)를 모든 (판, t)에 한꺼번에. b[s,a] > 0이어야 한다
    # 패딩 칸(S = A = 0)에서는 0/0 같은 값이 나올 수 있으나 아래 where에서 버린다
    ratio = pi[S, A] / b[S, A]
    # 패딩 칸은 곱해도 변하지 않게 1로 바꾼다
    ratio = torch.where(M, ratio, torch.ones_like(ratio))
    # 식: rho = prod_t ratio. 판마다 시간 방향 곱(NumPy판은 앞에서부터 하나씩 곱해 순서가 다를 수 있음)
    return ratio.prod(dim=1)


def off_policy_is(episodes, pi, b, gamma):
    """시작 칸(t=0) 기준 일반·가중 IS 추정값."""
    S, A, R, M = pad(episodes)
    # G0: 각 판 시작(t=0)의 리턴. 걸음이 하나도 없으면 0
    G0 = returns_batch(R, gamma)[:, 0] if R.shape[1] else torch.zeros(R.shape[0], dtype=R.dtype)
    rho = importance_ratios(episodes, pi, b)
    # 식: 일반 IS = sum rho G0 / N, 가중 IS = sum rho G0 / sum rho. NumPy판과 달리 분모가 0일 때의 처리는 없다
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
