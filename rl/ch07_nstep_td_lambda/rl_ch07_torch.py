"""강화학습 7장 PyTorch 대응 구현. 같은 판에서 NumPy 결과와 수치를 대조한다(float64).

- lambda_returns_t: 한 판의 λ-리턴을 반복문 없이 행렬 한 번으로 계산한다.
  b_t = R_{t+1} + γ(1-λ)V(S_{t+1}) (종료로 들어가면 V 항은 0)이라 두면
  G^λ_t = Σ_{k≥t} (γλ)^{k-t} b_k 이므로, 위삼각 행렬 W[t,k] = (γλ)^{k-t}를 곱하면 된다.
- n_step_returns_t: 모든 tau의 G_{tau:tau+n}을 할인 누적합과 인덱스 한 번으로 계산한다.
- lambda_return_summed_t: 판마다 위 λ-리턴으로 고칠 양을 모아 한 번에 더한다.
"""

import torch

DT = torch.float64


def _ep_tensors(ep):
    s = torch.tensor([tr[0] for tr in ep], dtype=torch.long)
    r = torch.tensor([tr[1] for tr in ep], dtype=DT)
    s2 = torch.tensor([tr[2] for tr in ep], dtype=torch.long)
    d = torch.tensor([tr[3] for tr in ep], dtype=torch.bool)
    return s, r, s2, d


def lambda_returns_t(ep, V, lam, gamma):
    s, r, s2, d = _ep_tensors(ep)
    V = torch.as_tensor(V, dtype=DT)
    T = r.shape[0]
    v_next = torch.where(d, torch.zeros((), dtype=DT), V[s2])
    # 식: b_t = R_{t+1} + γ (1-λ) V(S_{t+1})  (종료로 들어가면 v_next = 0). NumPy 재귀의 λ G^λ_{t+1} 항을 뺀 부분
    b = r + gamma * (1.0 - lam) * v_next
    k = torch.arange(T, dtype=DT)
    expo = k[None, :] - k[:, None]                       # k - t
    # 식: W[t,k] = (γλ)^(k-t) (k >= t), 그 밖은 0. G^λ = W b 는 G^λ_t = b_t + γλ G^λ_{t+1} 을 펼친 것
    # (clamp는 음수 지수에서 0^음수 같은 inf가 where 안에서 만들어지지 않게 하려는 안전장치)
    W = torch.where(expo >= 0, (gamma * lam) ** expo.clamp(min=0), torch.zeros((), dtype=DT))
    return W @ b


def n_step_returns_t(ep, V, n, gamma):
    """모든 tau에 대한 G_{tau:tau+n}. n=None이면 판 끝까지."""
    s, r, s2, d = _ep_tensors(ep)
    V = torch.as_tensor(V, dtype=DT)
    T = r.shape[0]
    disc = gamma ** torch.arange(T, dtype=DT)
    P = torch.cat([torch.zeros(1, dtype=DT), torch.cumsum(r * disc, 0)])   # P[i] = Σ_{j<i} γ^j R_{j+1}
    tau = torch.arange(T)
    end = torch.full((T,), T) if n is None else torch.clamp(tau + n, max=T)
    # 식: Σ_{j=τ}^{end-1} γ^(j-τ) R_{j+1} = (P[end] - P[τ]) / γ^τ   (disc[tau] = γ^τ)
    G = (P[end] - P[tau]) / disc
    if n is not None:
        states = torch.cat([s, s2[-1:]])                  # S_0 ... S_T
        boot = (tau + n) < T
        idx = torch.clamp(tau + n, max=T)
        # 식: + γ^n V(S_{τ+n}), 판이 끝난 tau(boot=False)는 0
        G = G + torch.where(boot, gamma ** n * V[states[idx]], torch.zeros((), dtype=DT))
    return G


def lambda_return_summed_t(episodes, n_states, lam, alpha, gamma=1.0, v0=0.0):
    """판마다 고칠 양을 index_add_로 모아 한 번에 더한다(NumPy lambda_return_summed와 대조)."""
    V = torch.full((n_states,), v0, dtype=DT)
    for ep in episodes:
        s = torch.tensor([tr[0] for tr in ep], dtype=torch.long)
        G = lambda_returns_t(ep, V, lam, gamma)
        # 식: ΔV(S_t) += α [G^λ_t - V(S_t)]. index_add_는 같은 칸의 값을 더해 주므로 중복 방문도 모두 반영된다
        dV = torch.zeros(n_states, dtype=DT).index_add_(0, s, alpha * (G - V[s]))
        V = V + dV
    return V
