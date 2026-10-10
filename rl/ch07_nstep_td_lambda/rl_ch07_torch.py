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
    b = r + gamma * (1.0 - lam) * v_next
    k = torch.arange(T, dtype=DT)
    expo = k[None, :] - k[:, None]                       # k - t
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
    G = (P[end] - P[tau]) / disc
    if n is not None:
        states = torch.cat([s, s2[-1:]])                  # S_0 ... S_T
        boot = (tau + n) < T
        idx = torch.clamp(tau + n, max=T)
        G = G + torch.where(boot, gamma ** n * V[states[idx]], torch.zeros((), dtype=DT))
    return G


def lambda_return_summed_t(episodes, n_states, lam, alpha, gamma=1.0, v0=0.0):
    """판마다 고칠 양을 index_add_로 모아 한 번에 더한다(NumPy lambda_return_summed와 대조)."""
    V = torch.full((n_states,), v0, dtype=DT)
    for ep in episodes:
        s = torch.tensor([tr[0] for tr in ep], dtype=torch.long)
        G = lambda_returns_t(ep, V, lam, gamma)
        dV = torch.zeros(n_states, dtype=DT).index_add_(0, s, alpha * (G - V[s]))
        V = V + dV
    return V
