"""강화학습 8장 PyTorch 대응 구현. 같은 스윕을 autograd로 계산해 NumPy와 대조한다(float64).

손실 L(w) = 1/2 * sum_s (r_s + gamma * V(s') - V(s))^2 를 두고 w <- w - alpha * dL/dw 를 한다.
- 목표 r + gamma V(s')에 detach()를 걸면 기울기가 V(s) 쪽으로만 흘러 반경사 TD(0)가 되고
  (Baird 식 7: dw = alpha * sum delta * phi(s)),
- detach()를 빼면 목표 쪽으로도 기울기가 흘러 잔차 경사가 된다
  (Baird 식 8: dw = -alpha * sum delta * (gamma phi(s') - phi(s))).
코드 한 단어 차이가 갱신 규칙을 바꾼다는 점이 이 파일의 요점이다.
"""

import torch

DT = torch.float64


def epoch_step(w, Phi, next_state, R, alpha, gamma, detach_target=True):
    """한 스윕(epoch-wise). w: (d,) 텐서. 새 w를 돌려준다(그래프 없이)."""
    w = w.detach().clone().requires_grad_(True)
    v = Phi @ w
    target = R + gamma * v[next_state]
    if detach_target:
        target = target.detach()
    loss = 0.5 * ((target - v) ** 2).sum()
    (g,) = torch.autograd.grad(loss, w)
    return (w - alpha * g).detach()


def run_sweeps(Phi, next_state, R, w0, alpha, gamma, sweeps, detach_target=True):
    Phi = torch.as_tensor(Phi, dtype=DT)
    R = torch.as_tensor(R, dtype=DT)
    nxt = torch.as_tensor(next_state, dtype=torch.long)
    w = torch.as_tensor(w0, dtype=DT)
    out = [w.clone()]
    for _ in range(sweeps):
        w = epoch_step(w, Phi, nxt, R, alpha, gamma, detach_target)
        out.append(w.clone())
    return torch.stack(out)
