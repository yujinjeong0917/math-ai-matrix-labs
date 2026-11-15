"""강화학습 11장 PyTorch 대응 구현. float64.

- PPO 클리핑 손실과 그 기울기를 autograd로 구해 NumPy의 손으로 쓴 기울기와 대조한다(rtol=1e-9).
- 평균 KL의 헤세 행렬을 autograd로 두 번 미분해 구한 피셔-벡터 곱이 NumPy의 J^T M J v와 같은지 본다.
- PPO 갱신(같은 묶음, 같은 미니배치 순서, torch.optim.Adam)이 NumPy 갱신과 같은 매개변수에 닿는지 본다.
  판을 모으는 일(환경과 행동 표본)은 NumPy 쪽 함수를 그대로 써서 두 구현이 같은 데이터를 보게 한다.
"""

import numpy as np
import torch

import rl_ch11_ppo_np as m


def log_prob(w, phi, a):
    z = phi @ w
    # a=1이면 log p, 아니면 log(1-p) = logsigmoid(-z). NumPy log_prob과 같은 값
    return torch.where(a == 1, torch.nn.functional.logsigmoid(z), torch.nn.functional.logsigmoid(-z))


def ppo_clip_loss(logp_new, logp_old, adv, eps):
    r = torch.exp(logp_new - logp_old)
    # 식: -L^CLIP. autograd가 min/clamp의 기울기를 처리하므로 NumPy의 손으로 쓴 live 마스크와 대조된다
    return -torch.min(r * adv, torch.clamp(r, 1 - eps, 1 + eps) * adv).mean()


def loss_and_grad_w(w_np, phi, a, logp_old, adv, eps):
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    loss = ppo_clip_loss(log_prob(w, torch.tensor(phi), torch.tensor(a)), torch.tensor(logp_old), torch.tensor(adv), eps)
    loss.backward()
    return float(loss.detach()), w.grad.numpy().copy()


def mean_kl(w_old, w, phi):
    """평균 KL(pi_old || pi_w), 베르누이 정책."""
    z_old, z = phi @ w_old, phi @ w
    lo = torch.stack([torch.nn.functional.logsigmoid(-z_old), torch.nn.functional.logsigmoid(z_old)], 1)
    ln = torch.stack([torch.nn.functional.logsigmoid(-z), torch.nn.functional.logsigmoid(z)], 1)
    # KL(old||new) = sum_a p_old (log p_old - log p_new), 상태 평균
    return (lo.exp() * (lo - ln)).sum(1).mean()


def fisher_vector_product(w_np, phi_np, v_np):
    """KL을 두 번 미분하는 일반 방법(TRPO 부록 C.1 마지막 문단). w = w_old에서 계산."""
    w_old = torch.tensor(w_np, dtype=torch.float64)
    w = w_old.clone().requires_grad_(True)
    phi, v = torch.tensor(phi_np), torch.tensor(v_np)
    kl = mean_kl(w_old, w, phi)
    # 1차 기울기를 그래프를 남긴 채 구하고(create_graph) 아래에서 (g . v)를 다시 미분: 헤세-벡터 곱 H v = d(g.v)/dw. w = w_old에서 KL의 기울기는 0이지만 두 번째 미분은 F
    (g,) = torch.autograd.grad(kl, w, create_graph=True)
    (hv,) = torch.autograd.grad(g @ v, w)
    return hv.numpy()


def ppo_update(w_np, batch, adv, eps, epochs, n_minibatch, lr, rng):
    """NumPy의 ppo_update와 같은 일을 torch.optim.Adam으로. rng는 같은 미니배치 순서를 뽑는 데만 쓴다."""
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    # NumPy Adam과 같은 기본 betas(0.9, 0.999), eps 1e-8. 미니배치 순서를 같은 rng로 뽑아 같은 갱신을 재현
    opt = torch.optim.Adam([w], lr=lr)
    phi, a = torch.tensor(batch["phi"]), torch.tensor(batch["a"])
    logp_old, A = torch.tensor(batch["logp_old"]), torch.tensor(adv)
    for _ in range(epochs):
        perm = rng.permutation(adv.size)
        for mb in np.array_split(perm, n_minibatch):
            idx = torch.tensor(mb)
            loss = ppo_clip_loss(log_prob(w, phi[idx], a[idx]), logp_old[idx], A[idx], eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return w.detach().numpy().copy()


def train_ppo(seed, n_iters, n_episodes=16, eps=0.2, epochs=10, n_minibatch=4, adam_lr=0.05):
    """학습 루프 전체의 PyTorch 판. 판 모으기는 NumPy 함수를 쓰고 갱신만 torch로 한다.
    NumPy 판(m.train("ppo", ...))처럼 Adam 상태 하나를 학습 내내 이어 쓴다."""
    rng = np.random.default_rng(seed)
    rng_mb = np.random.default_rng([seed, 1])
    w = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    # NumPy train과 같이 Adam 상태(모멘트)를 갱신 사이에서 이어 쓴다(매 갱신마다 새로 만들지 않음)
    opt = torch.optim.Adam([w], lr=adam_lr)
    rets, ws = [], []
    for _ in range(n_iters):
        batch = m.collect(w.detach().numpy().copy(), n_episodes, rng)
        adv = m.advantages(batch)
        phi, a = torch.tensor(batch["phi"]), torch.tensor(batch["a"])
        logp_old, A = torch.tensor(batch["logp_old"]), torch.tensor(adv)
        for _ in range(epochs):
            perm = rng_mb.permutation(adv.size)
            for mb in np.array_split(perm, n_minibatch):
                idx = torch.tensor(mb)
                loss = ppo_clip_loss(log_prob(w, phi[idx], a[idx]), logp_old[idx], A[idx], eps)
                opt.zero_grad()
                loss.backward()
                opt.step()
        rets.append(float(batch["lengths"].mean()))
        ws.append(w.detach().numpy().copy())
    return rets, np.array(ws)
