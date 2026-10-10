"""강화학습 9장 PyTorch 대응 구현. float64.

NumPy 쪽(rl_ch09_dqn_np.py)은 역전파를 손으로 적었다. 여기서는 같은 가중치, 같은 미니배치로
    L = mean_j 0.5 * (y_j - Q(s_j, a_j))^2
를 만들고 autograd로 미분한다. 목표 y는 타깃망(또는 Double 규칙)으로 만든 상수라서 torch.no_grad 안에서 계산한다.
같은 입력에서 손실과 기울기가 rtol=1e-6 안에서 같아야 한다(실제 차이는 1e-15 수준).
"""

import numpy as np
import torch

SCALE = torch.tensor([2.4, 3.0, 0.21, 3.0], dtype=torch.float64)


class TorchMLP(torch.nn.Module):
    def __init__(self, Ws, bs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for W, b in zip(Ws, bs):
            lin = torch.nn.Linear(W.shape[0], W.shape[1]).double()
            with torch.no_grad():
                lin.weight.copy_(torch.tensor(W.T))   # nn.Linear는 (출력, 입력) 모양으로 저장
                lin.bias.copy_(torch.tensor(b))
            self.layers.append(lin)

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def td_target(net, tgt, R, S2, F, gamma, double):
    with torch.no_grad():
        q2 = tgt(S2 / SCALE)
        if double:
            a_star = net(S2 / SCALE).argmax(dim=1)
            nxt = q2.gather(1, a_star[:, None]).squeeze(1)
        else:
            nxt = q2.max(dim=1).values
        return R + gamma * (1.0 - F) * nxt


def loss_and_grads(np_net, np_tgt, batch, gamma=0.99, double=False):
    """NumPy MLP 두 개(지금 망, 타깃망)의 가중치를 복사해 손실과 기울기를 autograd로 계산한다.
    반환 순서는 NumPy의 grad()와 같다: [W1, W2, W3, b1, b2, b3] (W는 NumPy 모양 (입력, 출력))."""
    S, A, R, S2, F = (torch.tensor(np.asarray(x)) for x in batch)
    S, R, S2, F = S.double(), R.double(), S2.double(), F.double()
    A = A.long()
    net = TorchMLP(np_net.W, np_net.b)
    tgt = TorchMLP(np_tgt.W, np_tgt.b)
    y = td_target(net, tgt, R, S2, F, gamma, double)
    q = net(S / SCALE).gather(1, A[:, None]).squeeze(1)
    loss = 0.5 * ((y - q) ** 2).mean()
    loss.backward()
    gW = [lin.weight.grad.T.numpy().copy() for lin in net.layers]
    gb = [lin.bias.grad.numpy().copy() for lin in net.layers]
    return float(loss.detach()), gW + gb, y.numpy()
