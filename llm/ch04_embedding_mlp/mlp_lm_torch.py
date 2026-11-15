"""4장 PyTorch 대응 구현. mlp_lm_np.py와 같은 식을 nn.Module로 옮긴다.

NumPy 쪽 가중치를 그대로 복사해 넣고, autograd가 낸 기울기가 손으로 쓴 역전파와 같은지 본다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NPLM(nn.Module):
    def __init__(self, V, m, k, h):
        super().__init__()
        self.C = nn.Embedding(V, m)       # 공유 임베딩 |V| x m
        # nn.Linear는 y = x W^T + b 꼴이라 NumPy의 H, hb와 모양이 같다((h, k*m)).
        self.H = nn.Linear(k * m, h)      # 가중치 H, 편향 d(코드의 hb)
        self.U = nn.Linear(h, V, bias=False)
        self.W = nn.Linear(k * m, V)      # 직접 연결 W, 출력 편향 b

    def forward(self, ctx_ids):
        # (B, k, m) -> (B, k*m). 문맥 자리를 이어 붙인다.
        x = self.C(ctx_ids).flatten(1)
        return self.W(x) + self.U(torch.tanh(self.H(x)))  # y = b + Wx + U tanh(d + Hx)

    @classmethod
    def from_numpy(cls, p):
        V, m = p["C"].shape
        h, km = p["H"].shape
        # km // m = k. float64로 맞춰야 NumPy 기울기와 1e-10 수준으로 비교된다.
        model = cls(V, m, km // m, h).double()
        with torch.no_grad():
            model.C.weight.copy_(torch.from_numpy(p["C"]))
            model.H.weight.copy_(torch.from_numpy(p["H"]))
            model.H.bias.copy_(torch.from_numpy(p["hb"]))
            model.U.weight.copy_(torch.from_numpy(p["U"]))
            model.W.weight.copy_(torch.from_numpy(p["W"]))
            model.W.bias.copy_(torch.from_numpy(p["b"]))
        return model

    def grads_as_numpy(self):
        """NumPy 구현과 같은 이름으로 기울기를 꺼낸다."""
        return {
            "C": self.C.weight.grad.numpy(),
            "H": self.H.weight.grad.numpy(),
            "hb": self.H.bias.grad.numpy(),
            "U": self.U.weight.grad.numpy(),
            "W": self.W.weight.grad.numpy(),
            "b": self.W.bias.grad.numpy(),
        }


def loss(model, ctx_ids, targets):
    # 로짓 y에서 log_softmax와 NLL을 한 번에. NumPy의 loss와 같은 값.
    return F.cross_entropy(model(ctx_ids), targets)  # 평균 -log softmax(y)_target
