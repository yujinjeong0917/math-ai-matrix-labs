"""14장 PyTorch 대응 구현: 작은 MoE 층, 밀집 기준선, 합성 데이터, 학습 루프.

- Expert: d_in → h → d_out 두 층 MLP 한 명.
- MoE: 라우터(Linear) + 전문가 E명. 토큰마다 상위 k명만 계산한다(게이트 0인 전문가는 부르지 않는다).
  forward는 (출력, 전체 softmax 확률, 상위 k 인덱스)를 돌려준다. 게이트 방식은 두 가지다.
  gate="switch": 전체 softmax 확률 p_i(x)를 그대로 곱한다(Fedus 등 2021 식 (1)(2)). k=1에서도 라우터가 기울기를 받는다.
  gate="renorm": 상위 k개 로짓에만 다시 softmax(Shazeer 등 2017 식 (3), Mixtral의 Softmax(TopK)). k=1이면 게이트가 항상 1이라 라우터가 과제 기울기를 못 받는다.
- Dense: 같은 모양의 밀집 MLP 하나. hidden 크기만 바꿔 '활성 계산이 같은' 것과 '총 파라미터가 같은' 것을 만든다.
- make_data(seed): 8개 군집 각각이 서로 다른 선형 규칙을 따르는 합성 회귀. 군집 중심에서 규칙으로 가는 값에 잡음 없음.
- train(model, data, alpha, steps, seed): Adam. 손실 = MSE + (MoE일 때) balance_loss(Fedus 식 4).

입력: 모델 설정, 합성 데이터 텐서. 출력: 검증 MSE, 전문가별 토큰 비율 f.
기호: E = 전문가 수, k = 토큰당 전문가 수, α = 보조 손실 계수, f = 1순위 전문가별 토큰 비율.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN, D_OUT, N_CLUSTERS = 8, 4, 8


class Expert(nn.Module):
    def __init__(self, d_in, h, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class Dense(nn.Module):
    def __init__(self, h, d_in=D_IN, d_out=D_OUT):
        super().__init__()
        self.net = Expert(d_in, h, d_out)

    def forward(self, x):
        return self.net(x), None, None


class MoE(nn.Module):
    def __init__(self, E, k, h, d_in=D_IN, d_out=D_OUT, gate="switch"):
        super().__init__()
        self.E, self.k, self.gate, self.d_out = E, k, gate, d_out
        self.router = nn.Linear(d_in, E, bias=False)
        self.experts = nn.ModuleList([Expert(d_in, h, d_out) for _ in range(E)])

    def forward(self, x):
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)  # 보조 손실의 P_i용 전체 확률
        top_val, top_idx = logits.topk(self.k, dim=-1)
        if self.gate == "switch":
            gate = probs.gather(1, top_idx)  # Fedus 등 식 (2): 전체 softmax 확률 p_i(x)를 그대로
        else:
            gate = F.softmax(top_val, dim=-1)  # Shazeer 등 식 (3): 상위 k개에만 다시 softmax
        y = torch.zeros(x.shape[0], self.d_out)
        for i, ex in enumerate(self.experts):
            # 이 전문가가 뽑힌 (토큰, 자리) 쌍. 뽑히지 않은 토큰은 이 전문가에 보내지 않는다
            tok, slot = (top_idx == i).nonzero(as_tuple=True)
            if tok.numel():
                y = y.index_add(0, tok, gate[tok, slot][:, None] * ex(x[tok]))
        return y, probs, top_idx


def balance_loss(probs, top_idx, E, alpha):
    """Fedus 등(2021) 식 (4)~(6). f는 1순위 전문가 비율(미분 안 됨), P는 확률 평균(미분됨)."""
    f = torch.bincount(top_idx[:, 0], minlength=E).float() / probs.shape[0]
    P = probs.mean(dim=0)
    return alpha * E * (f * P).sum()


def make_data(seed, n_train=4096, n_test=2048):
    g = torch.Generator().manual_seed(seed)
    centers = 3.0 * torch.randn(N_CLUSTERS, D_IN, generator=g)
    rules = torch.randn(N_CLUSTERS, D_OUT, D_IN, generator=g) / np.sqrt(D_IN)

    def draw(n):
        c = torch.randint(0, N_CLUSTERS, (n,), generator=g)
        x = centers[c] + 0.5 * torch.randn(n, D_IN, generator=g)
        y = torch.einsum("nod,nd->no", rules[c], x)
        return x, y, c

    return draw(n_train), draw(n_test)


def train(model, data, alpha=0.0, steps=800, seed=0, lr=3e-3, batch=256):
    (xtr, ytr, _), (xte, yte, _) = data
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        b = torch.randint(0, xtr.shape[0], (batch,))
        y, probs, top = model(xtr[b])
        loss = F.mse_loss(y, ytr[b])
        if probs is not None and alpha > 0:
            loss = loss + balance_loss(probs, top, model.E, alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        y, probs, top = model(xte)
        mse = F.mse_loss(y, yte).item()
        f = None
        if top is not None:
            f = (torch.bincount(top[:, 0], minlength=model.E).float() / xte.shape[0]).numpy()
    return mse, f
