"""1장 PyTorch 대응 구현: 같은 (데이터, 초기값, 배치 순서)를 받아 같은 가중치를 내는지 본다.

경사는 손으로 쓰지 않고 autograd로 구한다. 무작위는 NumPy 쪽에서 미리 뽑아 넘긴다.
"""

import torch
import torch.nn.functional as F


def fit_sgd(X, y, w0, b0, orders, batch, lr):
    # 입력 배열의 dtype을 따라 float64/float32 중 하나를 쓴다
    dt = torch.float64 if X.dtype.name == "float64" else torch.float32
    X = torch.as_tensor(X, dtype=dt)
    y = torch.as_tensor(y, dtype=dt)
    w = torch.as_tensor(w0, dtype=dt).clone().requires_grad_(True)
    b = torch.tensor(float(b0), dtype=dt, requires_grad=True)
    for order in orders:
        order = torch.as_tensor(order)
        for start in range(0, len(order), batch):
            j = order[start:start + batch]
            # 로짓을 직접 받는 수치 안정형 교차엔트로피. 배치 평균을 돌려준다
            loss = F.binary_cross_entropy_with_logits(X[j] @ w + b, y[j])
            # NumPy 쪽 손계산 기울기(p - y)와 같은 값이 나오는지 대조하는 것이 이 파일의 목적
            gw, gb = torch.autograd.grad(loss, (w, b))
            with torch.no_grad():
                w -= lr * gw
                b -= lr * gb
    return w.detach().numpy(), float(b.detach())


def seeded_split(n, test_frac, seed):
    """torch.manual_seed로 같은 분할을 다시 낼 수 있는지 보는 용도."""
    # 전역이 아닌 독립 생성기에 시드를 준다. NumPy 분할과 같은 인덱스가 나온다는 보장은 없다(별도 난수 알고리즘)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    return idx[n_test:].numpy(), idx[:n_test].numpy()
