"""9장 PyTorch 대응 구현: 같은 입력에서 MMD^2_u가 NumPy와 같은 값인지 본다(float64, torch.cdist)."""

import torch


def mmd2_unbiased(x, y, sigma):
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    g = 2.0 * sigma * sigma
    kxx = torch.exp(-torch.cdist(x, x).pow(2) / g)
    kyy = torch.exp(-torch.cdist(y, y).pow(2) / g)
    kxy = torch.exp(-torch.cdist(x, y).pow(2) / g)
    m, n = x.shape[0], y.shape[0]
    sxx = kxx.sum() - kxx.diagonal().sum()
    syy = kyy.sum() - kyy.diagonal().sum()
    return float(sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * kxy.sum() / (m * n))
