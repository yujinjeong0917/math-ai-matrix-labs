"""모델 평가 3장의 PyTorch 대응 구현.

설계도는 sklearn.metrics.r2_score와 대조하기로 했지만 실습 저장소 의존성에 sklearn이 없어서,
1·2장처럼 torch 내장 연산(F.mse_loss, torch.corrcoef, torch.linalg.lstsq)으로 같은 값을 계산해 NumPy와 대조한다.
"""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)


def r2(y, yhat):
    """1 - MSE(모델) / MSE(평균 찍기). 분자·분모에 같은 n이 곱해져 있어 SSE/SST와 같다."""
    base = F.mse_loss(torch.full_like(y, y.mean().item()), y)
    if base.item() == 0.0:
        raise ValueError("정답이 모두 같아 R^2가 정의되지 않는다")
    return (1.0 - F.mse_loss(yhat, y) / base).item()


def squared_corr(y, yhat):
    return (torch.corrcoef(torch.stack([y, yhat]))[0, 1] ** 2).item()


def fit_line(x, y):
    """설계행렬 [1, x]에 lstsq. (b0, b1)."""
    X = torch.stack([torch.ones_like(x), x], dim=1)
    b = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    return b[0].item(), b[1].item()
