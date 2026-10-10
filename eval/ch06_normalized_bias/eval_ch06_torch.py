"""모델 평가 6장의 PyTorch 대응 구현. scikit-learn에는 NMBE·CV(RMSE)·NMAE가 없고 이 저장소 의존성에도 없어서,
손 계산 fixture와 함께 NumPy 구현과 같은 입력에서 값을 대조하는 용도다. MSE는 torch 내장 F.mse_loss와도 대조한다."""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)


def nmbe_ashrae(y, yhat, p=1):
    n = y.shape[0]
    return (100.0 * (y - yhat).sum() / ((n - p) * y.mean())).item()


def cv_rmse(y, yhat, p=1):
    n = y.shape[0]
    return (100.0 * torch.sqrt(((y - yhat) ** 2).sum() / (n - p)) / y.mean()).item()


def nmae(y, yhat, norm):
    return (100.0 * (y - yhat).abs().mean() / norm).item()


def mse_decomposition(y, yhat):
    e = y - yhat
    bias = e.mean()
    var = e.var(unbiased=False)  # 분모 n
    return {"mse_builtin": F.mse_loss(yhat, y).item(), "bias_sq": (bias ** 2).item(), "var": var.item()}
