"""모델 평가 7장의 PyTorch 대응 구현. scikit-learn은 이 저장소 의존성에 없어서(6장과 같은 사정),
손 계산 fixture와 함께 NumPy 구현과 같은 입력에서 지표 여섯 개를 대조하는 용도다.
MAE와 MSE는 torch 내장 F.l1_loss, F.mse_loss로 계산한다."""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)


def metrics(y_train, y, yhat):
    """y_train, y, yhat: float64 텐서. 반환: {지표: float}."""
    e = y - yhat
    mae = F.l1_loss(yhat, y)
    mse = F.mse_loss(yhat, y)
    scale = (y_train[1:] - y_train[:-1]).abs().mean()
    return {"mae": mae.item(), "rmse": torch.sqrt(mse).item(),
            "r2": (1.0 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()).item(),
            "mape": (100.0 * (e.abs() / y.abs()).mean()).item(),
            "mase": (mae / scale).item(),
            "nmbe": (100.0 * e.sum() / (y.shape[0] * y.mean())).item()}
