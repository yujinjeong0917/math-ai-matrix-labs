"""모델 평가 2장의 PyTorch 대응 구현.

sklearn이 실습 저장소 의존성에 없어서, 1장처럼 torch 내장 손실(l1_loss, mse_loss)로
분자(RMSE)와 분모(MAE)를 따로 계산해 NumPy 비율과 대조한다.
"""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)


def rmse_mae_ratio(y, yhat):
    m = F.l1_loss(yhat, y)
    if m.item() == 0.0:
        raise ValueError("MAE가 0이라 비율이 정의되지 않는다")
    return (torch.sqrt(F.mse_loss(yhat, y)) / m).item()
