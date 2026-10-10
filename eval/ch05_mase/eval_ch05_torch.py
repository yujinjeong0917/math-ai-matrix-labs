"""모델 평가 5장의 PyTorch 대응 구현. scikit-learn에는 MASE가 없어서(설계도) 손 계산 fixture와 함께
NumPy 구현과 같은 입력에서 값을 대조하는 용도다."""

import torch

torch.set_num_threads(2)


def naive_mae_in_sample(y_train, lag=1):
    d = (y_train[lag:] - y_train[:-lag]).abs().mean()
    if d.item() == 0.0:
        raise ValueError("학습 구간의 naive 오차가 0이다")
    return d


def mase(y_train, y_test, yhat, lag=1):
    return ((y_test - yhat).abs() / naive_mae_in_sample(y_train, lag)).mean().item()
