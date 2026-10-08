"""모델 평가 4장의 PyTorch 대응 구현.

설계도는 sklearn.metrics.mean_absolute_percentage_error와 대조하기로 했지만 실습 저장소 의존성에
sklearn이 없어서, 1~3장처럼 torch 연산으로 같은 값을 계산해 NumPy와 대조한다. sklearn 1.9.1의 0 처리
(abs(y)를 float64 eps로 아래에서 막기)는 torch.clamp_min으로 옮기고, sklearn 문서의 예제 출력과 테스트에서 맞춘다.
"""

import torch

torch.set_num_threads(2)
EPS = torch.finfo(torch.float64).eps


def mape(y, yhat, zero="raise"):
    """NumPy 구현과 같은 정책(raise, drop, eps). 단위는 %."""
    zeros = y == 0
    if zero == "raise":
        if bool(zeros.any()):
            raise ValueError("정답에 0이 있어 MAPE가 정의되지 않는다")
        return (100.0 * ((y - yhat).abs() / y.abs()).mean()).item()
    if zero == "drop":
        keep = ~zeros
        return (100.0 * ((y[keep] - yhat[keep]).abs() / y[keep].abs()).mean()).item()
    if zero == "eps":
        return (100.0 * ((yhat - y).abs() / y.abs().clamp_min(EPS)).mean()).item()
    raise ValueError(zero)


def mae(y, yhat):
    return torch.nn.functional.l1_loss(yhat, y).item()
