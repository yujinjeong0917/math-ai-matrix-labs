"""2장: 같은 내적을 PyTorch로 대조한다."""

from __future__ import annotations

import torch


def weighted_score(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if features.ndim != 1 or weights.ndim != 1:
        raise ValueError("두 입력은 1차원 백터여야 합니다.")
    if features.shape != weights.shape:
        raise ValueError("특징과 가중치의 자리 수가 같아야 합니다.")
    return torch.dot(features, weights)


if __name__ == "__main__":
    x = torch.tensor([2.0, 1.0])
    w = torch.tensor([3.0, 4.0])
    print(float(weighted_score(x, w)))
