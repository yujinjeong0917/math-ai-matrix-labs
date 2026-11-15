"""2장: 같은 내적을 PyTorch로 대조한다.

무엇을 재는가: dot_product_np.weighted_score와 같은 내적을 torch.dot으로 계산해 두 라이브러리의 결과가 같은지 본다.
입력: 1차원 텐서 두 개(같은 길이, 같은 dtype). 출력: 0차원 텐서(점수). 이 파일은 torch가 필요하다.
"""

from __future__ import annotations

import torch


def weighted_score(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """같은 자리끼리 곱한 뒤 더한 점수(0차원 텐서). torch.dot은 두 텐서의 dtype이 같아야 하고 1차원만 받는다."""
    if features.ndim != 1 or weights.ndim != 1:
        raise ValueError("두 입력은 1차원 백터여야 합니다.")
    if features.shape != weights.shape:
        raise ValueError("특징과 가중치의 자리 수가 같아야 합니다.")
    return torch.dot(features, weights)


if __name__ == "__main__":
    x = torch.tensor([2.0, 1.0])
    w = torch.tensor([3.0, 4.0])
    print(float(weighted_score(x, w)))
