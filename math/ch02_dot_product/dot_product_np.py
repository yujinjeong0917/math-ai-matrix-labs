"""2장: 내적을 가중합으로 계산하는 NumPy 최소 구현."""

from __future__ import annotations

import numpy as np


def weighted_score(features: np.ndarray, weights: np.ndarray) -> float:
    """같은 자리끼리 곱한 뒤 더한다."""
    features = np.asarray(features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if features.ndim != 1 or weights.ndim != 1:
        raise ValueError("두 입력은 1차원 벡터여야 합니다.")
    if features.shape != weights.shape:
        raise ValueError("특징과 가중치의 자리 수가 같아야 합니다.")
    return float(np.dot(features, weights))


def batch_scores(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """여러 행에 같은 가중치를 적용한다."""
    rows = np.asarray(rows, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[1] != weights.shape[0]:
        raise ValueError("각 행의 특징 수와 가중치 수가 같아야 합니다.")
    return rows @ weights


if __name__ == "__main__":
    x = np.array([2.0, 1.0])
    w = np.array([3.0, 4.0])
    print(weighted_score(x, w))
