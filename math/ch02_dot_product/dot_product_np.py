"""2장: 내적을 가중합으로 계산하는 NumPy 최소 구현.

무엇을 재는가: 특징 벡터 x와 가중치 벡터 w의 내적 sum(x_i w_i), 곧 "같은 자리끼리 곱해서 더한" 점수. 예: x = [2, 1], w = [3, 4]이면 2x3 + 1x4 = 10.
입력: 1차원 벡터 두 개(길이가 같아야 함), 또는 (행 수, 특징 수) 모양의 2차원 배열과 (특징 수,) 모양 가중치. 출력: 점수(float) 또는 행마다의 점수 배열.
장 본문의 두 글 점수 손계산(10, 11)과 experiments.py, 테스트가 같은 숫자로 대응한다. 이 파일은 numpy가 필요하다.
"""

from __future__ import annotations

import numpy as np


def weighted_score(features: np.ndarray, weights: np.ndarray) -> float:
    """같은 자리끼리 곱한 뒤 더한다."""
    # 정수 리스트가 들어와도 float 배열로 통일한다(정수 곱셈 오버플로나 dtype 혼용을 피한다)
    features = np.asarray(features, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if features.ndim != 1 or weights.ndim != 1:
        raise ValueError("두 입력은 1차원 벡터여야 합니다.")
    # 모양이 다르면 numpy가 조용히 브로드캐스트하거나 다른 오류를 내므로 먼저 막는다. 테스트가 이 메시지의 "자리 수"를 확인한다
    if features.shape != weights.shape:
        raise ValueError("특징과 가중치의 자리 수가 같아야 합니다.")
    return float(np.dot(features, weights))


def batch_scores(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """여러 행에 같은 가중치를 적용한다."""
    rows = np.asarray(rows, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[1] != weights.shape[0]:
        raise ValueError("각 행의 특징 수와 가중치 수가 같아야 합니다.")
    # (행 수, 특징 수) @ (특징 수,) -> (행 수,): 행마다 내적 한 번씩. 예: [[2, 1], [1, 2]] @ [3, 4] = [10, 11]
    return rows @ weights


if __name__ == "__main__":
    x = np.array([2.0, 1.0])
    w = np.array([3.0, 4.0])
    print(weighted_score(x, w))
