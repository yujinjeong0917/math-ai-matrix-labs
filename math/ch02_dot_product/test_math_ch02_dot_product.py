"""2장 테스트(pytest 형식). 손계산 값 10을 NumPy와 PyTorch가 모두 정확히 재현하는지, 행 단위 점수 [10, 11]이 맞는지,
길이가 다른 입력이 거부되는지 확인한다. numpy, torch, pytest가 모두 필요하다."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dot_product_np import batch_scores, weighted_score as np_score
from dot_product_torch import weighted_score as torch_score


def test_hand_numpy_torch_match() -> None:
    x = np.array([2.0, 1.0])
    w = np.array([3.0, 4.0])
    expected = 2 * 3 + 1 * 4  # 손계산 10
    # 정수 값으로만 이뤄진 곱셈·덧셈이라 부동소수점 오차 없이 == 비교가 맞는다
    assert np_score(x, w) == expected
    assert float(torch_score(torch.tensor(x), torch.tensor(w))) == expected


def test_two_posts_are_scored_row_by_row() -> None:
    rows = np.array([[2.0, 1.0], [1.0, 2.0]])
    # 첫 글 2x3 + 1x4 = 10, 둘째 글 1x3 + 2x4 = 11. assert_allclose는 배열을 상대 오차(기본 1e-7) 안에서 비교한다
    np.testing.assert_allclose(batch_scores(rows, np.array([3.0, 4.0])), [10.0, 11.0])


def test_mismatched_lengths_are_rejected() -> None:
    # 오류 메시지에 "자리 수"가 들어 있는지 정규식 검색(match)으로 확인한다. 메시지를 바꾸면 이 테스트도 고쳐야 한다
    with pytest.raises(ValueError, match="자리 수"):
        np_score(np.array([2.0, 1.0]), np.array([3.0]))
