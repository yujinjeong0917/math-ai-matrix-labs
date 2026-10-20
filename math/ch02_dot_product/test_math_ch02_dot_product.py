from __future__ import annotations

import numpy as np
import pytest
import torch

from dot_product_np import batch_scores, weighted_score as np_score
from dot_product_torch import weighted_score as torch_score


def test_hand_numpy_torch_match() -> None:
    x = np.array([2.0, 1.0])
    w = np.array([3.0, 4.0])
    expected = 2 * 3 + 1 * 4
    assert np_score(x, w) == expected
    assert float(torch_score(torch.tensor(x), torch.tensor(w))) == expected


def test_two_posts_are_scored_row_by_row() -> None:
    rows = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(batch_scores(rows, np.array([3.0, 4.0])), [10.0, 11.0])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="자리 수"):
        np_score(np.array([2.0, 1.0]), np.array([3.0]))
