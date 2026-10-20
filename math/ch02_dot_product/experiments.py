"""웹 2장에 넣는 숫자를 재현해 results/ch02.json으로 저장한다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dot_product_np import batch_scores, weighted_score as np_score
from dot_product_torch import weighted_score as torch_score


def main() -> None:
    features = np.array([2.0, 1.0])
    weights = np.array([3.0, 4.0])
    rows = np.array([[2.0, 1.0], [1.0, 2.0]])

    manual = 2 * 3 + 1 * 4
    numpy_value = np_score(features, weights)
    torch_value = float(
        torch_score(torch.tensor(features), torch.tensor(weights))
    )
    scores = batch_scores(rows, weights)

    result = {
        "manual": manual,
        "numpy": numpy_value,
        "torch": torch_value,
        "batch_scores": scores.tolist(),
        "all_equal": manual == numpy_value == torch_value,
    }
    out = Path(__file__).parent / "results" / "ch02.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
