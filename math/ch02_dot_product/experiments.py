"""웹 2장에 넣는 숫자를 재현해 results/ch02.json으로 저장한다.

무엇을 재는가: 손계산(2x3 + 1x4 = 10), NumPy 구현, PyTorch 구현이 같은 값 10을 내는지와, 두 행(글)의 점수 [10, 11]을 한꺼번에 계산한 결과.
입력: 코드에 고정된 특징 [2, 1], 가중치 [3, 4], 두 행 [[2, 1], [1, 2]]. 출력: results/ch02.json(manual, numpy, torch, batch_scores, all_equal).
이 파일은 numpy와 torch가 모두 필요하다.
"""

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

    manual = 2 * 3 + 1 * 4  # 손계산 값(정수 10)
    numpy_value = np_score(features, weights)
    # torch.tensor(float64 배열)에서 만든 텐서는 float64라 float32 반올림 오차 없이 정확히 10.0이 나온다
    torch_value = float(
        torch_score(torch.tensor(features), torch.tensor(weights))
    )
    scores = batch_scores(rows, weights)

    result = {
        "manual": manual,
        "numpy": numpy_value,
        "torch": torch_value,
        "batch_scores": scores.tolist(),
        "all_equal": manual == numpy_value == torch_value,  # 연쇄 비교: 세 값이 모두 같을 때만 True(정확한 정수 계산이라 == 비교가 안전)
    }
    # 결과 폴더가 없으면 만든다. ensure_ascii=False로 한글이 있어도 그대로 저장한다
    out = Path(__file__).parent / "results" / "ch02.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
