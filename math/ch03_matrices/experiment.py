"""3장 · 행렬: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 행렬(행 = 글 하나, 열 = 특징)에 가중치 벡터를 곱해 행마다 점수를 한꺼번에 만든다.
입력 없음. X = [[2, 1], [1, 2]], w = [3, 4]가 코드에 고정돼 있다.
출력: {"scores": [10, 11], "shape": [2, 2]}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch03.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # (2, 2) 행렬 @ (2,) 벡터 -> (2,) 점수. 행마다 내적: 2x3 + 1x4 = 10, 1x3 + 2x4 = 11
    X=np.array([[2,1],[1,2]]); w=np.array([3,4])
    return {'scores': (X@w).tolist(), 'shape': list(X.shape)}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch03.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
