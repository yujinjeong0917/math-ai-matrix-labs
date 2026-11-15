"""8장 · 확률과 기댓값: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 확률분포의 합이 1임을 확인하고 기댓값 E[X] = 값 x 확률의 합을 구한다.
입력 없음. 값 v = [4, 0], 확률 p = [0.25, 0.75]가 고정돼 있다.
출력: {"probability_sum": 1.0, "expectation": 1.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch08.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 확률은 음수가 없고 합이 1이어야 분포다. 기댓값 = 4 x 0.25 + 0 x 0.75 = 1.0(내적과 같은 계산)
    v=np.array([4.,0.]); p=np.array([.25,.75])
    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch08.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
