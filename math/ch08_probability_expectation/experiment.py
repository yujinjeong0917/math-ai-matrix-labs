"""8장 · 확률과 기댓값: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    v=np.array([4.,0.]); p=np.array([.25,.75])
    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch08.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
