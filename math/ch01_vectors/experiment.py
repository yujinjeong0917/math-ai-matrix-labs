"""1장 · 벡터: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    a=np.array([2,1,3]); b=np.array([1,2,1])
    return {'sum': (a+b).tolist(), 'dimension': int(a.size)}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch01.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
